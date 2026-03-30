import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack
from callbacks import EmbeddingAnalysisCallback


def init_clearml(cfg):
    """Initialize ClearML task and handle dataset download if configured.

    Returns the ClearML Task object (or None if ClearML is disabled).
    """
    clearml_cfg = cfg.get("clearml", {})
    if not clearml_cfg.get("enabled", False):
        return None

    try:
        from clearml import Task, Dataset
    except ImportError:
        print("[clearml] clearml package not installed, skipping. "
              "Install with: pip install clearml")
        return None

    task = Task.init(
        project_name=clearml_cfg.get("project", "lewm"),
        task_name=clearml_cfg.get("task_name", "lewm-train"),
        task_type=Task.TaskTypes.training,
        auto_connect_frameworks={"pytorch": True},
    )

    # Connect the full Hydra config so it's visible/editable in the ClearML UI
    task.connect(OmegaConf.to_container(cfg, resolve=True), name="hydra_config")

    # Apply tags
    tags = clearml_cfg.get("tags", [])
    if tags:
        task.add_tags(list(tags))

    # Auto-download ClearML Dataset if dataset_id is set
    dataset_id = clearml_cfg.get("dataset_id")
    if dataset_id:
        print(f"[clearml] Downloading dataset {dataset_id} ...")
        ds = Dataset.get(dataset_id=dataset_id)
        local_path = ds.get_local_copy()
        os.environ["STABLEWM_HOME"] = local_path
        print(f"[clearml] Dataset available at {local_path}")

    return task


def upload_clearml_model(task, run_dir, model_name):
    """Upload the final model checkpoint as a ClearML output model."""
    if task is None:
        return

    # Find the latest checkpoint
    ckpts = sorted(run_dir.glob(f"{model_name}*_object.ckpt"))
    if ckpts:
        latest = ckpts[-1]
        task.update_output_model(
            model_path=str(latest),
            model_name=model_name,
        )
        print(f"[clearml] Uploaded model: {latest}")

    weights_ckpt = run_dir / f"{model_name}_weights.ckpt"
    if weights_ckpt.exists():
        task.upload_artifact(
            name=f"{model_name}_weights",
            artifact_object=str(weights_ckpt),
        )
        print(f"[clearml] Uploaded weights: {weights_ckpt}")


def load_vjepa_encoder(checkpoint_path, patch_size=16, img_size=224):
    """Load a VJEPA encoder from checkpoint.

    Attempts to load from the VJEPA codebase (facebookresearch/jepa).
    Falls back to loading a raw state_dict into a ViT if the VJEPA package
    is not available.
    """
    try:
        from jepa_src.models.vision_transformer import vit_base, vit_large, vit_huge
    except ImportError:
        try:
            from timm.models.vision_transformer import VisionTransformer
        except ImportError:
            raise ImportError(
                "VJEPA encoder requires either the facebookresearch/jepa package "
                "or timm. Install one of them to use encoder_type=vjepa."
            )
        # Fallback: build a ViT via timm and load weights
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("target_encoder", ckpt.get("model", ckpt))
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Detect architecture from checkpoint weights
        embed_key = [k for k in state_dict if "patch_embed" in k and "weight" in k]
        embed_dim = int(state_dict[embed_key[0]].shape[0]) if embed_key else 768
        depth = len({k.split(".")[1] for k in state_dict if k.startswith("blocks.")})
        # Standard ViT head counts: B=12, L=16, H=16
        num_heads = {768: 12, 1024: 16, 1280: 16}.get(embed_dim, 12)

        encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth or 12,
            num_heads=num_heads,
            mlp_ratio=4.0,
            num_classes=0,  # no classification head
        )
        encoder.load_state_dict(state_dict, strict=False)
        encoder.embed_dim = embed_dim
        return encoder

    # Infer model scale from checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("target_encoder", ckpt.get("model", ckpt))

    # Detect hidden dim from patch_embed weight shape
    embed_key = [k for k in state_dict if "patch_embed" in k and "weight" in k]
    if embed_key:
        embed_dim = state_dict[embed_key[0]].shape[0]
    else:
        embed_dim = 768

    model_fn = {768: vit_base, 1024: vit_large, 1280: vit_huge}.get(embed_dim, vit_base)
    encoder = model_fn(patch_size=patch_size, tubelet_size=1)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    encoder.load_state_dict(state_dict, strict=False)
    return encoder


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # --- Losses ---
    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_loss = self.sigreg(emb.transpose(0, 1))
    loss = pred_loss + lambd * sigreg_loss

    output["pred_loss"] = pred_loss
    output["sigreg_loss"] = sigreg_loss
    output["loss"] = loss

    # --- Detailed metrics ---
    with torch.no_grad():
        # Per-step prediction error (if multiple prediction steps)
        per_step_mse = (pred_emb - tgt_emb).pow(2).mean(dim=(0, 2))  # (T,)
        for t in range(per_step_mse.shape[0]):
            output[f"pred_mse_step_{t}"] = per_step_mse[t]

        # Cosine similarity between pred and target embeddings
        cos_sim = torch.nn.functional.cosine_similarity(
            pred_emb.reshape(-1, pred_emb.shape[-1]),
            tgt_emb.reshape(-1, tgt_emb.shape[-1]),
            dim=-1,
        ).mean()
        output["pred_cosine_sim"] = cos_sim

        # Embedding statistics (collapse detection)
        emb_flat = emb.reshape(-1, emb.shape[-1])  # (B*T, D)
        output["emb_mean_norm"] = emb_flat.norm(dim=-1).mean()
        output["emb_var"] = emb_flat.var(dim=0).mean()  # avg per-dim variance
        output["emb_std"] = emb_flat.std(dim=0).mean()

        # Effective rank of embeddings (entropy of singular values)
        # Approximated on current batch — cast to float32 for SVD
        _, S, _ = torch.linalg.svd((emb_flat - emb_flat.mean(0)).float(), full_matrices=False)
        S_norm = S / S.sum()
        eff_rank = torch.exp(-(S_norm * torch.log(S_norm + 1e-10)).sum())
        output["emb_effective_rank"] = eff_rank

        # Action embedding stats
        act_flat = act_emb.reshape(-1, act_emb.shape[-1])
        output["act_emb_mean_norm"] = act_flat.norm(dim=-1).mean()
        output["act_emb_var"] = act_flat.var(dim=0).mean()

        # Prediction embedding stats
        pred_flat = pred_emb.reshape(-1, pred_emb.shape[-1])
        output["pred_emb_mean_norm"] = pred_flat.norm(dim=-1).mean()

        # Loss ratio (pred vs sigreg contribution)
        output["loss_ratio_pred_over_sigreg"] = pred_loss / (lambd * sigreg_loss + 1e-8)

        # Gradient-free L1 prediction error
        output["pred_l1"] = (pred_emb - tgt_emb).abs().mean()

        # Copy baseline: MSE of just predicting the input frame (no dynamics)
        # ctx_emb[:, -1:] is the last context frame, tgt_emb is the target
        copy_baseline = (ctx_emb[:, -1:].expand_as(tgt_emb) - tgt_emb).pow(2).mean()
        output["copy_baseline_mse"] = copy_baseline
        # Ratio: pred_loss / copy_baseline — <1 means model beats copying, >1 means worse
        output["pred_over_copy_ratio"] = pred_loss / (copy_baseline + 1e-8)

    # Log everything via Lightning
    is_val = stage in ("val", "validate")
    on_epoch = is_val
    log_dict = {}
    for k, v in output.items():
        if torch.is_tensor(v) and v.ndim == 0:
            log_dict[f"{stage}/{k}"] = v.detach()

    self.log_dict(log_dict, on_step=True, on_epoch=on_epoch, sync_dist=True, prog_bar=False)

    # Progress bar — use distinct keys to avoid duplicate log error
    self.log(f"loss/{stage}", loss.detach(), prog_bar=True, on_step=True, on_epoch=on_epoch, sync_dist=True)
    self.log(f"pred/{stage}", pred_loss.detach(), prog_bar=True, on_step=True, on_epoch=on_epoch, sync_dist=True)
    self.log(f"sreg/{stage}", sigreg_loss.detach(), prog_bar=True, on_step=True, on_epoch=on_epoch, sync_dist=True)

    # Report scalars directly to ClearML (Lightning's log_dict isn't auto-captured)
    # Each metric gets its own plot (title), with train/validate as separate series
    try:
        from clearml import Task
        task = Task.current_task()
        if task is not None:
            logger = task.get_logger()
            step = self.global_step
            for k, v in log_dict.items():
                parts = k.split("/", 1)
                series = parts[0]  # train or validate
                metric = parts[1] if len(parts) > 1 else parts[0]
                logger.report_scalar(title=metric, series=series, value=v.item(), iteration=step)
    except Exception:
        pass

    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       clearml       ##
    #########################

    clearml_task = init_clearml(cfg)

    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder_type = cfg.get("encoder_type", "vit_hf")
    encoder_frozen = cfg.get("encoder_frozen", False)
    encoder_pooling = cfg.get("encoder_pooling", "cls")

    if encoder_type == "vjepa":
        assert cfg.vjepa_checkpoint is not None, (
            "vjepa_checkpoint must be set when encoder_type=vjepa"
        )
        encoder = load_vjepa_encoder(
            cfg.vjepa_checkpoint,
            patch_size=cfg.patch_size,
            img_size=cfg.img_size,
        )
        hidden_dim = encoder.embed_dim
        encoder_pooling = "mean"  # VJEPA has no CLS token

        if encoder_frozen:
            encoder.requires_grad_(False)
            encoder.eval()
    else:
        encoder = spt.backbone.utils.vit_hf(
            cfg.encoder_scale,
            patch_size=cfg.patch_size,
            image_size=cfg.img_size,
            pretrained=False,
            use_mask_token=False,
        )
        hidden_dim = encoder.config.hidden_size

    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    # Predictor hidden_dim is decoupled from encoder hidden_dim
    predictor_hidden = cfg.predictor.get("hidden_dim_override", hidden_dim)

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=predictor_hidden,
        output_dim=predictor_hidden,
        **{k: v for k, v in cfg.predictor.items() if k != "hidden_dim_override"},
    )

    action_encoder = Embedder(
        input_dim=effective_act_dim,
        smoothed_dim=max(effective_act_dim, 10),
        emb_dim=embed_dim,
    )

    proj_hidden = cfg.get("proj_hidden_dim", 2048)

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=proj_hidden,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=predictor_hidden,
        output_dim=embed_dim,
        hidden_dim=proj_hidden,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        pooling=encoder_pooling,
    )

    # Note: when encoder_frozen=True, requires_grad_(False) ensures the encoder
    # is not updated. Optimizer states are allocated but unused for frozen params.
    # For large encoders, consider passing only trainable params to the optimizer
    # if spt.Module supports per-parameter-group configuration.
    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    loggers = []
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(**cfg.wandb.config)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
        loggers.append(wandb_logger)

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=loggers or None,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()

    # Upload model checkpoint to ClearML if enabled
    clearml_cfg = cfg.get("clearml", {})
    if clearml_cfg.get("upload_model", False):
        upload_clearml_model(clearml_task, run_dir, cfg.output_model_name)

    return


if __name__ == "__main__":
    run()
