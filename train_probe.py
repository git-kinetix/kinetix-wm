"""Train LeWM predictor on precomputed embeddings (no encoder needed).

Loads precomputed frame embeddings from HDF5 and trains the predictor
+ projector + action encoder with SIGReg loss. Much faster than full
pipeline since no ViT forward pass.

Usage:
    python train_probe.py \
        --dataset prod_beta0_200ep_vitb16emb \
        --embed-dim 768 \
        --epochs 100 \
        --batch-size 256
"""

import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from module import ARPredictor, Embedder, MLP, SIGReg


class EmbeddingDataset(Dataset):
    """Dataset that loads precomputed embeddings + actions from HDF5."""

    def __init__(self, h5_path, num_steps=4, frameskip=5, split="train", train_ratio=0.9, seed=42):
        self.h5 = h5py.File(h5_path, "r")
        self.embeddings = self.h5["embedding"]
        self.actions = self.h5["action"]
        self.ep_len = self.h5["ep_len"][:]
        self.ep_offset = self.h5["ep_offset"][:]
        self.num_steps = num_steps
        self.frameskip = frameskip
        self.span = num_steps * frameskip

        # Build valid (episode, start_idx) pairs
        self.indices = []
        for ep_idx in range(len(self.ep_len)):
            length = int(self.ep_len[ep_idx])
            offset = int(self.ep_offset[ep_idx])
            for start in range(length - self.span + 1):
                self.indices.append((offset, start))

        # Split
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(self.indices))
        split_idx = int(len(self.indices) * train_ratio)
        if split == "train":
            self.indices = [self.indices[i] for i in perm[:split_idx]]
        else:
            self.indices = [self.indices[i] for i in perm[split_idx:]]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        offset, start = self.indices[idx]
        # Sample frames at frameskip intervals
        frame_indices = [offset + start + t * self.frameskip for t in range(self.num_steps)]

        embs = np.stack([self.embeddings[i] for i in frame_indices])  # (T, D)

        # Actions: concatenate frameskip actions between sampled frames
        acts = []
        for t in range(self.num_steps):
            base = offset + start + t * self.frameskip
            act_chunk = []
            for s in range(self.frameskip):
                act_idx = base + s
                if act_idx < offset + int(self.ep_len[0]):  # bounds check
                    act_chunk.append(self.actions[act_idx])
                else:
                    act_chunk.append(np.zeros(self.actions.shape[1], dtype=np.float32))
            acts.append(np.concatenate(act_chunk))  # (frameskip * action_dim,)

        acts = np.stack(acts)  # (T, frameskip * action_dim)

        return {
            "embedding": torch.from_numpy(embs).float(),
            "action": torch.from_numpy(acts).float(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="prod_beta0_200ep_vitb16emb")
    parser.add_argument("--embed-dim", type=int, default=768, help="Encoder embedding dim")
    parser.add_argument("--proj-dim", type=int, default=192, help="Projected embedding dim")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save checkpoint every N epochs (0=only final)")
    parser.add_argument("--clearml", action="store_true")
    parser.add_argument("--project", default="lewm")
    parser.add_argument("--task-name", default="probe-vitb16-frozen")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))
    h5_path = os.path.join(cache_dir, args.dataset + ".h5")

    # ClearML
    cl_logger = None
    if args.clearml:
        try:
            from clearml import Task
            task = Task.init(project_name=args.project, task_name=args.task_name)
            task.connect(vars(args), name="config")
            cl_logger = task.get_logger()
            print(f"ClearML: {task.get_output_log_web_page()}")
        except Exception as e:
            print(f"ClearML init failed: {e}")

    # Data
    num_steps = args.history_size + args.num_preds
    train_ds = EmbeddingDataset(h5_path, num_steps=num_steps, frameskip=args.frameskip, split="train")
    val_ds = EmbeddingDataset(h5_path, num_steps=num_steps, frameskip=args.frameskip, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    print(f"Batches per epoch: {len(train_loader)}")

    # Model components (no encoder!)
    effective_act_dim = args.frameskip * args.action_dim
    proj_dim = args.proj_dim

    projector = MLP(input_dim=args.embed_dim, output_dim=proj_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d).to(device)
    predictor = ARPredictor(
        num_frames=args.history_size, input_dim=proj_dim, hidden_dim=proj_dim,
        output_dim=proj_dim, depth=6, heads=16, mlp_dim=2048, dim_head=64,
        dropout=0.1, emb_dropout=0.0,
    ).to(device)
    pred_proj = MLP(input_dim=proj_dim, output_dim=proj_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d).to(device)
    action_encoder = Embedder(input_dim=effective_act_dim, smoothed_dim=max(effective_act_dim, 10), emb_dim=proj_dim).to(device)
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)

    all_params = list(projector.parameters()) + list(predictor.parameters()) + \
                 list(pred_proj.parameters()) + list(action_encoder.parameters())
    total = sum(p.numel() for p in all_params)
    print(f"Trainable params: {total:,} (no encoder)")

    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_ratio = float("inf")
    main._history = []
    for epoch in range(args.epochs):
        # Train
        projector.train(); predictor.train(); pred_proj.train(); action_encoder.train()
        train_metrics = {"pred_loss": 0, "sigreg_loss": 0, "loss": 0, "copy_baseline": 0, "n": 0}

        for batch in train_loader:
            emb_raw = batch["embedding"].to(device)  # (B, T, encoder_dim)
            act_raw = batch["action"].to(device)      # (B, T, effective_act_dim)
            act_raw = torch.nan_to_num(act_raw, 0.0)

            B, T, D = emb_raw.shape
            # Project embeddings
            emb = projector(emb_raw.reshape(B * T, D)).reshape(B, T, -1)  # (B, T, proj_dim)
            act_emb = action_encoder(act_raw)  # (B, T, proj_dim)

            ctx_emb = emb[:, :args.history_size]
            ctx_act = act_emb[:, :args.history_size]
            tgt_emb = emb[:, args.num_preds:]

            # Predict
            pred_raw = predictor(ctx_emb, ctx_act)
            pred_emb = pred_proj(pred_raw.reshape(B * args.history_size, -1)).reshape(B, args.history_size, -1)

            pred_loss = (pred_emb - tgt_emb).pow(2).mean()
            sr_loss = sigreg(emb.transpose(0, 1))
            loss = pred_loss + args.sigreg_weight * sr_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            with torch.no_grad():
                copy_base = (ctx_emb[:, -1:].expand_as(tgt_emb) - tgt_emb).pow(2).mean()

            train_metrics["pred_loss"] += pred_loss.item()
            train_metrics["sigreg_loss"] += sr_loss.item()
            train_metrics["loss"] += loss.item()
            train_metrics["copy_baseline"] += copy_base.item()
            train_metrics["n"] += 1

        scheduler.step()
        n = train_metrics["n"]

        # Validate
        projector.eval(); predictor.eval(); pred_proj.eval(); action_encoder.eval()
        val_metrics = {"pred_loss": 0, "copy_baseline": 0, "sigreg_loss": 0, "n": 0}

        with torch.no_grad():
            for batch in val_loader:
                emb_raw = batch["embedding"].to(device)
                act_raw = torch.nan_to_num(batch["action"].to(device), 0.0)
                B, T, D = emb_raw.shape

                emb = projector(emb_raw.reshape(B * T, D)).reshape(B, T, -1)
                act_emb = action_encoder(act_raw)
                ctx_emb = emb[:, :args.history_size]
                ctx_act = act_emb[:, :args.history_size]
                tgt_emb = emb[:, args.num_preds:]

                pred_raw = predictor(ctx_emb, ctx_act)
                pred_emb = pred_proj(pred_raw.reshape(-1, pred_raw.shape[-1])).reshape(B, args.history_size, -1)

                val_metrics["pred_loss"] += (pred_emb - tgt_emb).pow(2).mean().item()
                val_metrics["copy_baseline"] += (ctx_emb[:, -1:].expand_as(tgt_emb) - tgt_emb).pow(2).mean().item()
                val_metrics["sigreg_loss"] += sigreg(emb.transpose(0, 1)).item()
                val_metrics["n"] += 1

        vn = val_metrics["n"]
        t_pred = train_metrics["pred_loss"] / n
        t_copy = train_metrics["copy_baseline"] / n
        t_ratio = t_pred / (t_copy + 1e-8)
        v_pred = val_metrics["pred_loss"] / vn
        v_copy = val_metrics["copy_baseline"] / vn
        v_ratio = v_pred / (v_copy + 1e-8)
        v_sreg = val_metrics["sigreg_loss"] / vn

        if v_ratio < best_val_ratio:
            best_val_ratio = v_ratio
            marker = " *best*"
        else:
            marker = ""

        main._history.append({
            "epoch": epoch, "train_pred": t_pred, "train_ratio": t_ratio,
            "train_copy": t_copy, "val_pred": v_pred, "val_ratio": v_ratio,
            "val_copy": v_copy, "val_sigreg": v_sreg,
            "lr": scheduler.get_last_lr()[0],
        })

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train pred={t_pred:.6f} ratio={t_ratio:.2f} | "
              f"val pred={v_pred:.6f} ratio={v_ratio:.2f} sreg={v_sreg:.2f}{marker}")

        # Periodic checkpoint
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            ep_path = os.path.join(cache_dir, f"{args.task_name}_epoch{epoch+1}_probe_components.pt")
            torch.save({
                "projector": projector.state_dict(),
                "predictor": predictor.state_dict(),
                "pred_proj": pred_proj.state_dict(),
                "action_encoder": action_encoder.state_dict(),
                "config": vars(args),
                "epoch": epoch + 1,
                "val_copy_ratio": v_ratio,
            }, ep_path)

        # ClearML logging
        if cl_logger:
            cl_logger.report_scalar("pred_loss", "train", t_pred, epoch)
            cl_logger.report_scalar("pred_loss", "val", v_pred, epoch)
            cl_logger.report_scalar("copy_ratio", "train", t_ratio, epoch)
            cl_logger.report_scalar("copy_ratio", "val", v_ratio, epoch)
            cl_logger.report_scalar("copy_baseline", "train", t_copy, epoch)
            cl_logger.report_scalar("copy_baseline", "val", v_copy, epoch)
            cl_logger.report_scalar("sigreg_loss", "val", v_sreg, epoch)
            cl_logger.report_scalar("lr", "optimizer", scheduler.get_last_lr()[0], epoch)

    print(f"\nBest val copy_ratio: {best_val_ratio:.4f}")

    # Save training metrics JSON (full curve for later analysis)
    import json as _json
    metrics_path = os.path.join(cache_dir, f"{args.task_name}_metrics.json")
    _json.dump({
        "config": vars(args),
        "best_val_copy_ratio": best_val_ratio,
        "epochs": list(range(args.epochs)),
        "history": getattr(main, '_history', []),
    }, open(metrics_path, "w"), indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Save components for assembly into full JEPA checkpoint
    save_path = os.path.join(cache_dir, f"{args.task_name}_probe_components.pt")
    torch.save({
        "projector": projector.state_dict(),
        "predictor": predictor.state_dict(),
        "pred_proj": pred_proj.state_dict(),
        "action_encoder": action_encoder.state_dict(),
        "config": vars(args),
    }, save_path)
    print(f"Saved probe components to {save_path}")


if __name__ == "__main__":
    main()
