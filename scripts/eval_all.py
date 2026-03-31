#!/usr/bin/env python3
"""Evaluate all trained models across all environments.

For each environment x model configuration:
  1. Loads the checkpoint (ViT-Tiny object or assembled probe)
  2. Evaluates on the validation set: copy ratio, pred loss, sigreg loss, cosine sim
  3. Saves results to a JSON file per environment

Models evaluated per environment:
  - ViT-Tiny:    lewm_{env}_tiny_epoch_*_object.ckpt  (latest epoch)
  - Probe sr=0.5: {env}-probe-sr05_probe_components.pt
  - Probe best:   iterate over sweep results, pick lowest copy ratio

Usage:
    export STABLEWM_HOME=$HOME/.stable_worldmodel
    export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGINS_PATH)")
    python scripts/eval_all.py
"""

import glob
import json
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from module import ARPredictor, Embedder, MLP, SIGReg


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))

# Environment configs: dataset_name, action_dim, frameskip, keys for eval config
ENVIRONMENTS = {
    "pusht_expert_train": {"action_dim": 2, "frameskip": 5, "env_short": "pusht"},
    "tworoom":            {"action_dim": 2, "frameskip": 5, "env_short": "tworoom"},
    "reacher":            {"action_dim": 2, "frameskip": 5, "env_short": "reacher"},
    "humanoid":           {"action_dim": 21, "frameskip": 5, "env_short": "humanoid"},
    "cube_single_expert": {"action_dim": None, "frameskip": 5, "env_short": "cube"},
}

# Sigreg values used in the sweep (scripts/train_all_probes.sh)
SIGREG_SWEEP_VALUES = [0.01, 0.1, 0.5, 1.0, 2.0]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Dataset (reuse the EmbeddingDataset from train_probe.py)
# ---------------------------------------------------------------------------

class EmbeddingDataset(torch.utils.data.Dataset):
    """Dataset that loads precomputed embeddings + actions from HDF5."""

    def __init__(self, h5_path, num_steps=4, frameskip=5, split="val",
                 train_ratio=0.9, seed=42):
        self.h5 = h5py.File(h5_path, "r")
        self.embeddings = self.h5["embedding"]
        self.actions = self.h5["action"]
        self.ep_len = self.h5["ep_len"][:]
        self.ep_offset = self.h5["ep_offset"][:]
        self.num_steps = num_steps
        self.frameskip = frameskip
        self.span = num_steps * frameskip

        # Build valid (offset, start_idx) pairs
        self.indices = []
        for ep_idx in range(len(self.ep_len)):
            length = int(self.ep_len[ep_idx])
            offset = int(self.ep_offset[ep_idx])
            for start in range(length - self.span + 1):
                self.indices.append((offset, start))

        # Train/val split (same logic as train_probe.py)
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
        frame_indices = [offset + start + t * self.frameskip
                         for t in range(self.num_steps)]

        embs = np.stack([self.embeddings[i] for i in frame_indices])  # (T, D)

        acts = []
        for t in range(self.num_steps):
            base = offset + start + t * self.frameskip
            act_chunk = []
            for s in range(self.frameskip):
                act_idx = base + s
                if act_idx < offset + int(self.ep_len[0]):
                    act_chunk.append(self.actions[act_idx])
                else:
                    act_chunk.append(np.zeros(self.actions.shape[1],
                                              dtype=np.float32))
            acts.append(np.concatenate(act_chunk))

        acts = np.stack(acts)  # (T, frameskip * action_dim)

        return {
            "embedding": torch.from_numpy(embs).float(),
            "action": torch.from_numpy(acts).float(),
        }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_probe(probe_path, h5_path, env_cfg, batch_size=256,
                   num_workers=4, history_size=3, num_preds=1):
    """Evaluate a probe checkpoint on the validation set.

    Returns a dict of metrics:
      pred_loss, copy_baseline, copy_ratio, sigreg_loss, cosine_sim
    """
    probe = torch.load(probe_path, map_location="cpu", weights_only=False)
    cfg = probe["config"]

    embed_dim = cfg["embed_dim"]
    proj_dim = cfg["proj_dim"]
    frameskip = cfg["frameskip"]
    action_dim = cfg["action_dim"]
    effective_act_dim = frameskip * action_dim

    # Reconstruct model components
    projector = MLP(input_dim=embed_dim, output_dim=proj_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d).to(DEVICE)
    predictor = ARPredictor(
        num_frames=cfg.get("history_size", history_size),
        input_dim=proj_dim, hidden_dim=proj_dim,
        output_dim=proj_dim, depth=6, heads=16,
        mlp_dim=2048, dim_head=64,
        dropout=0.1, emb_dropout=0.0,
    ).to(DEVICE)
    pred_proj = MLP(input_dim=proj_dim, output_dim=proj_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d).to(DEVICE)
    action_encoder = Embedder(
        input_dim=effective_act_dim,
        smoothed_dim=max(effective_act_dim, 10),
        emb_dim=proj_dim,
    ).to(DEVICE)
    sigreg = SIGReg(knots=17, num_proj=1024).to(DEVICE)

    # Load trained weights
    projector.load_state_dict(probe["projector"])
    predictor.load_state_dict(probe["predictor"])
    pred_proj.load_state_dict(probe["pred_proj"])
    action_encoder.load_state_dict(probe["action_encoder"])

    # Set to eval mode
    projector.eval()
    predictor.eval()
    pred_proj.eval()
    action_encoder.eval()

    hs = cfg.get("history_size", history_size)
    np_ = cfg.get("num_preds", num_preds)
    num_steps = hs + np_

    # Load validation data
    val_ds = EmbeddingDataset(h5_path, num_steps=num_steps,
                              frameskip=frameskip, split="val")
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    metrics = {
        "pred_loss": 0.0,
        "copy_baseline": 0.0,
        "sigreg_loss": 0.0,
        "cosine_sim": 0.0,
        "n_batches": 0,
    }

    with torch.no_grad():
        for batch in val_loader:
            emb_raw = batch["embedding"].to(DEVICE)
            act_raw = torch.nan_to_num(batch["action"].to(DEVICE), 0.0)
            B, T, D = emb_raw.shape

            emb = projector(emb_raw.reshape(B * T, D)).reshape(B, T, -1)
            act_emb = action_encoder(act_raw)
            ctx_emb = emb[:, :hs]
            ctx_act = act_emb[:, :hs]
            tgt_emb = emb[:, np_:]

            pred_raw = predictor(ctx_emb, ctx_act)
            pred_emb = pred_proj(
                pred_raw.reshape(-1, pred_raw.shape[-1])
            ).reshape(B, hs, -1)

            pred_loss = (pred_emb - tgt_emb).pow(2).mean()
            copy_base = (ctx_emb[:, -1:].expand_as(tgt_emb) - tgt_emb).pow(2).mean()
            sr_loss = sigreg(emb.transpose(0, 1))

            cos_sim = F.cosine_similarity(
                pred_emb.reshape(-1, pred_emb.shape[-1]),
                tgt_emb.reshape(-1, tgt_emb.shape[-1]),
                dim=-1,
            ).mean()

            metrics["pred_loss"] += pred_loss.item()
            metrics["copy_baseline"] += copy_base.item()
            metrics["sigreg_loss"] += sr_loss.item()
            metrics["cosine_sim"] += cos_sim.item()
            metrics["n_batches"] += 1

    n = metrics.pop("n_batches")
    if n > 0:
        for k in metrics:
            metrics[k] /= n
        metrics["copy_ratio"] = metrics["pred_loss"] / (metrics["copy_baseline"] + 1e-8)
    else:
        metrics["copy_ratio"] = float("nan")

    metrics["n_val_samples"] = len(val_ds)
    return metrics


def evaluate_vit_tiny(ckpt_path, dataset_name, env_cfg, batch_size=64,
                      num_workers=4, history_size=3, num_preds=1):
    """Evaluate a ViT-Tiny (full JEPA object) checkpoint on the validation set.

    The checkpoint is a pickled JEPA model object saved by ModelObjectCallBack.
    We run it on the raw pixel dataset (not precomputed embeddings).

    Returns a dict of metrics: pred_loss, copy_baseline, copy_ratio,
    sigreg_loss, cosine_sim
    """
    # Import dependencies for full-model evaluation
    import stable_worldmodel as swm
    import stable_pretraining as spt
    from utils import get_img_preprocessor, get_column_normalizer

    model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = model.to(DEVICE)
    model.eval()
    model.requires_grad_(False)

    sigreg = SIGReg(knots=17, num_proj=1024).to(DEVICE)

    # Load dataset
    cache_dir = Path(CACHE)
    frameskip = env_cfg["frameskip"]
    num_steps = history_size + num_preds

    # Determine keys_to_load based on environment
    env_short = env_cfg["env_short"]
    keys_map = {
        "pusht":    ["pixels", "action", "proprio", "state"],
        "tworoom":  ["pixels", "action", "proprio"],
        "reacher":  ["pixels", "action", "observation"],
        "humanoid": ["pixels", "action", "observation"],
        "cube":     ["pixels", "action", "observation"],
    }
    keys_to_load = keys_map.get(env_short, ["pixels", "action", "observation"])

    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_load=keys_to_load,
        num_steps=num_steps,
        frameskip=frameskip,
        cache_dir=cache_dir,
        transform=None,
    )

    # Build transforms
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=224)]
    for col in keys_to_load:
        if col.startswith("pixels"):
            continue
        try:
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
        except Exception:
            pass

    dataset.transform = spt.data.transforms.Compose(*transforms)

    # Split — val only
    rnd_gen = torch.Generator().manual_seed(42)
    _, val_set = spt.data.random_split(
        dataset, lengths=[0.9, 0.1], generator=rnd_gen
    )
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, drop_last=False)

    metrics = {
        "pred_loss": 0.0,
        "copy_baseline": 0.0,
        "sigreg_loss": 0.0,
        "cosine_sim": 0.0,
        "n_batches": 0,
    }

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)

            output = model.encode(batch)
            emb = output["emb"]        # (B, T, D)
            act_emb = output["act_emb"]

            ctx_emb = emb[:, :history_size]
            ctx_act = act_emb[:, :history_size]
            tgt_emb = emb[:, num_preds:]

            pred_emb = model.predict(ctx_emb, ctx_act)

            pred_loss = (pred_emb - tgt_emb).pow(2).mean()
            copy_base = (ctx_emb[:, -1:].expand_as(tgt_emb) - tgt_emb).pow(2).mean()
            sr_loss = sigreg(emb.transpose(0, 1))

            cos_sim = F.cosine_similarity(
                pred_emb.reshape(-1, pred_emb.shape[-1]),
                tgt_emb.reshape(-1, tgt_emb.shape[-1]),
                dim=-1,
            ).mean()

            metrics["pred_loss"] += pred_loss.item()
            metrics["copy_baseline"] += copy_base.item()
            metrics["sigreg_loss"] += sr_loss.item()
            metrics["cosine_sim"] += cos_sim.item()
            metrics["n_batches"] += 1

    n = metrics.pop("n_batches")
    if n > 0:
        for k in metrics:
            metrics[k] /= n
        metrics["copy_ratio"] = metrics["pred_loss"] / (metrics["copy_baseline"] + 1e-8)
    else:
        metrics["copy_ratio"] = float("nan")

    metrics["n_val_samples"] = len(val_set)
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_latest_vit_tiny(env_name):
    """Find the latest ViT-Tiny object checkpoint for an environment.

    Looks for: lewm_{env}_tiny_epoch_*_object.ckpt in CACHE and subdirs.
    """
    # ViT-Tiny checkpoints may be in CACHE root or a subdirectory
    patterns = [
        os.path.join(CACHE, f"lewm_{env_name}_tiny_epoch_*_object.ckpt"),
        os.path.join(CACHE, "**", f"lewm_{env_name}_tiny_epoch_*_object.ckpt"),
    ]

    # Also try the short name
    env_short = ENVIRONMENTS.get(env_name, {}).get("env_short", env_name)
    if env_short != env_name:
        patterns.extend([
            os.path.join(CACHE, f"lewm_{env_short}_tiny_epoch_*_object.ckpt"),
            os.path.join(CACHE, "**", f"lewm_{env_short}_tiny_epoch_*_object.ckpt"),
        ])

    all_ckpts = []
    for pattern in patterns:
        all_ckpts.extend(glob.glob(pattern, recursive=True))

    if not all_ckpts:
        return None

    # Sort by epoch number (extract from filename)
    def epoch_num(path):
        m = re.search(r"epoch_(\d+)_object\.ckpt", path)
        return int(m.group(1)) if m else 0

    all_ckpts.sort(key=epoch_num)
    return all_ckpts[-1]  # latest epoch


def find_probe_components(env_name, sigreg_value=None):
    """Find probe component checkpoints.

    If sigreg_value is None, finds the sr=0.5 probe.
    Otherwise finds the probe for the given sigreg value.
    """
    if sigreg_value is None:
        # Default: sr=0.5
        pattern = f"{env_name}-probe-sr05_probe_components.pt"
    else:
        # Sweep probe: task-name was {env}-probe-sr{value}
        pattern = f"{env_name}-probe-sr{sigreg_value}_probe_components.pt"

    path = os.path.join(CACHE, pattern)
    if os.path.exists(path):
        return path

    # Try glob for fuzzy matching
    fuzzy = os.path.join(CACHE, f"{env_name}*probe*sr*{sigreg_value or '05'}*_probe_components.pt")
    matches = glob.glob(fuzzy)
    return matches[0] if matches else None


def find_best_probe(env_name, h5_path, env_cfg):
    """Find the best probe from the sigreg sweep (lowest copy ratio on val).

    Evaluates each sweep checkpoint and returns the path of the best one.
    """
    best_path = None
    best_ratio = float("inf")
    best_sr = None

    for sr in SIGREG_SWEEP_VALUES:
        probe_path = find_probe_components(env_name, sigreg_value=sr)
        if probe_path is None:
            continue

        try:
            metrics = evaluate_probe(probe_path, h5_path, env_cfg, batch_size=256)
            ratio = metrics["copy_ratio"]
            print(f"    sr={sr}: copy_ratio={ratio:.4f}")
            if ratio < best_ratio:
                best_ratio = ratio
                best_path = probe_path
                best_sr = sr
        except Exception as e:
            print(f"    sr={sr}: ERROR -- {e}")

    if best_path:
        print(f"    Best probe: sr={best_sr} (copy_ratio={best_ratio:.4f})")

    return best_path, best_sr, best_ratio


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(CACHE, exist_ok=True)
    print(f"STABLEWM_HOME = {CACHE}")
    print(f"Device: {DEVICE}")
    print()

    all_results = {}

    for env_name, env_cfg in ENVIRONMENTS.items():
        print(f"\n{'=' * 70}")
        print(f"  Environment: {env_name}")
        print(f"{'=' * 70}")

        # Skip cube if action_dim is unknown
        if env_cfg["action_dim"] is None:
            print("  SKIP -- action_dim not configured (cube needs manual setup)")
            continue

        env_results = {"environment": env_name, "models": {}}

        # Path to precomputed embeddings (for probe evaluation)
        emb_h5 = os.path.join(CACHE, f"{env_name}_vitb16emb.h5")
        has_embeddings = os.path.exists(emb_h5)

        # ---------------------------------------------------------------
        # Model A: ViT-Tiny from scratch
        # ---------------------------------------------------------------
        print("\n  [A] ViT-Tiny from scratch")
        vit_ckpt = find_latest_vit_tiny(env_name)
        if vit_ckpt:
            print(f"      Checkpoint: {vit_ckpt}")
            try:
                metrics = evaluate_vit_tiny(vit_ckpt, env_name, env_cfg)
                env_results["models"]["vit_tiny"] = {
                    "checkpoint": vit_ckpt,
                    "metrics": metrics,
                }
                print(f"      copy_ratio={metrics['copy_ratio']:.4f}  "
                      f"pred_loss={metrics['pred_loss']:.6f}  "
                      f"cosine_sim={metrics['cosine_sim']:.4f}  "
                      f"sigreg={metrics['sigreg_loss']:.4f}")
            except Exception as e:
                print(f"      ERROR: {e}")
                env_results["models"]["vit_tiny"] = {
                    "checkpoint": vit_ckpt,
                    "error": str(e),
                }
        else:
            print("      Not found -- skipping")

        # ---------------------------------------------------------------
        # Model B: Frozen ViT-B/16 Probe (sigreg=0.5)
        # ---------------------------------------------------------------
        print("\n  [B] Probe (sigreg=0.5)")
        if has_embeddings:
            probe_05_path = find_probe_components(env_name)
            if probe_05_path:
                print(f"      Checkpoint: {probe_05_path}")
                try:
                    metrics = evaluate_probe(probe_05_path, emb_h5, env_cfg)
                    env_results["models"]["probe_sr05"] = {
                        "checkpoint": probe_05_path,
                        "sigreg_weight": 0.5,
                        "metrics": metrics,
                    }
                    print(f"      copy_ratio={metrics['copy_ratio']:.4f}  "
                          f"pred_loss={metrics['pred_loss']:.6f}  "
                          f"cosine_sim={metrics['cosine_sim']:.4f}  "
                          f"sigreg={metrics['sigreg_loss']:.4f}")
                except Exception as e:
                    print(f"      ERROR: {e}")
                    env_results["models"]["probe_sr05"] = {
                        "checkpoint": probe_05_path,
                        "error": str(e),
                    }
            else:
                print("      Not found -- skipping")
        else:
            print("      No embeddings -- skipping")

        # ---------------------------------------------------------------
        # Model C: Best probe from sigreg sweep
        # ---------------------------------------------------------------
        print("\n  [C] Probe (best sigreg from sweep)")
        if has_embeddings:
            print("      Evaluating sweep checkpoints...")
            best_path, best_sr, best_ratio = find_best_probe(
                env_name, emb_h5, env_cfg
            )
            if best_path:
                # Re-evaluate the best to get full metrics (already computed
                # during search, but let's be explicit and store them)
                try:
                    metrics = evaluate_probe(best_path, emb_h5, env_cfg)
                    env_results["models"]["probe_best"] = {
                        "checkpoint": best_path,
                        "sigreg_weight": best_sr,
                        "metrics": metrics,
                    }
                    print(f"      copy_ratio={metrics['copy_ratio']:.4f}  "
                          f"pred_loss={metrics['pred_loss']:.6f}  "
                          f"cosine_sim={metrics['cosine_sim']:.4f}  "
                          f"sigreg={metrics['sigreg_loss']:.4f}")
                except Exception as e:
                    print(f"      ERROR: {e}")
                    env_results["models"]["probe_best"] = {
                        "checkpoint": best_path,
                        "error": str(e),
                    }
            else:
                print("      No sweep checkpoints found -- skipping")
        else:
            print("      No embeddings -- skipping")

        all_results[env_name] = env_results

        # Save per-environment JSON
        json_path = os.path.join(CACHE, f"eval_{env_name}.json")
        with open(json_path, "w") as f:
            json.dump(env_results, f, indent=2, default=str)
        print(f"\n  Saved: {json_path}")

    # ---------------------------------------------------------------
    # Save combined results
    # ---------------------------------------------------------------
    combined_path = os.path.join(CACHE, "eval_all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'=' * 70}")
    print(f"Combined results saved to: {combined_path}")

    # Print summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY: Copy Ratios")
    print(f"{'=' * 70}")
    print(f"{'Environment':<25s} {'ViT-Tiny':>10s} {'Probe 0.5':>10s} {'Probe Best':>12s}")
    print("-" * 60)
    for env_name, res in all_results.items():
        models = res.get("models", {})
        vt = models.get("vit_tiny", {}).get("metrics", {}).get("copy_ratio", "N/A")
        p05 = models.get("probe_sr05", {}).get("metrics", {}).get("copy_ratio", "N/A")
        pb = models.get("probe_best", {}).get("metrics", {}).get("copy_ratio", "N/A")

        vt_str = f"{vt:.4f}" if isinstance(vt, float) else str(vt)
        p05_str = f"{p05:.4f}" if isinstance(p05, float) else str(p05)
        pb_str = f"{pb:.4f}" if isinstance(pb, float) else str(pb)

        print(f"{env_name:<25s} {vt_str:>10s} {p05_str:>10s} {pb_str:>12s}")

    print()


if __name__ == "__main__":
    main()
