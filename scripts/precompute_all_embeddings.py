#!/usr/bin/env python3
"""Precompute ViT-B/16 embeddings for all datasets.

For each dataset, checks if embeddings already exist, skips if so.
Ensures the ViT-B/16 checkpoint is available (creates via timm if not).

Usage:
    export STABLEWM_HOME=$HOME/.stable_worldmodel
    export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGINS_PATH)")
    python scripts/precompute_all_embeddings.py
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CACHE = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))
CKPT = os.path.join(CACHE, "vjepa_checkpoints", "vitb16_pretrained.pt")

# All datasets to precompute embeddings for.
# The key is the dataset name (matches {name}.h5 in STABLEWM_HOME).
# "cube_single_expert" is the canonical name; we also check "cube" as a fallback.
DATASETS = [
    "pusht_expert_train",
    "tworoom",
    "reacher",
    "humanoid",
    "cube_single_expert",  # canonical name from download script
]

# Fallback names for cube dataset (may appear under different names)
CUBE_ALIASES = ["cube_single_expert", "cube"]


def ensure_checkpoint():
    """Create the ViT-B/16 checkpoint from timm pretrained weights if missing."""
    if os.path.exists(CKPT):
        print(f"[checkpoint] Found: {CKPT}")
        return

    print("[checkpoint] ViT-B/16 checkpoint not found, creating from timm...")
    import timm
    import torch

    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    torch.save({"model": model.state_dict()}, CKPT)
    size_mb = os.path.getsize(CKPT) / 1e6
    print(f"[checkpoint] Saved: {CKPT} ({size_mb:.0f} MB)")


def find_dataset_h5(name):
    """Find the .h5 file for a given dataset name, handling aliases for cube."""
    h5_path = os.path.join(CACHE, f"{name}.h5")
    if os.path.exists(h5_path):
        return h5_path, name

    # For cube, try aliases
    if "cube" in name:
        for alias in CUBE_ALIASES:
            alt_path = os.path.join(CACHE, f"{alias}.h5")
            if os.path.exists(alt_path):
                return alt_path, alias

    return None, name


def precompute_one(name):
    """Precompute embeddings for a single dataset."""
    # Check if output already exists
    out_path = os.path.join(CACHE, f"{name}_vitb16emb.h5")
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  [{name}] SKIP -- embeddings already exist ({size_mb:.0f} MB)")
        return True

    # Also check aliases for the output
    if "cube" in name:
        for alias in CUBE_ALIASES:
            alt_out = os.path.join(CACHE, f"{alias}_vitb16emb.h5")
            if os.path.exists(alt_out):
                print(f"  [{name}] SKIP -- embeddings exist as {alias}_vitb16emb.h5")
                return True

    # Find the source dataset
    h5_path, actual_name = find_dataset_h5(name)
    if h5_path is None:
        print(f"  [{name}] SKIP -- dataset .h5 not found")
        return False

    # Use the actual name for the output if different
    out_name = f"{actual_name}_vitb16emb.h5"

    print(f"\n{'=' * 60}")
    print(f"  [{actual_name}] Precomputing embeddings...")
    print(f"  Source: {h5_path}")
    print(f"  Output: {os.path.join(CACHE, out_name)}")
    print(f"{'=' * 60}")

    # Run the existing precompute script
    script = os.path.join(Path(__file__).parent, "precompute_embeddings.py")
    cmd = [
        sys.executable, script,
        "--checkpoint", CKPT,
        "--dataset", actual_name,
        "--output", out_name,
        "--batch-size", "512",
    ]

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    if result.returncode != 0:
        print(f"  [{actual_name}] ERROR: precompute script exited with code {result.returncode}")
        return False

    print(f"  [{actual_name}] DONE")
    return True


def main():
    os.makedirs(CACHE, exist_ok=True)
    print(f"STABLEWM_HOME = {CACHE}")
    print()

    # Step 1: Ensure ViT-B/16 checkpoint exists
    ensure_checkpoint()
    print()

    # Step 2: Precompute embeddings for each dataset
    results = {}
    for name in DATASETS:
        try:
            ok = precompute_one(name)
            results[name] = "OK" if ok else "SKIPPED (no source)"
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
            results[name] = f"ERROR: {e}"

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for name, status in results.items():
        print(f"  {name:30s} {status}")
    print()

    # Verify all embedding files
    print("Embedding files in STABLEWM_HOME:")
    for f in sorted(os.listdir(CACHE)):
        if f.endswith("_vitb16emb.h5"):
            size_mb = os.path.getsize(os.path.join(CACHE, f)) / 1e6
            print(f"  {f:40s} {size_mb:8.0f} MB")


if __name__ == "__main__":
    main()
