#!/usr/bin/env python3
"""Precompute ViT-B/16 embeddings for all frames and save to HDF5.

Runs the frozen encoder once over the entire dataset, saves the embeddings
alongside the original action data. Training then loads embeddings directly
instead of running the encoder.

Usage:
    python scripts/precompute_embeddings.py \
        --checkpoint ~/.stable_worldmodel/vjepa_checkpoints/vitb16_pretrained.pt \
        --dataset prod_beta0_200ep \
        --output prod_beta0_200ep_vitb16emb.h5 \
        --batch-size 256
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="prod_beta0_200ep")
    parser.add_argument("--output", default="prod_beta0_200ep_vitb16emb.h5")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    import stable_worldmodel as swm
    import stable_pretraining as spt
    from utils import get_img_preprocessor
    from train import load_vjepa_encoder

    cache_dir = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load encoder
    print(f"Loading encoder from {args.checkpoint} ...")
    encoder = load_vjepa_encoder(args.checkpoint, patch_size=args.patch_size, img_size=args.img_size)
    encoder = encoder.to(device)
    encoder.eval()
    encoder.requires_grad_(False)

    embed_dim = encoder.embed_dim
    print(f"Encoder embed_dim: {embed_dim}")

    # Load dataset — just pixels, no sequences, raw frames
    # We need to process ALL frames, not sequences, so we load the raw HDF5
    h5_path = os.path.join(cache_dir, args.dataset + ".h5")
    print(f"Loading dataset from {h5_path} ...")

    with h5py.File(h5_path, "r") as f:
        total_steps = f["pixels"].shape[0]
        img_h, img_w = f["pixels"].shape[1], f["pixels"].shape[2]
        action_dim = f["action"].shape[1]
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]
        print(f"Total frames: {total_steps}, image: {img_h}x{img_w}, action_dim: {action_dim}")
        print(f"Episodes: {len(ep_len)}")

    # Image preprocessing
    from torchvision import transforms as T
    from stable_pretraining.data import dataset_stats
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(**dataset_stats.ImageNet),
        T.Resize(args.img_size),
    ])

    # Process in batches directly from HDF5
    output_path = os.path.join(cache_dir, args.output)
    print(f"Computing embeddings → {output_path}")

    with h5py.File(h5_path, "r") as src, h5py.File(output_path, "w") as dst:
        # Copy metadata
        dst.create_dataset("ep_len", data=ep_len)
        dst.create_dataset("ep_offset", data=ep_offset)

        # Copy actions as-is
        dst.create_dataset("action", data=src["action"][:])

        # Create embedding dataset
        emb_ds = dst.create_dataset(
            "embedding", shape=(total_steps, embed_dim), dtype=np.float32,
        )

        # Process frames in batches
        processed = 0
        while processed < total_steps:
            end = min(processed + args.batch_size, total_steps)
            # Load pixel batch from HDF5
            pixels = src["pixels"][processed:end]  # (B, H, W, 3) uint8

            # Transform each image
            batch_tensors = []
            for img in pixels:
                t = transform(img)  # (3, img_size, img_size)
                batch_tensors.append(t)
            batch = torch.stack(batch_tensors).to(device)  # (B, 3, H, W)

            # Encode
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = encoder(batch)
                if torch.is_tensor(output):
                    tokens = output
                elif hasattr(output, "last_hidden_state"):
                    tokens = output.last_hidden_state
                else:
                    tokens = output
                if tokens.ndim == 2:
                    embs = tokens
                else:
                    embs = tokens.mean(dim=1)

            emb_ds[processed:end] = embs.float().cpu().numpy()
            processed = end

            if processed % (args.batch_size * 10) == 0 or processed == total_steps:
                print(f"  {processed}/{total_steps} frames ({100*processed/total_steps:.0f}%)")

    print(f"\nDone. Saved to {output_path}")
    print(f"  embedding: ({total_steps}, {embed_dim})")
    print(f"  action: ({total_steps}, {action_dim})")
    print(f"  ep_len: ({len(ep_len)},)")

    # Verify
    with h5py.File(output_path, "r") as f:
        print(f"\nVerification:")
        for k in f.keys():
            print(f"  {k}: {f[k].shape}")


if __name__ == "__main__":
    main()
