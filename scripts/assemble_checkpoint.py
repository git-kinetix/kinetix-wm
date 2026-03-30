#!/usr/bin/env python3
"""Assemble a full JEPA checkpoint from probe components + frozen encoder.

Usage:
    python scripts/assemble_checkpoint.py \
        --probe-components ~/.stable_worldmodel/best-probe_probe_components.pt \
        --encoder-checkpoint ~/.stable_worldmodel/vjepa_checkpoints/vitb16_pretrained.pt \
        --output ~/.stable_worldmodel/lewm_best_assembled_object.ckpt
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from jepa import JEPA
from module import ARPredictor, Embedder, MLP
from train import load_vjepa_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-components", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("Loading probe components ...")
    probe = torch.load(args.probe_components, map_location="cpu", weights_only=False)
    cfg = probe["config"]

    embed_dim = cfg["embed_dim"]
    proj_dim = cfg["proj_dim"]
    action_dim = cfg["action_dim"]
    frameskip = cfg["frameskip"]
    effective_act_dim = frameskip * action_dim

    print(f"  embed_dim={embed_dim}, proj_dim={proj_dim}, action_dim={action_dim}, frameskip={frameskip}")

    # Load encoder
    print("Loading encoder ...")
    encoder = load_vjepa_encoder(args.encoder_checkpoint, patch_size=16, img_size=224)
    encoder.requires_grad_(False)
    encoder.eval()
    hidden_dim = encoder.embed_dim
    print(f"  encoder embed_dim={hidden_dim}")

    # Reconstruct components
    projector = MLP(input_dim=hidden_dim, output_dim=proj_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    predictor = ARPredictor(
        num_frames=cfg["history_size"], input_dim=proj_dim, hidden_dim=proj_dim,
        output_dim=proj_dim, depth=6, heads=16, mlp_dim=2048, dim_head=64,
        dropout=0.1, emb_dropout=0.0,
    )
    pred_proj = MLP(input_dim=proj_dim, output_dim=proj_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    action_encoder = Embedder(input_dim=effective_act_dim, smoothed_dim=max(effective_act_dim, 10), emb_dim=proj_dim)

    # Load trained weights
    projector.load_state_dict(probe["projector"])
    predictor.load_state_dict(probe["predictor"])
    pred_proj.load_state_dict(probe["pred_proj"])
    action_encoder.load_state_dict(probe["action_encoder"])

    # Assemble JEPA
    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
        pooling="mean",
    )
    model.eval()

    # Verify with dummy input
    print("Verifying ...")
    with torch.no_grad():
        info = {"pixels": torch.randn(1, 4, 3, 224, 224), "action": torch.randn(1, 4, effective_act_dim)}
        out = model.encode(info)
        emb = out["emb"]
        act_emb = out["act_emb"]
        pred = model.predict(emb[:, :3], act_emb[:, :3])
        print(f"  emb: {emb.shape}, pred: {pred.shape}")

    # Save
    torch.save(model, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Saved assembled checkpoint: {args.output} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
