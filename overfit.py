"""Overfitting test: train on a single batch and print losses every step.

Usage:
    python overfit.py --dataset prod_beta0_debug --steps 200 --batch-size 16 --action-dim 6
"""
import argparse
import os
import torch
import numpy as np
import stable_worldmodel as swm
import stable_pretraining as spt

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_img_preprocessor, get_column_normalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="prod_beta0_debug")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load dataset
    cache_dir = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))
    num_steps = args.history_size + args.num_preds  # 4

    dataset = swm.data.HDF5Dataset(
        name=args.dataset,
        num_steps=num_steps,
        frameskip=args.frameskip,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
        transform=None,
    )

    # Set up transforms
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=224)]
    normalizer = get_column_normalizer(dataset, "action", "action")
    transforms.append(normalizer)
    dataset.transform = spt.data.transforms.Compose(*transforms)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2,
    )

    # Grab a single batch to overfit on
    batch = next(iter(loader))
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    print(f"Batch pixels: {batch['pixels'].shape}")
    print(f"Batch action: {batch['action'].shape}")

    # Build model
    embed_dim = 192
    effective_act_dim = args.frameskip * args.action_dim

    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size

    predictor = ARPredictor(
        num_frames=args.history_size, input_dim=embed_dim, hidden_dim=hidden_dim,
        output_dim=hidden_dim, depth=6, heads=16, mlp_dim=2048, dim_head=64,
        dropout=0.0, emb_dropout=0.0,
    )
    action_encoder = Embedder(
        input_dim=effective_act_dim, smoothed_dim=max(effective_act_dim, 10), emb_dim=embed_dim,
    )
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)

    model = JEPA(
        encoder=encoder, predictor=predictor, action_encoder=action_encoder,
        projector=projector, pred_proj=pred_proj, pooling="cls",
    ).to(device)

    sigreg = SIGReg(knots=17, num_proj=1024).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {train_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    # Overfit loop
    model.train()
    print(f"\n{'Step':>5} | {'Loss':>10} | {'Pred':>10} | {'SIGReg':>10} | {'Ratio':>8}")
    print("-" * 60)

    for step in range(args.steps):
        optimizer.zero_grad()

        b = {k: v.clone() for k, v in batch.items()}
        b["action"] = torch.nan_to_num(b["action"], 0.0)

        output = model.encode(b)
        emb = output["emb"]
        act_emb = output["act_emb"]

        ctx_emb = emb[:, :args.history_size]
        ctx_act = act_emb[:, :args.history_size]
        tgt_emb = emb[:, args.num_preds:]
        pred_emb = model.predict(ctx_emb, ctx_act)

        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        sigreg_loss = sigreg(emb.transpose(0, 1))
        loss = pred_loss + args.sigreg_weight * sigreg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0 or step == args.steps - 1:
            ratio = pred_loss.item() / (args.sigreg_weight * sigreg_loss.item() + 1e-8)
            print(f"{step:5d} | {loss.item():10.6f} | {pred_loss.item():10.6f} | {sigreg_loss.item():10.6f} | {ratio:8.2f}")

    print("\nDone. If pred_loss → 0, the model can memorize the batch (overfitting works).")


if __name__ == "__main__":
    main()
