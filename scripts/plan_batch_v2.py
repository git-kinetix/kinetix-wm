#!/usr/bin/env python3
"""Batch planning with ground truth comparison.

For each episode:
  - Rollout with GT actions → GT predicted trajectory
  - Rollout with CEM-planned actions → planned trajectory
  - Compare both against actual future embeddings

This normalizes the comparison: both use the same model, same scale.

Usage:
    python scripts/plan_batch_v2.py \
        --checkpoint lewm_best_assembled_object.ckpt \
        --name best-probe \
        --num-episodes 16 --horizons 3,5,10,15
"""

import argparse, json, os, sys
from pathlib import Path
import h5py, numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_episode(h5_path, ep_idx, frameskip=5):
    with h5py.File(h5_path, "r") as f:
        L = int(f["ep_len"][ep_idx]); O = int(f["ep_offset"][ep_idx])
        px = f["pixels"][O:O+L]; ac = f["action"][O:O+L]
    idx = list(range(0, L, frameskip)); px = px[idx]
    acts = []
    for i in range(len(idx)):
        b = idx[i]; ch = [ac[b+s] if b+s < L else np.zeros_like(ac[0]) for s in range(frameskip)]
        acts.append(np.concatenate(ch))
    return px, np.stack(acts)


def encode_frames(model, pixels, transform, device):
    from einops import rearrange
    batch = torch.stack([transform(img) for img in pixels]).to(device).unsqueeze(0)
    with torch.no_grad():
        px = rearrange(batch, "b t ... -> (b t) ...")
        if model.pooling == "cls":
            out = model.encoder(px, interpolate_pos_encoding=True); emb = out.last_hidden_state[:, 0]
        else:
            out = model.encoder(px)
            tok = out if torch.is_tensor(out) else out.last_hidden_state
            emb = tok.mean(dim=1) if tok.ndim > 2 else tok
        return model.projector(emb)


def rollout(model, start_embs, actions_np, hs=3, device="cuda"):
    """Rollout model with given actions, return predicted embeddings."""
    preds = []
    with torch.no_grad():
        seq = start_embs.clone()
        for t in range(len(actions_np)):
            act = torch.from_numpy(actions_np[t:t+1]).float().to(device).unsqueeze(0)
            ae = model.action_encoder(act)
            ctx = seq[-hs:].unsqueeze(0)
            ca = ae.expand(1, min(ctx.shape[1], ae.shape[1]), -1)
            if ca.shape[1] < ctx.shape[1]:
                ca = torch.cat([torch.zeros(1, ctx.shape[1]-ca.shape[1], ca.shape[-1], device=device), ca], 1)
            p = model.predict(ctx, ca[:, :ctx.shape[1]])
            seq = torch.cat([seq, p[0, -1:]], 0)
            preds.append(p[0, -1].cpu().numpy())
    return np.stack(preds)


def cem_plan(model, start_embs, goal_emb, horizon, hs=3, ns=256, ni=5, adim=30, device="cuda"):
    ne = max(int(ns * 0.1), 2)
    mu = torch.zeros(horizon, adim, device=device)
    sig = torch.ones(horizon, adim, device=device) * 0.5
    best_a, best_c = None, float("inf")
    with torch.no_grad():
        for _ in range(ni):
            acts = mu + sig * torch.randn(ns, horizon, adim, device=device)
            costs = []
            for s in range(ns):
                seq = start_embs.clone()
                for t in range(horizon):
                    a = acts[s, t:t+1].unsqueeze(0); ae = model.action_encoder(a)
                    ctx = seq[-hs:].unsqueeze(0)
                    ca = ae.expand(1, min(ctx.shape[1], ae.shape[1]), -1)
                    if ca.shape[1] < ctx.shape[1]:
                        ca = torch.cat([torch.zeros(1, ctx.shape[1]-ca.shape[1], ca.shape[-1], device=device), ca], 1)
                    p = model.predict(ctx, ca[:, :ctx.shape[1]])
                    seq = torch.cat([seq, p[0, -1:]], 0)
                costs.append(F.mse_loss(seq[-1], goal_emb, reduction="sum").item())
            ct = torch.tensor(costs, device=device)
            ei = ct.argsort()[:ne]; mu = acts[ei].mean(0); sig = acts[ei].std(0).clamp(min=0.01)
            if ct.min().item() < best_c: best_c = ct.min().item(); best_a = acts[ct.argmin()].clone()
    return best_a.cpu().numpy(), best_c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", default="model")
    parser.add_argument("--dataset", default="prod_beta0_200ep")
    parser.add_argument("--num-episodes", type=int, default=16)
    parser.add_argument("--horizons", default="3,5,10,15")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--hs", type=int, default=3)
    parser.add_argument("--cem-samples", type=int, default=256)
    parser.add_argument("--cem-iters", type=int, default=5)
    parser.add_argument("--output-dir", default="plan_results")
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cd = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))
    h5 = os.path.join(cd, args.dataset + ".h5")
    od = Path(args.output_dir); od.mkdir(parents=True, exist_ok=True)

    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(cd, args.checkpoint)
    model = torch.load(ckpt_path, map_location=device, weights_only=False); model.to(device).eval()

    with h5py.File(h5, "r") as f: ntot = len(f["ep_len"])
    rng = np.random.RandomState(42)
    perm = rng.permutation(ntot); sp = int(ntot * 0.9)
    train_ep = sorted(perm[:sp]); val_ep = sorted(perm[sp:])
    n = args.num_episodes // 2
    train_sel = train_ep[:n]; val_sel = val_ep[:n]

    from torchvision import transforms as T
    from stable_pretraining.data import dataset_stats
    tf = T.Compose([T.ToTensor(), T.Normalize(**dataset_stats.ImageNet), T.Resize(224)])

    results = []
    for split, episodes in [("train", train_sel), ("val", val_sel)]:
        for ep in episodes:
            px, ac = load_episode(h5, ep, args.frameskip)
            if len(px) < args.hs + max(horizons) + 1: continue

            # Save frames
            fd = od / "frames" / f"{split}_ep{ep}"; fd.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            for i, img in enumerate(px): Image.fromarray(img).save(fd / f"frame_{i:04d}.png")

            embs = encode_frames(model, px, tf, device)

            # Normalize actions for this episode
            ac_t = torch.from_numpy(ac).float()
            ac_t = torch.nan_to_num(ac_t, 0.0)

            for hz in horizons:
                if args.hs + hz >= len(px): continue
                ce = args.hs; gf = ce + hz
                se = embs[:ce]; ge = embs[gf]

                # GT rollout: use actual actions from the episode
                gt_actions = ac_t[ce:gf].numpy()
                gt_pred = rollout(model, se, gt_actions, args.hs, device)

                # CEM rollout
                cem_actions, cem_cost = cem_plan(model, se, ge, hz, args.hs, args.cem_samples, args.cem_iters, ac.shape[1], device)
                cem_pred = rollout(model, se, cem_actions, args.hs, device)

                # Actual future embeddings
                actual = embs[ce:gf+1].cpu().numpy()

                # Metrics (all in the same embedding scale)
                gt_goal_dist = float(np.linalg.norm(gt_pred[-1] - embs[gf].cpu().numpy()))
                cem_goal_dist = float(np.linalg.norm(cem_pred[-1] - embs[gf].cpu().numpy()))
                gt_traj_err = float(np.mean([np.linalg.norm(gt_pred[i] - actual[i+1]) for i in range(len(gt_pred))]))
                cem_traj_err = float(np.mean([np.linalg.norm(cem_pred[i] - actual[i+1]) for i in range(len(cem_pred))]))

                # Cosine sim at goal
                gt_cos = float(F.cosine_similarity(torch.tensor(gt_pred[-1:]), embs[gf:gf+1].cpu(), dim=-1).item())
                cem_cos = float(F.cosine_similarity(torch.tensor(cem_pred[-1:]), embs[gf:gf+1].cpu(), dim=-1).item())

                # PCA
                all_np = np.concatenate([embs.cpu().numpy(), gt_pred, cem_pred])
                c = all_np - all_np.mean(0); _, _, Vt = np.linalg.svd(c, full_matrices=False)
                pca = c @ Vt[:2].T
                N = len(px)

                results.append({
                    "model": args.name, "split": split, "episode": int(ep),
                    "horizon": hz, "num_frames": len(px), "context_end": ce, "goal_frame": gf,
                    "gt_goal_dist": gt_goal_dist, "cem_goal_dist": cem_goal_dist,
                    "gt_traj_err": gt_traj_err, "cem_traj_err": cem_traj_err,
                    "gt_cos_sim": gt_cos, "cem_cos_sim": cem_cos,
                    "cem_cost": float(cem_cost),
                    "gt_traj_2d": pca[:N].tolist(),
                    "gt_pred_2d": pca[N:N+len(gt_pred)].tolist(),
                    "cem_pred_2d": pca[N+len(gt_pred):].tolist(),
                    "frames_dir": f"frames/{split}_ep{ep}",
                })

                print(f"  [{split}] ep={ep} h={hz}: GT_dist={gt_goal_dist:.3f} CEM_dist={cem_goal_dist:.3f} GT_cos={gt_cos:.4f} CEM_cos={cem_cos:.4f}")

    # Save
    with open(od / f"results_{args.name}.json", "w") as f: json.dump(results, f)

    # Summary
    print(f"\n=== {args.name} Summary ===")
    for hz in horizons:
        for s in ["train", "val"]:
            sub = [r for r in results if r["horizon"] == hz and r["split"] == s]
            if sub:
                print(f"  {s} h={hz}: GT_dist={np.mean([r['gt_goal_dist'] for r in sub]):.3f} "
                      f"CEM_dist={np.mean([r['cem_goal_dist'] for r in sub]):.3f} "
                      f"GT_cos={np.mean([r['gt_cos_sim'] for r in sub]):.4f} "
                      f"CEM_cos={np.mean([r['cem_cos_sim'] for r in sub]):.4f}")

    print(f"\nSaved to {od / f'results_{args.name}.json'}")


if __name__ == "__main__":
    main()
