#!/usr/bin/env python3
"""Plan a trajectory using CEM and visualize results.

Picks a validation episode, sets a goal frame, uses Cross-Entropy Method
to find the optimal action sequence in embedding space, then outputs:
1. planned_trajectory.json — planned vs actual embedding trajectories
2. plan_frames/ — extracted video frames for the episode
3. plan_viewer.html — interactive synced viewer

Usage:
    python scripts/plan_and_visualize.py \
        --checkpoint ~/.stable_worldmodel/lewm_vitb16_frozen_epoch_30_object.ckpt \
        --dataset prod_beta0_200ep \
        --episode 5 \
        --horizon 20 \
        --output-dir ~/le-wm/plan_output
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_episode(h5_path, episode_idx, frameskip=5):
    """Load a single episode's pixels and actions from HDF5."""
    with h5py.File(h5_path, "r") as f:
        ep_len = f["ep_len"][episode_idx]
        ep_offset = f["ep_offset"][episode_idx]
        pixels = f["pixels"][ep_offset:ep_offset + ep_len]  # (T, H, W, 3)
        actions = f["action"][ep_offset:ep_offset + ep_len]  # (T, action_dim)

    # Subsample at frameskip
    frame_indices = list(range(0, len(pixels), frameskip))
    pixels_sub = pixels[frame_indices]

    # Concatenate frameskip actions between sampled frames
    actions_concat = []
    for i in range(len(frame_indices)):
        base = frame_indices[i]
        chunk = []
        for s in range(frameskip):
            idx = base + s
            if idx < len(actions):
                chunk.append(actions[idx])
            else:
                chunk.append(np.zeros_like(actions[0]))
        actions_concat.append(np.concatenate(chunk))
    actions_sub = np.stack(actions_concat)

    return pixels_sub, actions_sub, frame_indices


def encode_frames(model, pixels, transform, device):
    """Encode pixel frames into embeddings using the model's encoder + projector."""
    from einops import rearrange

    batch = []
    for img in pixels:
        t = transform(img)
        batch.append(t)
    batch = torch.stack(batch).to(device)  # (T, 3, H, W)

    model.eval()
    with torch.no_grad():
        # Add batch dim
        batch = batch.unsqueeze(0)  # (1, T, 3, H, W)
        b = 1
        px = rearrange(batch, "b t ... -> (b t) ...")

        if model.pooling == "cls":
            output = model.encoder(px, interpolate_pos_encoding=True)
            px_emb = output.last_hidden_state[:, 0]
        else:
            output = model.encoder(px)
            tokens = output if torch.is_tensor(output) else output.last_hidden_state
            px_emb = tokens.mean(dim=1) if tokens.ndim > 2 else tokens

        emb = model.projector(px_emb)  # (T, D)
    return emb


def normalize_actions(actions, mean, std):
    return (actions - mean) / (std + 1e-8)


def cem_plan(model, start_embs, goal_emb, action_mean, action_std,
             horizon=20, history_size=3, num_samples=512, num_iters=5,
             elite_frac=0.1, device="cuda"):
    """Cross-Entropy Method planning in embedding space.

    Returns the best action sequence and all candidate costs per iteration.
    """
    action_dim = action_mean.shape[0]
    num_elite = max(int(num_samples * elite_frac), 2)

    # Initialize action distribution
    mu = torch.zeros(horizon, action_dim, device=device)
    sigma = torch.ones(horizon, action_dim, device=device) * 0.5

    best_actions = None
    best_cost = float("inf")
    all_costs = []

    model.eval()
    with torch.no_grad():
        for iteration in range(num_iters):
            # Sample action sequences: (num_samples, horizon, action_dim)
            noise = torch.randn(num_samples, horizon, action_dim, device=device)
            actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise

            # Rollout each sample
            costs = []
            for s in range(num_samples):
                # Start from the context embeddings
                emb = start_embs.clone()  # (history_size, D)

                for t in range(horizon):
                    act = actions[s, t:t+1].unsqueeze(0)  # (1, 1, action_dim)
                    act_emb = model.action_encoder(act)  # (1, 1, D)

                    # Use last history_size embeddings
                    ctx = emb[-history_size:].unsqueeze(0)  # (1, HS, D)
                    ctx_act = act_emb.expand(1, ctx.shape[1], -1)

                    # Pad act_emb to match context length
                    if ctx_act.shape[1] < ctx.shape[1]:
                        pad = torch.zeros(1, ctx.shape[1] - ctx_act.shape[1], ctx_act.shape[-1], device=device)
                        ctx_act = torch.cat([pad, ctx_act], dim=1)

                    pred = model.predict(ctx, ctx_act[:, :ctx.shape[1]])  # (1, HS, D)
                    next_emb = pred[0, -1]  # (D,)
                    emb = torch.cat([emb, next_emb.unsqueeze(0)], dim=0)

                # Cost = distance to goal at final predicted embedding
                final_emb = emb[-1]
                cost = F.mse_loss(final_emb, goal_emb, reduction="sum").item()
                costs.append(cost)

            costs = torch.tensor(costs, device=device)
            all_costs.append(costs.cpu().numpy().tolist())

            # Select elite
            elite_idx = costs.argsort()[:num_elite]
            elite_actions = actions[elite_idx]

            # Update distribution
            mu = elite_actions.mean(dim=0)
            sigma = elite_actions.std(dim=0).clamp(min=0.01)

            best_idx = costs.argmin()
            if costs[best_idx].item() < best_cost:
                best_cost = costs[best_idx].item()
                best_actions = actions[best_idx].clone()

            print(f"  CEM iter {iteration}: best_cost={best_cost:.6f}, "
                  f"mean_cost={costs.mean():.6f}, elite_mean={costs[elite_idx].mean():.6f}")

    return best_actions.cpu().numpy(), best_cost, all_costs


def rollout_with_actions(model, start_embs, actions, history_size=3, device="cuda"):
    """Rollout the model with a given action sequence, return predicted embeddings."""
    model.eval()
    embs = [start_embs.clone()]

    with torch.no_grad():
        emb_seq = start_embs.clone()  # (HS, D)

        for t in range(len(actions)):
            act = torch.from_numpy(actions[t:t+1]).float().to(device).unsqueeze(0)
            act_emb = model.action_encoder(act)

            ctx = emb_seq[-history_size:].unsqueeze(0)
            ctx_act = act_emb.expand(1, ctx.shape[1], -1)
            if ctx_act.shape[1] < ctx.shape[1]:
                pad = torch.zeros(1, ctx.shape[1] - ctx_act.shape[1], ctx_act.shape[-1], device=device)
                ctx_act = torch.cat([pad, ctx_act], dim=1)

            pred = model.predict(ctx, ctx_act[:, :ctx.shape[1]])
            next_emb = pred[0, -1]
            emb_seq = torch.cat([emb_seq, next_emb.unsqueeze(0)], dim=0)
            embs.append(next_emb.cpu().numpy())

    return np.stack(embs[1:])  # (horizon, D)


def build_viewer_html(output_dir, num_frames, gt_traj_2d, plan_traj_2d, goal_idx, start_idx):
    """Build an interactive HTML viewer with synced video + trajectory plot."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LeWM Path Planning Viewer</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Source+Serif+4:wght@400;600&display=swap" rel="stylesheet">
<style>
:root {{ --bg: #0f0f13; --card: #1a1a24; --text: #e2e2e8; --muted: #6b6b80; --accent: #4f8ff7; --green: #22c55e; --red: #ef4444; --orange: #f59e0b; --border: #2a2a3a; }}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'JetBrains Mono', monospace; background: var(--bg); color: var(--text); }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}
h1 {{ font-family: 'Source Serif 4', serif; font-size: 1.8rem; margin-bottom: 0.5rem; }}
.subtitle {{ color: var(--muted); font-size: 0.8rem; margin-bottom: 2rem; }}
.viewer {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
.panel {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
.panel-header {{ padding: 0.8rem 1rem; border-bottom: 1px solid var(--border); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); display: flex; justify-content: space-between; }}
.frame-container {{ position: relative; padding: 1rem; display: flex; justify-content: center; align-items: center; min-height: 300px; background: #000; }}
.frame-container img {{ max-width: 100%; max-height: 400px; border-radius: 4px; image-rendering: auto; }}
.frame-label {{ position: absolute; top: 0.5rem; left: 0.5rem; background: rgba(0,0,0,0.7); padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.7rem; }}
canvas {{ width: 100%; height: 400px; cursor: crosshair; }}
.controls {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; display: flex; align-items: center; gap: 1rem; }}
.controls button {{ background: var(--accent); color: white; border: none; padding: 0.4rem 1rem; border-radius: 4px; cursor: pointer; font-family: inherit; font-size: 0.75rem; }}
.controls button:hover {{ opacity: 0.9; }}
.controls input[type=range] {{ flex: 1; accent-color: var(--accent); }}
.controls .frame-num {{ font-size: 0.8rem; min-width: 80px; text-align: center; }}
.stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }}
.stat {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; text-align: center; }}
.stat-value {{ font-size: 1.4rem; font-weight: 500; }}
.stat-label {{ font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.3rem; }}
.legend {{ display: flex; gap: 1.5rem; padding: 0.5rem 1rem; font-size: 0.7rem; }}
.legend-item {{ display: flex; align-items: center; gap: 0.4rem; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
@media (max-width: 768px) {{ .viewer {{ grid-template-columns: 1fr; }} .stats {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
</head>
<body>
<div class="container">
<h1>Path Planning: LeWM + CEM</h1>
<p class="subtitle">Frozen ViT-B/16 encoder | 200 episodes | frameskip=5 | CEM: 512 samples x 5 iters</p>

<div class="stats">
  <div class="stat"><div class="stat-value" style="color:var(--accent)" id="s-horizon">{num_frames}</div><div class="stat-label">Planning Horizon</div></div>
  <div class="stat"><div class="stat-value" style="color:var(--green)" id="s-cost">—</div><div class="stat-label">Final Cost (MSE)</div></div>
  <div class="stat"><div class="stat-value" style="color:var(--orange)" id="s-dist">—</div><div class="stat-label">Goal Distance</div></div>
  <div class="stat"><div class="stat-value" id="s-frame">0</div><div class="stat-label">Current Frame</div></div>
</div>

<div class="viewer">
  <div class="panel">
    <div class="panel-header"><span>Video Frame</span><span id="frame-idx">0 / {num_frames-1}</span></div>
    <div class="frame-container">
      <img id="frame-img" src="frames/frame_0000.png" />
      <div class="frame-label" id="frame-type">Context</div>
    </div>
  </div>
  <div class="panel">
    <div class="panel-header">
      <span>Embedding Trajectory (PCA 2D)</span>
      <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:var(--accent)"></div>Ground Truth</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--green)"></div>Planned</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--red)"></div>Goal</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--orange)"></div>Current</div>
      </div>
    </div>
    <div style="padding: 1rem;"><canvas id="traj-canvas"></canvas></div>
  </div>
</div>

<div class="controls">
  <button id="btn-play">Play</button>
  <button id="btn-reset">Reset</button>
  <input type="range" id="slider" min="0" max="{num_frames-1}" value="0" />
  <div class="frame-num" id="slider-label">0 / {num_frames-1}</div>
</div>
</div>

<script>
const gtTraj = {json.dumps(gt_traj_2d.tolist())};
const planTraj = {json.dumps(plan_traj_2d.tolist())};
const goalIdx = {goal_idx};
const startIdx = {start_idx};
const numFrames = {num_frames};

let currentFrame = 0;
let playing = false;
let playInterval = null;

const img = document.getElementById('frame-img');
const slider = document.getElementById('slider');
const frameIdx = document.getElementById('frame-idx');
const sliderLabel = document.getElementById('slider-label');
const frameType = document.getElementById('frame-type');
const canvas = document.getElementById('traj-canvas');
const ctx = canvas.getContext('2d');

function resizeCanvas() {{
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = canvas.offsetHeight * 2;
  ctx.scale(2, 2);
  drawTrajectory();
}}
window.addEventListener('resize', resizeCanvas);

function padNum(n) {{ return String(n).padStart(4, '0'); }}

function setFrame(f) {{
  currentFrame = f;
  img.src = `frames/frame_${{padNum(f)}}.png`;
  slider.value = f;
  frameIdx.textContent = `${{f}} / ${{numFrames - 1}}`;
  sliderLabel.textContent = `${{f}} / ${{numFrames - 1}}`;
  document.getElementById('s-frame').textContent = f;

  if (f < startIdx) frameType.textContent = 'Context';
  else if (f === goalIdx) frameType.textContent = 'Goal';
  else if (f >= startIdx) frameType.textContent = 'Planning';

  drawTrajectory();
}}

function drawTrajectory() {{
  const w = canvas.offsetWidth;
  const h = canvas.offsetHeight;
  ctx.clearRect(0, 0, w, h);

  // Compute bounds
  const all = gtTraj.concat(planTraj);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  all.forEach(p => {{ minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]); minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]); }});
  const pad = 30;
  const sx = (w - 2*pad) / (maxX - minX + 1e-8);
  const sy = (h - 2*pad) / (maxY - minY + 1e-8);
  const s = Math.min(sx, sy);
  const cx = w/2 - (minX+maxX)/2*s;
  const cy = h/2 - (minY+maxY)/2*s;
  const tx = p => p[0]*s + cx;
  const ty = p => p[1]*s + cy;

  // Ground truth trajectory
  ctx.strokeStyle = '#4f8ff7';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([]);
  ctx.globalAlpha = 0.4;
  ctx.beginPath();
  gtTraj.forEach((p, i) => i === 0 ? ctx.moveTo(tx(p), ty(p)) : ctx.lineTo(tx(p), ty(p)));
  ctx.stroke();
  ctx.globalAlpha = 1;

  // GT dots
  gtTraj.forEach((p, i) => {{
    ctx.fillStyle = i <= currentFrame ? '#4f8ff7' : 'rgba(79,143,247,0.15)';
    ctx.beginPath();
    ctx.arc(tx(p), ty(p), i === currentFrame ? 5 : 2.5, 0, Math.PI*2);
    ctx.fill();
  }});

  // Planned trajectory (only show up to current frame)
  if (planTraj.length > 0 && currentFrame >= startIdx) {{
    const showUpto = Math.min(currentFrame - startIdx + 1, planTraj.length);
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.globalAlpha = 0.7;
    ctx.beginPath();
    // Start from last context GT point
    const startPt = gtTraj[startIdx];
    ctx.moveTo(tx(startPt), ty(startPt));
    for (let i = 0; i < showUpto; i++) {{
      ctx.lineTo(tx(planTraj[i]), ty(planTraj[i]));
    }}
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    // Plan dots
    for (let i = 0; i < showUpto; i++) {{
      ctx.fillStyle = '#22c55e';
      ctx.beginPath();
      ctx.arc(tx(planTraj[i]), ty(planTraj[i]), 3, 0, Math.PI*2);
      ctx.fill();
    }}
  }}

  // Goal marker
  if (goalIdx < gtTraj.length) {{
    const gp = gtTraj[goalIdx];
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(tx(gp), ty(gp), 8, 0, Math.PI*2);
    ctx.stroke();
    ctx.fillStyle = 'rgba(239,68,68,0.3)';
    ctx.fill();
    ctx.fillStyle = '#ef4444';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText('GOAL', tx(gp)+12, ty(gp)+4);
  }}

  // Current frame marker
  if (currentFrame < gtTraj.length) {{
    const cp = gtTraj[currentFrame];
    ctx.fillStyle = '#f59e0b';
    ctx.beginPath();
    ctx.arc(tx(cp), ty(cp), 6, 0, Math.PI*2);
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.stroke();
  }}

  // Start marker
  ctx.fillStyle = '#fff';
  ctx.font = '9px JetBrains Mono';
  ctx.fillText('START', tx(gtTraj[0])+8, ty(gtTraj[0])-6);
}}

slider.addEventListener('input', () => setFrame(parseInt(slider.value)));

document.getElementById('btn-play').addEventListener('click', () => {{
  if (playing) {{
    clearInterval(playInterval);
    playing = false;
    document.getElementById('btn-play').textContent = 'Play';
  }} else {{
    playing = true;
    document.getElementById('btn-play').textContent = 'Pause';
    playInterval = setInterval(() => {{
      if (currentFrame >= numFrames - 1) {{ setFrame(0); }}
      else {{ setFrame(currentFrame + 1); }}
    }}, 200);
  }}
}});

document.getElementById('btn-reset').addEventListener('click', () => {{
  clearInterval(playInterval);
  playing = false;
  document.getElementById('btn-play').textContent = 'Play';
  setFrame(0);
}});

// Init
setTimeout(resizeCanvas, 100);
setFrame(0);

// Load planning stats
fetch('plan_stats.json').then(r => r.json()).then(d => {{
  document.getElementById('s-cost').textContent = d.best_cost.toFixed(4);
  document.getElementById('s-dist').textContent = d.goal_distance.toFixed(4);
}}).catch(() => {{}});
</script>
</body>
</html>"""
    (output_dir / "plan_viewer.html").write_text(html)
    print(f"Wrote {output_dir / 'plan_viewer.html'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="prod_beta0_200ep")
    parser.add_argument("--episode", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--cem-samples", type=int, default=512)
    parser.add_argument("--cem-iters", type=int, default=5)
    parser.add_argument("--output-dir", default="plan_output")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))
    h5_path = os.path.join(cache_dir, args.dataset + ".h5")
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint} ...")
    model = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    # Load episode
    print(f"Loading episode {args.episode} from {h5_path} ...")
    pixels, actions, frame_indices = load_episode(h5_path, args.episode, args.frameskip)
    print(f"  Frames: {len(pixels)}, Actions: {actions.shape}")

    # Save frames as PNGs
    from PIL import Image
    for i, img_arr in enumerate(pixels):
        Image.fromarray(img_arr).save(frames_dir / f"frame_{i:04d}.png")
    print(f"  Saved {len(pixels)} frames to {frames_dir}")

    # Build transform
    from torchvision import transforms as T
    from stable_pretraining.data import dataset_stats
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(**dataset_stats.ImageNet),
        T.Resize(224),
    ])

    # Encode all frames
    print("Encoding all frames ...")
    all_embs = encode_frames(model, pixels, transform, device)  # (T, D)
    print(f"  Embeddings: {all_embs.shape}")

    # Normalize actions
    actions_tensor = torch.from_numpy(actions).float()
    actions_tensor = torch.nan_to_num(actions_tensor, 0.0)
    act_mean = actions_tensor.mean(0)
    act_std = actions_tensor.std(0)
    actions_norm = normalize_actions(actions_tensor, act_mean, act_std).numpy()

    # Set up planning
    context_end = args.history_size
    goal_frame = min(context_end + args.horizon, len(pixels) - 1)
    start_embs = all_embs[:context_end]  # (HS, D)
    goal_emb = all_embs[goal_frame]  # (D,)

    print(f"\nPlanning from frame {context_end-1} to frame {goal_frame} (horizon={args.horizon})")
    print(f"  Context frames: 0-{context_end-1}")
    print(f"  Goal frame: {goal_frame}")

    # Run CEM
    print("\nRunning CEM planning ...")
    best_actions, best_cost, all_costs = cem_plan(
        model, start_embs, goal_emb,
        action_mean=act_mean, action_std=act_std,
        horizon=args.horizon, history_size=args.history_size,
        num_samples=args.cem_samples, num_iters=args.cem_iters,
        device=device,
    )
    print(f"  Best cost: {best_cost:.6f}")

    # Rollout planned actions
    print("Rolling out planned trajectory ...")
    planned_embs = rollout_with_actions(
        model, start_embs, best_actions, args.history_size, device
    )

    # Ground truth embeddings for the planning horizon
    gt_embs = all_embs[context_end:goal_frame+1].cpu().numpy()  # (horizon+1, D)

    # PCA for 2D visualization
    all_for_pca = np.concatenate([all_embs.cpu().numpy(), planned_embs], axis=0)
    mean = all_for_pca.mean(0)
    centered = all_for_pca - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_2d = centered @ Vt[:2].T  # (N, 2)

    gt_traj_2d = pca_2d[:len(pixels)]
    plan_traj_2d = pca_2d[len(pixels):]

    # Goal distance
    goal_dist = np.linalg.norm(planned_embs[-1] - all_embs[goal_frame].cpu().numpy())

    # Save stats
    stats = {
        "episode": args.episode,
        "horizon": args.horizon,
        "context_frames": context_end,
        "goal_frame": goal_frame,
        "best_cost": float(best_cost),
        "goal_distance": float(goal_dist),
        "num_frames": len(pixels),
    }
    with open(output_dir / "plan_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Build viewer
    print("Building viewer ...")
    build_viewer_html(output_dir, len(pixels), gt_traj_2d, plan_traj_2d, goal_frame, context_end)

    print(f"\nDone! Open {output_dir / 'plan_viewer.html'}")
    print(f"  Best cost: {best_cost:.6f}")
    print(f"  Goal distance: {goal_dist:.6f}")


if __name__ == "__main__":
    main()
