#!/usr/bin/env python3
"""Batch planning across multiple episodes, horizons, and train/val splits.

Runs CEM planning for each combination and produces a single interactive viewer.

Usage:
    python scripts/plan_batch.py \
        --checkpoint ~/.stable_worldmodel/lewm_vitb16_frozen_epoch_30_object.ckpt \
        --dataset prod_beta0_200ep \
        --num-episodes 10 \
        --horizons 5,10,15,20 \
        --output-dir ~/le-wm/plan_viewer
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
    with h5py.File(h5_path, "r") as f:
        ep_len = int(f["ep_len"][episode_idx])
        ep_offset = int(f["ep_offset"][episode_idx])
        pixels = f["pixels"][ep_offset:ep_offset + ep_len]
        actions = f["action"][ep_offset:ep_offset + ep_len]

    frame_indices = list(range(0, len(pixels), frameskip))
    pixels_sub = pixels[frame_indices]

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

    return pixels_sub, np.stack(actions_concat)


def encode_frames(model, pixels, transform, device):
    from einops import rearrange
    batch = torch.stack([transform(img) for img in pixels]).to(device).unsqueeze(0)
    with torch.no_grad():
        px = rearrange(batch, "b t ... -> (b t) ...")
        if model.pooling == "cls":
            output = model.encoder(px, interpolate_pos_encoding=True)
            px_emb = output.last_hidden_state[:, 0]
        else:
            output = model.encoder(px)
            tokens = output if torch.is_tensor(output) else output.last_hidden_state
            px_emb = tokens.mean(dim=1) if tokens.ndim > 2 else tokens
        emb = model.projector(px_emb)
    return emb


def cem_plan(model, start_embs, goal_emb, horizon, history_size=3,
             num_samples=256, num_iters=5, action_dim=30, device="cuda"):
    num_elite = max(int(num_samples * 0.1), 2)
    mu = torch.zeros(horizon, action_dim, device=device)
    sigma = torch.ones(horizon, action_dim, device=device) * 0.5
    best_actions = None
    best_cost = float("inf")

    with torch.no_grad():
        for it in range(num_iters):
            noise = torch.randn(num_samples, horizon, action_dim, device=device)
            actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise
            costs = []

            for s in range(num_samples):
                emb = start_embs.clone()
                for t in range(horizon):
                    act = actions[s, t:t+1].unsqueeze(0)
                    act_emb = model.action_encoder(act)
                    ctx = emb[-history_size:].unsqueeze(0)
                    ctx_act = act_emb.expand(1, min(ctx.shape[1], act_emb.shape[1]), -1)
                    if ctx_act.shape[1] < ctx.shape[1]:
                        pad = torch.zeros(1, ctx.shape[1] - ctx_act.shape[1], ctx_act.shape[-1], device=device)
                        ctx_act = torch.cat([pad, ctx_act], dim=1)
                    pred = model.predict(ctx, ctx_act[:, :ctx.shape[1]])
                    emb = torch.cat([emb, pred[0, -1:]], dim=0)
                costs.append(F.mse_loss(emb[-1], goal_emb, reduction="sum").item())

            costs_t = torch.tensor(costs, device=device)
            elite_idx = costs_t.argsort()[:num_elite]
            mu = actions[elite_idx].mean(dim=0)
            sigma = actions[elite_idx].std(dim=0).clamp(min=0.01)

            if costs_t[costs_t.argsort()[0]].item() < best_cost:
                best_cost = costs_t[costs_t.argsort()[0]].item()
                best_actions = actions[costs_t.argsort()[0]].clone()

    return best_actions.cpu().numpy(), best_cost


def rollout_actions(model, start_embs, actions, history_size=3, device="cuda"):
    embs = []
    with torch.no_grad():
        emb_seq = start_embs.clone()
        for t in range(len(actions)):
            act = torch.from_numpy(actions[t:t+1]).float().to(device).unsqueeze(0)
            act_emb = model.action_encoder(act)
            ctx = emb_seq[-history_size:].unsqueeze(0)
            ctx_act = act_emb.expand(1, min(ctx.shape[1], act_emb.shape[1]), -1)
            if ctx_act.shape[1] < ctx.shape[1]:
                pad = torch.zeros(1, ctx.shape[1] - ctx_act.shape[1], ctx_act.shape[-1], device=device)
                ctx_act = torch.cat([pad, ctx_act], dim=1)
            pred = model.predict(ctx, ctx_act[:, :ctx.shape[1]])
            next_emb = pred[0, -1]
            emb_seq = torch.cat([emb_seq, next_emb.unsqueeze(0)], dim=0)
            embs.append(next_emb.cpu().numpy())
    return np.stack(embs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="prod_beta0_200ep")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--horizons", default="5,10,15")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--cem-samples", type=int, default=256)
    parser.add_argument("--cem-iters", type=int, default=5)
    parser.add_argument("--output-dir", default="plan_viewer")
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))
    h5_path = os.path.join(cache_dir, args.dataset + ".h5")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model ...")
    model = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.to(device).eval()

    # Get episode count and split
    with h5py.File(h5_path, "r") as f:
        num_total = len(f["ep_len"])
        ep_lens = f["ep_len"][:]

    rng = np.random.RandomState(42)
    all_idx = rng.permutation(num_total)
    split = int(num_total * 0.9)
    train_idx = sorted(all_idx[:split])
    val_idx = sorted(all_idx[split:])

    n_per_split = args.num_episodes // 2
    train_episodes = train_idx[:n_per_split]
    val_episodes = val_idx[:n_per_split]

    from torchvision import transforms as T
    from stable_pretraining.data import dataset_stats
    transform = T.Compose([T.ToTensor(), T.Normalize(**dataset_stats.ImageNet), T.Resize(224)])

    # Run planning for all combos
    results = []
    all_gt_embs = []
    all_plan_embs = []

    for split_name, episodes in [("train", train_episodes), ("val", val_episodes)]:
        for ep_idx in episodes:
            pixels, actions = load_episode(h5_path, ep_idx, args.frameskip)
            if len(pixels) < args.history_size + max(horizons) + 1:
                continue

            ep_dir = output_dir / "frames" / f"{split_name}_ep{ep_idx}"
            ep_dir.mkdir(parents=True, exist_ok=True)

            from PIL import Image
            for i, img in enumerate(pixels):
                Image.fromarray(img).save(ep_dir / f"frame_{i:04d}.png")

            embs = encode_frames(model, pixels, transform, device)

            for horizon in horizons:
                if args.history_size + horizon >= len(pixels):
                    continue

                ctx_end = args.history_size
                goal_frame = ctx_end + horizon
                start_embs = embs[:ctx_end]
                goal_emb = embs[goal_frame]

                print(f"  [{split_name}] ep={ep_idx} h={horizon}: planning ...", end=" ", flush=True)

                best_actions, best_cost = cem_plan(
                    model, start_embs, goal_emb, horizon=horizon,
                    history_size=args.history_size,
                    num_samples=args.cem_samples, num_iters=args.cem_iters,
                    action_dim=actions.shape[1], device=device,
                )

                planned_embs = rollout_actions(model, start_embs, best_actions, args.history_size, device)

                gt_segment = embs[ctx_end:goal_frame+1].cpu().numpy()
                goal_dist = float(np.linalg.norm(planned_embs[-1] - embs[goal_frame].cpu().numpy()))

                # PCA for this episode
                all_np = np.concatenate([embs.cpu().numpy(), planned_embs], axis=0)
                centered = all_np - all_np.mean(0)
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                pca = centered @ Vt[:2].T

                gt_2d = pca[:len(pixels)].tolist()
                plan_2d = pca[len(pixels):].tolist()

                results.append({
                    "split": split_name,
                    "episode": int(ep_idx),
                    "horizon": horizon,
                    "num_frames": len(pixels),
                    "context_end": ctx_end,
                    "goal_frame": goal_frame,
                    "best_cost": float(best_cost),
                    "goal_distance": goal_dist,
                    "gt_traj_2d": gt_2d,
                    "plan_traj_2d": plan_2d,
                    "frames_dir": f"frames/{split_name}_ep{ep_idx}",
                })

                print(f"cost={best_cost:.4f} goal_dist={goal_dist:.4f}")

    # Save results
    with open(output_dir / "plan_results.json", "w") as f:
        json.dump(results, f)

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total plans: {len(results)}")
    for h in horizons:
        for s in ["train", "val"]:
            subset = [r for r in results if r["horizon"] == h and r["split"] == s]
            if subset:
                avg_cost = np.mean([r["best_cost"] for r in subset])
                avg_dist = np.mean([r["goal_distance"] for r in subset])
                print(f"  {s} h={h}: avg_cost={avg_cost:.4f} avg_goal_dist={avg_dist:.4f} (n={len(subset)})")

    # Build viewer
    build_multi_viewer(output_dir, results, horizons)
    print(f"\nDone! Open {output_dir / 'index.html'}")


def build_multi_viewer(output_dir, results, horizons):
    results_json = json.dumps(results)

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LeWM Planning Results</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Source+Serif+4:ital,wght@0,400;0,600;0,700;1,400&display=swap" rel="stylesheet">
<style>
:root { --bg: #0c0c10; --card: #16161e; --card2: #1e1e2a; --text: #e4e4ec; --muted: #6b6b82; --accent: #6366f1; --green: #22c55e; --red: #ef4444; --orange: #f59e0b; --cyan: #06b6d4; --border: #28283a; --hover: #22223a; }
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'JetBrains Mono', monospace; background: var(--bg); color: var(--text); }
.app { max-width: 1400px; margin: 0 auto; padding: 1.5rem; }
header { margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 1rem; }
header h1 { font-family: 'Source Serif 4', serif; font-size: 1.6rem; font-weight: 700; }
header .sub { font-size: 0.72rem; color: var(--muted); }

/* Filters */
.filters { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.filter-group { display: flex; align-items: center; gap: 0.3rem; }
.filter-group label { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
.pill { padding: 0.3rem 0.7rem; border-radius: 4px; border: 1px solid var(--border); font-size: 0.72rem; cursor: pointer; background: var(--card); transition: all 0.15s; font-family: inherit; color: var(--text); }
.pill:hover { border-color: var(--accent); }
.pill.active { background: var(--accent); border-color: var(--accent); color: white; }

/* Stats bar */
.stats-bar { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.8rem; margin-bottom: 1.5rem; }
.stat-card { background: var(--card); border: 1px solid var(--border); border-radius: 6px; padding: 0.8rem; text-align: center; }
.stat-val { font-size: 1.3rem; font-weight: 600; }
.stat-lbl { font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem; }

/* Episode grid */
.ep-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 0.8rem; margin-bottom: 1.5rem; }
.ep-card { background: var(--card); border: 1px solid var(--border); border-radius: 6px; cursor: pointer; overflow: hidden; transition: border-color 0.15s; }
.ep-card:hover { border-color: var(--accent); }
.ep-card.selected { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
.ep-card .thumb { position: relative; background: #000; height: 160px; display: flex; align-items: center; justify-content: center; }
.ep-card .thumb img { max-height: 100%; max-width: 100%; }
.ep-badge { position: absolute; top: 0.4rem; left: 0.4rem; padding: 0.15rem 0.45rem; border-radius: 3px; font-size: 0.6rem; font-weight: 500; }
.ep-badge.train { background: rgba(99,102,241,0.3); color: #a5b4fc; }
.ep-badge.val { background: rgba(6,182,212,0.3); color: #67e8f9; }
.ep-info { padding: 0.6rem 0.8rem; display: flex; justify-content: space-between; font-size: 0.7rem; }
.ep-ratio { font-weight: 600; }
.ep-ratio.good { color: var(--green); }
.ep-ratio.bad { color: var(--red); }

/* Detail viewer */
.detail { background: var(--card); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; display: none; }
.detail.open { display: block; margin-bottom: 1.5rem; }
.detail-header { padding: 0.8rem 1rem; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
.detail-header h3 { font-size: 0.85rem; }
.detail-body { display: grid; grid-template-columns: 1fr 1fr; min-height: 420px; }
.detail-frame { background: #000; display: flex; align-items: center; justify-content: center; position: relative; }
.detail-frame img { max-width: 100%; max-height: 420px; }
.detail-frame .frame-info { position: absolute; bottom: 0.5rem; left: 0.5rem; background: rgba(0,0,0,0.8); padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.65rem; }
.detail-canvas-wrap { padding: 1rem; }
.detail-canvas-wrap canvas { width: 100%; height: 380px; }
.detail-controls { padding: 0.6rem 1rem; border-top: 1px solid var(--border); display: flex; align-items: center; gap: 0.8rem; }
.detail-controls button { background: var(--accent); color: white; border: none; padding: 0.3rem 0.8rem; border-radius: 4px; cursor: pointer; font-family: inherit; font-size: 0.7rem; }
.detail-controls input[type=range] { flex: 1; accent-color: var(--accent); }
.detail-controls .fc { font-size: 0.72rem; min-width: 60px; text-align: center; }

/* Aggregate chart */
.agg-chart { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.2rem; margin-bottom: 1.5rem; }
.agg-chart h3 { font-size: 0.8rem; margin-bottom: 1rem; }
.agg-chart canvas { width: 100%; height: 250px; }

@media (max-width: 900px) { .detail-body { grid-template-columns: 1fr; } .stats-bar { grid-template-columns: repeat(3, 1fr); } }
</style>
</head>
<body>
<div class="app">
<header>
  <div><h1>LeWM Path Planning</h1><div class="sub">Frozen ViT-B/16 | CEM (256 samples x 5 iters) | Precomputed embeddings</div></div>
</header>

<div class="filters" id="filters"></div>
<div class="stats-bar" id="stats-bar"></div>
<div class="agg-chart"><h3>Average Goal Distance by Horizon</h3><canvas id="agg-canvas"></canvas></div>
<div class="detail" id="detail">
  <div class="detail-header"><h3 id="detail-title">—</h3><button onclick="closeDetail()" style="background:none;border:none;color:var(--muted);cursor:pointer;font-size:1.2rem">✕</button></div>
  <div class="detail-body">
    <div class="detail-frame"><img id="d-img" /><div class="frame-info" id="d-frame-info">Frame 0</div></div>
    <div class="detail-canvas-wrap"><canvas id="d-canvas"></canvas></div>
  </div>
  <div class="detail-controls">
    <button id="d-play">Play</button>
    <button onclick="dSetFrame(0)">Reset</button>
    <input type="range" id="d-slider" min="0" max="0" value="0" />
    <div class="fc" id="d-fc">0/0</div>
  </div>
</div>
<div class="ep-grid" id="ep-grid"></div>
</div>

<script>
const DATA = """ + results_json + """;
const HORIZONS = """ + json.dumps(horizons) + """;

let filterSplit = 'all';
let filterHorizon = 'all';
let selectedIdx = null;
let dPlaying = false;
let dInterval = null;
let dFrame = 0;

// Build filters
const filtersEl = document.getElementById('filters');
function addFilterGroup(label, options, setter) {
  const g = document.createElement('div'); g.className = 'filter-group';
  const l = document.createElement('label'); l.textContent = label; g.appendChild(l);
  options.forEach(([val, txt]) => {
    const b = document.createElement('button');
    b.className = 'pill' + (val === 'all' ? ' active' : '');
    b.textContent = txt;
    b.dataset.val = val;
    b.dataset.group = label;
    b.onclick = () => { document.querySelectorAll(`.pill[data-group="${label}"]`).forEach(p => p.classList.remove('active')); b.classList.add('active'); setter(val); render(); };
    g.appendChild(b);
  });
  filtersEl.appendChild(g);
}
addFilterGroup('Split', [['all','All'],['train','Train'],['val','Val']], v => filterSplit = v);
addFilterGroup('Horizon', [['all','All'], ...HORIZONS.map(h => [String(h), 'h='+h])], v => filterHorizon = v);

function filtered() {
  return DATA.filter(r => (filterSplit === 'all' || r.split === filterSplit) && (filterHorizon === 'all' || r.horizon === parseInt(filterHorizon)));
}

function render() {
  const items = filtered();
  // Stats
  const sb = document.getElementById('stats-bar');
  const avgCost = items.length ? (items.reduce((s,r) => s+r.best_cost, 0)/items.length) : 0;
  const avgDist = items.length ? (items.reduce((s,r) => s+r.goal_distance, 0)/items.length) : 0;
  const bestDist = items.length ? Math.min(...items.map(r=>r.goal_distance)) : 0;
  const nTrain = items.filter(r=>r.split==='train').length;
  const nVal = items.filter(r=>r.split==='val').length;
  sb.innerHTML = `
    <div class="stat-card"><div class="stat-val" style="color:var(--accent)">${items.length}</div><div class="stat-lbl">Plans</div></div>
    <div class="stat-card"><div class="stat-val" style="color:var(--cyan)">${nTrain}/${nVal}</div><div class="stat-lbl">Train / Val</div></div>
    <div class="stat-card"><div class="stat-val" style="color:var(--orange)">${avgCost.toFixed(3)}</div><div class="stat-lbl">Avg Cost</div></div>
    <div class="stat-card"><div class="stat-val" style="color:var(--green)">${avgDist.toFixed(3)}</div><div class="stat-lbl">Avg Goal Dist</div></div>
    <div class="stat-card"><div class="stat-val" style="color:var(--green)">${bestDist.toFixed(3)}</div><div class="stat-lbl">Best Goal Dist</div></div>`;

  // Grid
  const grid = document.getElementById('ep-grid');
  grid.innerHTML = '';
  items.forEach((r, i) => {
    const globalIdx = DATA.indexOf(r);
    const card = document.createElement('div');
    card.className = 'ep-card' + (selectedIdx === globalIdx ? ' selected' : '');
    const ratio = r.goal_distance;
    const ratioClass = ratio < 1.5 ? 'good' : 'bad';
    card.innerHTML = `
      <div class="thumb"><img src="${r.frames_dir}/frame_${String(r.goal_frame).padStart(4,'0')}.png" loading="lazy" /><span class="ep-badge ${r.split}">${r.split}</span></div>
      <div class="ep-info"><span>ep${r.episode} h=${r.horizon}</span><span class="ep-ratio ${ratioClass}">d=${ratio.toFixed(2)}</span></div>`;
    card.onclick = () => openDetail(globalIdx);
    grid.appendChild(card);
  });

  drawAggChart();
}

// Aggregate chart
function drawAggChart() {
  const canvas = document.getElementById('agg-canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * 2; canvas.height = canvas.offsetHeight * 2;
  ctx.scale(2, 2);
  const w = canvas.offsetWidth, h = canvas.offsetHeight;
  ctx.clearRect(0,0,w,h);

  const pad = {l:50,r:20,t:20,b:40};
  const pw = w-pad.l-pad.r, ph = h-pad.t-pad.b;

  // Compute avg goal_distance per horizon per split
  const series = {};
  ['train','val'].forEach(s => { series[s] = {}; HORIZONS.forEach(hz => { const sub = DATA.filter(r => r.split===s && r.horizon===hz); series[s][hz] = sub.length ? sub.reduce((a,r)=>a+r.goal_distance,0)/sub.length : null; }); });

  const maxVal = Math.max(...Object.values(series).flatMap(s => Object.values(s).filter(v=>v!==null)), 0.1);

  // Axes
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t+ph); ctx.lineTo(pad.l+pw, pad.t+ph); ctx.stroke();

  // X labels
  ctx.fillStyle = '#666'; ctx.font = '10px JetBrains Mono'; ctx.textAlign = 'center';
  HORIZONS.forEach((hz, i) => {
    const x = pad.l + (i+0.5) * pw / HORIZONS.length;
    ctx.fillText('h='+hz, x, pad.t+ph+20);
  });

  // Y labels
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = maxVal * i / 4;
    const y = pad.t + ph - (i/4) * ph;
    ctx.fillText(v.toFixed(2), pad.l-8, y+3);
    ctx.strokeStyle = '#222'; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l+pw, y); ctx.stroke();
  }

  // Bars
  const barW = pw / HORIZONS.length * 0.35;
  const colors = {train: '#6366f1', val: '#06b6d4'};
  HORIZONS.forEach((hz, i) => {
    const cx = pad.l + (i+0.5) * pw / HORIZONS.length;
    ['train','val'].forEach((s, si) => {
      const v = series[s][hz];
      if (v === null) return;
      const bh = (v / maxVal) * ph;
      const x = cx + (si-1) * barW - 2;
      ctx.fillStyle = colors[s];
      ctx.globalAlpha = 0.8;
      ctx.fillRect(x, pad.t+ph-bh, barW, bh);
      ctx.globalAlpha = 1;
      ctx.fillStyle = '#ccc'; ctx.font = '9px JetBrains Mono'; ctx.textAlign = 'center';
      ctx.fillText(v.toFixed(2), x+barW/2, pad.t+ph-bh-5);
    });
  });

  // Legend
  ctx.font = '10px JetBrains Mono';
  [['train','#6366f1',w-160],['val','#06b6d4',w-80]].forEach(([lbl,col,x]) => {
    ctx.fillStyle = col; ctx.fillRect(x, 8, 10, 10);
    ctx.fillStyle = '#999'; ctx.textAlign = 'left'; ctx.fillText(lbl, x+14, 17);
  });
}

// Detail viewer
function openDetail(idx) {
  selectedIdx = idx;
  const r = DATA[idx];
  document.getElementById('detail').classList.add('open');
  document.getElementById('detail-title').textContent = `${r.split} | Episode ${r.episode} | Horizon ${r.horizon} | Cost ${r.best_cost.toFixed(3)} | Goal Dist ${r.goal_distance.toFixed(3)}`;
  document.getElementById('d-slider').max = r.num_frames - 1;
  dSetFrame(0);
  render();
  document.getElementById('detail').scrollIntoView({behavior:'smooth'});
}

function closeDetail() { selectedIdx = null; document.getElementById('detail').classList.remove('open'); render(); }

function dSetFrame(f) {
  const r = DATA[selectedIdx];
  dFrame = f;
  document.getElementById('d-img').src = `${r.frames_dir}/frame_${String(f).padStart(4,'0')}.png`;
  document.getElementById('d-slider').value = f;
  document.getElementById('d-fc').textContent = `${f}/${r.num_frames-1}`;
  const lbl = f < r.context_end ? 'Context' : f === r.goal_frame ? 'GOAL' : f <= r.goal_frame ? 'Planning' : 'Beyond';
  document.getElementById('d-frame-info').textContent = `Frame ${f} — ${lbl}`;
  drawDetailTraj();
}

function drawDetailTraj() {
  const r = DATA[selectedIdx];
  const canvas = document.getElementById('d-canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth*2; canvas.height = canvas.offsetHeight*2;
  ctx.scale(2,2);
  const w = canvas.offsetWidth, h = canvas.offsetHeight;
  ctx.clearRect(0,0,w,h);

  const gt = r.gt_traj_2d, pl = r.plan_traj_2d;
  const all = gt.concat(pl);
  let mnx=Infinity,mxx=-Infinity,mny=Infinity,mxy=-Infinity;
  all.forEach(p=>{mnx=Math.min(mnx,p[0]);mxx=Math.max(mxx,p[0]);mny=Math.min(mny,p[1]);mxy=Math.max(mxy,p[1]);});
  const pd=25, sx=(w-2*pd)/(mxx-mnx+1e-8), sy=(h-2*pd)/(mxy-mny+1e-8), s=Math.min(sx,sy);
  const cx=w/2-(mnx+mxx)/2*s, cy=h/2-(mny+mxy)/2*s;
  const tx=p=>p[0]*s+cx, ty=p=>p[1]*s+cy;

  // GT line
  ctx.strokeStyle='#6366f1'; ctx.lineWidth=1.5; ctx.globalAlpha=0.3;
  ctx.beginPath(); gt.forEach((p,i)=>i?ctx.lineTo(tx(p),ty(p)):ctx.moveTo(tx(p),ty(p))); ctx.stroke();
  ctx.globalAlpha=1;

  // GT dots
  gt.forEach((p,i)=>{ctx.fillStyle=i<=dFrame?'#6366f1':'rgba(99,102,241,0.12)'; ctx.beginPath(); ctx.arc(tx(p),ty(p),i===dFrame?5:2,0,Math.PI*2); ctx.fill();});

  // Plan line
  if(pl.length && dFrame >= r.context_end) {
    const show = Math.min(dFrame-r.context_end+1, pl.length);
    ctx.strokeStyle='#22c55e'; ctx.lineWidth=2; ctx.setLineDash([4,3]); ctx.globalAlpha=0.7;
    ctx.beginPath(); ctx.moveTo(tx(gt[r.context_end]),ty(gt[r.context_end]));
    for(let i=0;i<show;i++) ctx.lineTo(tx(pl[i]),ty(pl[i]));
    ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha=1;
    for(let i=0;i<show;i++){ctx.fillStyle='#22c55e';ctx.beginPath();ctx.arc(tx(pl[i]),ty(pl[i]),3,0,Math.PI*2);ctx.fill();}
  }

  // Goal
  if(r.goal_frame<gt.length){const g=gt[r.goal_frame];ctx.strokeStyle='#ef4444';ctx.lineWidth=2;ctx.beginPath();ctx.arc(tx(g),ty(g),8,0,Math.PI*2);ctx.stroke();ctx.fillStyle='rgba(239,68,68,0.2)';ctx.fill();ctx.fillStyle='#ef4444';ctx.font='9px JetBrains Mono';ctx.fillText('GOAL',tx(g)+11,ty(g)+3);}

  // Current
  if(dFrame<gt.length){const c=gt[dFrame];ctx.fillStyle='#f59e0b';ctx.beginPath();ctx.arc(tx(c),ty(c),6,0,Math.PI*2);ctx.fill();}

  ctx.fillStyle='#888';ctx.font='9px JetBrains Mono';ctx.fillText('START',tx(gt[0])+8,ty(gt[0])-6);
}

document.getElementById('d-slider').addEventListener('input', e => dSetFrame(parseInt(e.target.value)));
document.getElementById('d-play').addEventListener('click', () => {
  if(dPlaying){clearInterval(dInterval);dPlaying=false;document.getElementById('d-play').textContent='Play';}
  else{dPlaying=true;document.getElementById('d-play').textContent='Pause';dInterval=setInterval(()=>{const r=DATA[selectedIdx];dFrame>=r.num_frames-1?dSetFrame(0):dSetFrame(dFrame+1);},150);}
});

render();
</script>
</body>
</html>"""
    (output_dir / "index.html").write_text(html)
    print(f"Wrote {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
