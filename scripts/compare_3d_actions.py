#!/usr/bin/env python3
"""Compare planned vs GT actions in 3D body pose space.

For each episode, extracts the actual beta-0 root translation + rotation,
compares CEM-planned action deltas against GT deltas, and saves 3D trajectory
data for visualization.

Usage:
    python scripts/compare_3d_actions.py \
        --checkpoint lewm_best_assembled_object.ckpt \
        --num-episodes 8 --horizons 3,5,10
"""

import argparse, json, os, sys, tempfile
from pathlib import Path
import boto3, h5py, numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_episode_rendering_ids(h5_path, num_episodes, seed=42):
    """Get episode indices and their corresponding DB rendering IDs."""
    # We need the rendering_ids to fetch NPZ with 3D poses
    # Since we don't store them in the HDF5, we re-scan the DB
    session = boto3.Session(profile_name='rd-ireland', region_name='eu-west-1')
    dynamodb = session.resource('dynamodb', region_name='eu-west-1')
    s3 = session.client('s3', region_name='eu-west-1')
    table = dynamodb.Table('Prod_Renderings')

    items = []
    scan_kwargs = {}
    while True:
        resp = table.scan(**scan_kwargs)
        for item in resp.get('Items', []):
            if item.get('frames_uri') and item.get('features_uri'):
                items.append(item)
                if len(items) >= 200:
                    break
        if 'LastEvaluatedKey' not in resp or len(items) >= 200:
            break
        scan_kwargs['ExclusiveStartKey'] = resp['LastEvaluatedKey']

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(items))
    split = int(len(items) * 0.9)
    val_idx = sorted(perm[split:])

    return items, val_idx, s3


def download_npz_poses(s3, features_uri):
    """Download NPZ and extract beta-0 root poses."""
    bucket = features_uri.split('/')[2]
    key = '/'.join(features_uri.split('/')[3:])
    with tempfile.NamedTemporaryFile(suffix='.npz') as tmp:
        s3.download_file(bucket, key, tmp.name)
        npz = np.load(tmp.name, allow_pickle=True)
        poses = npz['poses_beta0_world']  # (1, T, J, 3)
        trans = npz['trans_beta0_world']  # (1, T, 3)
        if poses.ndim == 4: poses = poses[0]
        if trans.ndim == 3: trans = trans[0]
        return trans, poses[:, 0, :]  # root translation + root rotation


def rollout_actions_to_3d(start_trans, start_rot, action_deltas, frameskip=5):
    """Integrate action deltas back into 3D trajectories."""
    # action_deltas: (horizon, frameskip*6) — concatenated [dtrans, drot] per substep
    traj_trans = [start_trans.copy()]
    traj_rot = [start_rot.copy()]

    for t in range(len(action_deltas)):
        acts = action_deltas[t].reshape(frameskip, 6)  # (fs, 6)
        cur_t = traj_trans[-1].copy()
        cur_r = traj_rot[-1].copy()
        for s in range(frameskip):
            cur_t = cur_t + acts[s, :3]
            cur_r = cur_r + acts[s, 3:]
        traj_trans.append(cur_t)
        traj_rot.append(cur_r)

    return np.stack(traj_trans), np.stack(traj_rot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="prod_beta0_200ep")
    parser.add_argument("--num-episodes", type=int, default=6)
    parser.add_argument("--horizons", default="3,5,10")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--hs", type=int, default=3)
    parser.add_argument("--cem-samples", type=int, default=256)
    parser.add_argument("--cem-iters", type=int, default=5)
    parser.add_argument("--output-dir", default="action_3d_comparison")
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cd = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))
    h5 = os.path.join(cd, args.dataset + ".h5")
    od = Path(args.output_dir); od.mkdir(parents=True, exist_ok=True)

    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(cd, args.checkpoint)
    model = torch.load(ckpt_path, map_location=device, weights_only=False); model.to(device).eval()

    from torchvision import transforms as T
    from stable_pretraining.data import dataset_stats
    tf = T.Compose([T.ToTensor(), T.Normalize(**dataset_stats.ImageNet), T.Resize(224)])

    # Get rendering items from DB
    print("Fetching rendering IDs from Prod_Renderings ...")
    items, val_indices, s3 = get_episode_rendering_ids(h5, 200)

    # Use val episodes
    episodes_to_eval = val_indices[:args.num_episodes]

    from scripts.plan_batch_v2 import load_episode, encode_frames, cem_plan

    results = []
    for ep_idx in episodes_to_eval:
        if ep_idx >= len(items):
            continue
        item = items[ep_idx]
        rid = item.get('rendering_id', '?')[:40]

        # Load frames + actions from HDF5
        px, ac = load_episode(h5, ep_idx, args.frameskip)
        if len(px) < args.hs + max(horizons) + 1:
            continue

        # Download 3D poses from NPZ
        print(f"\n[ep={ep_idx}] {rid}")
        try:
            gt_trans, gt_rot = download_npz_poses(s3, item['features_uri'])
        except Exception as e:
            print(f"  Skip: {e}")
            continue

        # Subsample 3D at frameskip
        fs_idx = list(range(0, len(gt_trans), args.frameskip))
        gt_trans_sub = gt_trans[fs_idx]
        gt_rot_sub = gt_rot[fs_idx]

        # Encode frames
        embs = encode_frames(model, px, tf, device)
        ac_t = torch.nan_to_num(torch.from_numpy(ac).float(), 0.0)

        for hz in horizons:
            if args.hs + hz >= len(px) or args.hs + hz >= len(gt_trans_sub):
                continue

            ce = args.hs; gf = ce + hz
            se = embs[:ce]; ge = embs[gf]

            # GT actions
            gt_actions = ac_t[ce:gf].numpy()

            # CEM plan
            cem_actions, cem_cost = cem_plan(model, se, ge, hz, args.hs,
                                             args.cem_samples, args.cem_iters, ac.shape[1], device)

            # Integrate actions into 3D trajectories
            start_t = gt_trans_sub[ce - 1]
            start_r = gt_rot_sub[ce - 1]

            gt_traj_t, gt_traj_r = rollout_actions_to_3d(start_t, start_r, gt_actions, args.frameskip)
            cem_traj_t, cem_traj_r = rollout_actions_to_3d(start_t, start_r, cem_actions, args.frameskip)

            # Actual 3D trajectory
            actual_traj_t = gt_trans_sub[ce-1:gf+1]
            actual_traj_r = gt_rot_sub[ce-1:gf+1]

            # Align lengths
            n_pts = min(len(gt_traj_t), len(cem_traj_t), len(actual_traj_t)) - 1
            gt_t = gt_traj_t[1:n_pts+1]; cem_t = cem_traj_t[1:n_pts+1]; act_t = actual_traj_t[1:n_pts+1]
            gt_r = gt_traj_r[1:n_pts+1]; cem_r = cem_traj_r[1:n_pts+1]; act_r = actual_traj_r[1:n_pts+1]

            # 3D metrics
            gt_pos_err = np.mean(np.linalg.norm(gt_t - act_t, axis=1))
            cem_pos_err = np.mean(np.linalg.norm(cem_t - act_t, axis=1))
            gt_final_err = float(np.linalg.norm(gt_t[-1] - act_t[-1]))
            cem_final_err = float(np.linalg.norm(cem_t[-1] - act_t[-1]))
            gt_rot_err = np.mean(np.linalg.norm(gt_r - act_r, axis=1))
            cem_rot_err = np.mean(np.linalg.norm(cem_r - act_r, axis=1))

            results.append({
                "episode": int(ep_idx), "horizon": hz,
                "gt_pos_err_m": float(gt_pos_err), "cem_pos_err_m": float(cem_pos_err),
                "gt_final_pos_err": gt_final_err, "cem_final_pos_err": cem_final_err,
                "gt_rot_err_rad": float(gt_rot_err), "cem_rot_err_rad": float(cem_rot_err),
                # 3D trajectories for viewer
                "actual_3d": actual_traj_t.tolist(),
                "gt_rollout_3d": gt_traj_t.tolist(),
                "cem_rollout_3d": cem_traj_t.tolist(),
                "actual_rot": actual_traj_r.tolist(),
                "gt_rollout_rot": gt_traj_r.tolist(),
                "cem_rollout_rot": cem_traj_r.tolist(),
            })

            print(f"  h={hz}: GT_pos_err={gt_pos_err:.4f}m CEM_pos_err={cem_pos_err:.4f}m "
                  f"GT_rot_err={gt_rot_err:.4f}rad CEM_rot_err={cem_rot_err:.4f}rad")

    # Save
    with open(od / "action_3d_results.json", "w") as f:
        json.dump(results, f)

    # Summary
    print(f"\n=== 3D Action Comparison Summary ===")
    for hz in horizons:
        sub = [r for r in results if r["horizon"] == hz]
        if sub:
            print(f"  h={hz}: GT_pos={np.mean([r['gt_pos_err_m'] for r in sub]):.4f}m "
                  f"CEM_pos={np.mean([r['cem_pos_err_m'] for r in sub]):.4f}m "
                  f"GT_rot={np.mean([r['gt_rot_err_rad'] for r in sub]):.4f}rad "
                  f"CEM_rot={np.mean([r['cem_rot_err_rad'] for r in sub]):.4f}rad")

    # Build 3D viewer
    build_3d_viewer(od, results)
    print(f"\nDone! Open {od / 'viewer_3d.html'}")


def build_3d_viewer(od, results):
    html = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><title>3D Action Comparison</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Source+Serif+4:wght@400;600&display=swap" rel="stylesheet">
<style>
:root { --bg:#0c0c10; --card:#16161e; --text:#e4e4ec; --muted:#6b6b82; --accent:#6366f1; --green:#22c55e; --red:#ef4444; --orange:#f59e0b; --border:#28283a; }
*{margin:0;padding:0;box-sizing:border-box} body{font-family:'JetBrains Mono',monospace;background:var(--bg);color:var(--text);padding:2rem}
h1{font-family:'Source Serif 4',serif;font-size:1.6rem;margin-bottom:.5rem}
.sub{font-size:.72rem;color:var(--muted);margin-bottom:1.5rem}
.controls{display:flex;gap:.6rem;flex-wrap:wrap;margin-bottom:1.5rem}
select,button{background:var(--card);border:1px solid var(--border);color:var(--text);padding:.4rem .8rem;border-radius:4px;font-family:inherit;font-size:.75rem;cursor:pointer}
select:hover,button:hover{border-color:var(--accent)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}
.panel{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:1rem}
.panel h3{font-size:.8rem;margin-bottom:.8rem;color:var(--muted)}
canvas{width:100%;height:400px;border-radius:4px;background:#0a0a0e}
table{width:100%;border-collapse:collapse;font-size:.75rem;margin-top:1rem}
th{text-align:left;color:var(--muted);padding:.4rem;border-bottom:1px solid var(--border);font-size:.65rem;text-transform:uppercase;letter-spacing:.05em}
td{padding:.4rem;border-bottom:1px solid var(--border)}
.good{color:var(--green)} .bad{color:var(--red)}
</style>
</head><body>
<h1>3D Body Trajectory: Planned vs Ground Truth</h1>
<p class="sub">Root body position (meters) and rotation (radians) compared in world coordinates</p>

<div class="controls">
  <select id="sel-ep"></select>
  <select id="sel-hz"></select>
  <button onclick="animateToggle()">Play</button>
</div>

<div class="grid">
  <div class="panel">
    <h3>Top-Down View (XZ plane)</h3>
    <canvas id="c-top"></canvas>
  </div>
  <div class="panel">
    <h3>Side View (XY plane)</h3>
    <canvas id="c-side"></canvas>
  </div>
</div>

<div class="panel" style="margin-top:1.5rem">
<h3>Metrics</h3>
<table>
<thead><tr><th>Metric</th><th>GT Actions</th><th>CEM Planned</th><th>Ratio</th></tr></thead>
<tbody id="metrics"></tbody>
</table>
</div>

<script>
const DATA = """ + json.dumps(results) + """;

const selEp = document.getElementById('sel-ep');
const selHz = document.getElementById('sel-hz');
const eps = [...new Set(DATA.map(r=>r.episode))];
const hzs = [...new Set(DATA.map(r=>r.horizon))].sort((a,b)=>a-b);
eps.forEach(e=>{const o=document.createElement('option');o.value=e;o.textContent='Episode '+e;selEp.appendChild(o)});
hzs.forEach(h=>{const o=document.createElement('option');o.value=h;o.textContent='h='+h;selHz.appendChild(o)});

let animFrame=0, animating=false, animInterval=null;

function getCurrent(){
  return DATA.find(r=>r.episode==parseInt(selEp.value)&&r.horizon==parseInt(selHz.value));
}

function draw(){
  const r=getCurrent(); if(!r) return;
  drawCanvas('c-top',r,0,2,'X','Z');  // XZ
  drawCanvas('c-side',r,0,1,'X','Y'); // XY
  drawMetrics(r);
}

function drawCanvas(id,r,ax1,ax2,lbl1,lbl2){
  const canvas=document.getElementById(id);
  const ctx=canvas.getContext('2d');
  canvas.width=canvas.offsetWidth*2;canvas.height=canvas.offsetHeight*2;ctx.scale(2,2);
  const w=canvas.offsetWidth,h=canvas.offsetHeight;ctx.clearRect(0,0,w,h);

  const actual=r.actual_3d, gt=r.gt_rollout_3d, cem=r.cem_rollout_3d;
  const all=[...actual,...gt,...cem];
  let mn1=Infinity,mx1=-Infinity,mn2=Infinity,mx2=-Infinity;
  all.forEach(p=>{mn1=Math.min(mn1,p[ax1]);mx1=Math.max(mx1,p[ax1]);mn2=Math.min(mn2,p[ax2]);mx2=Math.max(mx2,p[ax2])});
  const pad=40,rng1=mx1-mn1||1,rng2=mx2-mn2||1;
  const s=Math.min((w-2*pad)/rng1,(h-2*pad)/rng2)*0.9;
  const cx=w/2-(mn1+mx1)/2*s,cy=h/2-(mn2+mx2)/2*s;
  const tx=p=>p[ax1]*s+cx,ty=p=>p[ax2]*s+cy;

  // Grid
  ctx.strokeStyle='#1a1a2a';ctx.lineWidth=0.5;
  for(let i=0;i<5;i++){const y=pad+i*(h-2*pad)/4;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(w-pad,y);ctx.stroke()}
  for(let i=0;i<5;i++){const x=pad+i*(w-2*pad)/4;ctx.beginPath();ctx.moveTo(x,pad);ctx.lineTo(x,h-pad);ctx.stroke()}

  // Axis labels
  ctx.fillStyle='#444';ctx.font='10px JetBrains Mono';
  ctx.fillText(lbl1,w-30,h-10);ctx.fillText(lbl2,10,20);

  function drawPath(pts,color,dash){
    ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(dash||[]);
    ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(tx(p),ty(p)):ctx.moveTo(tx(p),ty(p)));ctx.stroke();
    ctx.setLineDash([]);
    const last=Math.min(animFrame+1,pts.length);
    for(let i=0;i<last;i++){
      ctx.fillStyle=i===last-1?color:'rgba(255,255,255,0.1)';
      ctx.beginPath();ctx.arc(tx(pts[i]),ty(pts[i]),i===last-1?5:2,0,Math.PI*2);ctx.fill();
    }
  }

  drawPath(actual,'#6366f1',[]);
  drawPath(gt,'#22c55e',[5,3]);
  drawPath(cem,'#ef4444',[2,2]);

  // Legend
  const ly=15;
  [{c:'#6366f1',l:'Actual'},{c:'#22c55e',l:'GT Actions'},{c:'#ef4444',l:'CEM Planned'}].forEach((lg,i)=>{
    ctx.fillStyle=lg.c;ctx.fillRect(w-150,ly+i*16,10,10);
    ctx.fillStyle='#888';ctx.font='10px JetBrains Mono';ctx.fillText(lg.l,w-135,ly+i*16+9);
  });

  // Start/end markers
  ctx.fillStyle='#fff';ctx.font='9px JetBrains Mono';
  if(actual.length){ctx.fillText('START',tx(actual[0])+8,ty(actual[0])-6);ctx.fillText('GOAL',tx(actual[actual.length-1])+8,ty(actual[actual.length-1])-6)}
}

function drawMetrics(r){
  const tbody=document.getElementById('metrics');
  const rows=[
    ['Avg Position Error (m)',r.gt_pos_err_m.toFixed(4),r.cem_pos_err_m.toFixed(4)],
    ['Final Position Error (m)',r.gt_final_pos_err.toFixed(4),r.cem_final_pos_err.toFixed(4)],
    ['Avg Rotation Error (rad)',r.gt_rot_err_rad.toFixed(4),r.cem_rot_err_rad.toFixed(4)],
  ];
  tbody.innerHTML=rows.map(([m,gt,cem])=>{
    const ratio=(parseFloat(cem)/parseFloat(gt)).toFixed(1);
    const cls=parseFloat(ratio)>2?'bad':'good';
    return `<tr><td>${m}</td><td class="good">${gt}</td><td class="bad">${cem}</td><td class="${cls}">${ratio}x</td></tr>`;
  }).join('');
}

function animateToggle(){
  if(animating){clearInterval(animInterval);animating=false;return}
  animating=true;animFrame=0;
  const r=getCurrent();if(!r)return;
  const maxF=Math.max(r.actual_3d.length,r.gt_rollout_3d.length,r.cem_rollout_3d.length);
  animInterval=setInterval(()=>{animFrame++;if(animFrame>=maxF){animFrame=maxF-1;clearInterval(animInterval);animating=false}draw()},200);
}

selEp.onchange=()=>{animFrame=999;draw()};
selHz.onchange=()=>{animFrame=999;draw()};
animFrame=999;
setTimeout(draw,100);
</script>
</body></html>"""
    (od / "viewer_3d.html").write_text(html)


if __name__ == "__main__":
    main()
