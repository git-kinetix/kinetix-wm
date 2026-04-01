# True Motion Training Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fetch all 2,722 SDP10 renderings with animation_radius > 200cm, precompute VJEPA embeddings in a streaming pipeline (no raw pixel storage), train probes with sigreg sweep, generate CEM planning rollouts at multiple horizons, and build an interactive visualization.

**Architecture:** A streaming fetch-encode pipeline downloads each video, encodes frames through frozen ViT-B/16 on-the-fly, extracts beta-0 body pose deltas, and writes only the 768d embeddings + 6d actions to HDF5. This avoids the 185GB pixel storage problem (only ~5GB for embeddings). The probe training and evaluation reuse existing `train_probe.py` and `eval` infrastructure.

**Tech Stack:** boto3 (DynamoDB + S3), PyAV (video decode), timm (ViT-B/16), torch, h5py, stable_worldmodel

**Compute:** H200 GPU, ~3h total (1.5h fetch+encode, 30min probe training, 30min eval, 30min viz)

**Disk:** ~5GB for embeddings + actions (fits in 62GB available)

---

## File Structure

### New files:
```
scripts/fetch_and_encode_true_motion.py   # Streaming fetch → encode → HDF5 (Task 1)
config/train/data/true_motion.yaml        # Hydra data config for probe training (Task 3)
scripts/run_true_motion_experiment.sh     # Master orchestrator (Task 6)
```

### Existing files used (no modification):
```
train_probe.py                            # Probe training with --save-every
scripts/assemble_checkpoint.py            # Build full JEPA from probe + encoder
scripts/precompute_embeddings.py          # Reference for encoding pattern
scripts/build_explorers.py                # HTML explorer generator
scripts/humanoid_plan_rollout.py          # Reference for CEM rollout capture
```

### Files to modify:
```
scripts/fetch_filtered_renderings.py      # Add --stream-encode flag (Task 1)
```

---

### Task 1: Streaming Fetch + Encode Pipeline

**Files:**
- Create: `scripts/fetch_and_encode_true_motion.py`

- [ ] **Step 1: Write the streaming pipeline**

This script does everything in one pass per rendering: download video → decode frames → resize to 224 → encode through ViT-B/16 → extract body pose deltas from NPZ → write embeddings + actions to HDF5. Raw pixels are never stored on disk.

```python
#!/usr/bin/env python3
"""Streaming fetch + encode: download SDP10 videos, encode with ViT-B/16, save embeddings.

Filters Prod_Renderings by animation_radius > threshold via Dev_Animations.
Processes each rendering as: video → frames → ViT-B/16 → 768d embeddings.
Stores only embeddings + actions (no raw pixels) → ~5GB for 1.2M frames.

Usage:
    python scripts/fetch_and_encode_true_motion.py \
        --min-radius 200 \
        --max-items 2722 \
        --output true_motion_vitb16emb.h5 \
        --checkpoint $STABLEWM_HOME/vjepa_checkpoints/vitb16_pretrained.pt
"""

import sys, os, argparse, tempfile
from pathlib import Path
from io import BytesIO

import boto3
import h5py
import numpy as np
import torch
import av
from PIL import Image
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import load_vjepa_encoder

# ImageNet normalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_s3_uri(uri):
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


def get_filtered_renderings(dynamodb, min_radius, max_items):
    """Get Prod_Renderings items whose animation has radius > min_radius."""
    # Step 1: Get qualifying animation_ids
    anim_table = dynamodb.Table("Dev_Animations")
    qualifying_aids = set()
    scan_kw = {}
    print(f"Scanning Dev_Animations for radius > {min_radius}...")
    while True:
        resp = anim_table.scan(**scan_kw)
        for item in resp.get("Items", []):
            r = item.get("animation_radius")
            if r is not None:
                try:
                    if float(r) > min_radius:
                        qualifying_aids.add(item["animation_id"])
                except (ValueError, TypeError):
                    pass
        if "LastEvaluatedKey" not in resp:
            break
        scan_kw["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    print(f"  {len(qualifying_aids)} qualifying animations")

    # Step 2: Get renderings for those animations
    prod_table = dynamodb.Table("Prod_Renderings")
    items = []
    scan_kw = {}
    print(f"Scanning Prod_Renderings (max {max_items})...")
    while True:
        resp = prod_table.scan(**scan_kw)
        for item in resp.get("Items", []):
            if (item.get("animation_id") in qualifying_aids
                and item.get("features_uri") and item.get("frames_uri")):
                items.append(item)
                if len(items) >= max_items:
                    print(f"  {len(items)} renderings collected")
                    return items
        if "LastEvaluatedKey" not in resp:
            break
        scan_kw["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    print(f"  {len(items)} renderings collected")
    return items


def decode_video_frames(s3, rendering_id, img_size=224, bucket="kinetix-rd-storage"):
    """Download video from S3 and decode to numpy array."""
    prefix = f"preprocessed_datasets/SDP_10_static_camera/videos/{rendering_id}/"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=5)
    mp4_keys = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".mp4")]
    if not mp4_keys:
        return None
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        s3.download_file(bucket, mp4_keys[0], tmp.name)
        container = av.open(tmp.name)
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_image().convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            frames.append(np.asarray(img))
        container.close()
    return np.stack(frames) if frames else None


def encode_frames(encoder, frames_np, device, batch_size=256):
    """Encode numpy frames (N, H, W, 3) uint8 → (N, 768) float32 embeddings."""
    all_embs = []
    for i in range(0, len(frames_np), batch_size):
        batch = frames_np[i:i+batch_size]
        # Convert to tensor: (B, 3, H, W) float, ImageNet normalized
        t = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
        t = (t - MEAN) / STD
        t = t.to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = encoder(t)
            if torch.is_tensor(out):
                emb = out if out.ndim == 2 else out.mean(dim=1)
            elif hasattr(out, "last_hidden_state"):
                emb = out.last_hidden_state.mean(dim=1)
            else:
                emb = out
        all_embs.append(emb.float().cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def extract_actions(s3, features_uri):
    """Download NPZ, extract root body delta poses (6D)."""
    bucket, key = parse_s3_uri(features_uri)
    try:
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            s3.download_file(bucket, key, tmp.name)
            npz = np.load(tmp.name, allow_pickle=True)
    except Exception:
        return None

    if "poses_beta0_world" not in npz or "trans_beta0_world" not in npz:
        return None

    poses = npz["poses_beta0_world"]
    trans = npz["trans_beta0_world"]
    if poses.ndim == 4:
        poses = poses[0]
    if trans.ndim == 3:
        trans = trans[0]

    root_rot = poses[:, 0, :]
    d_trans = np.diff(trans, axis=0)
    d_rot = np.diff(root_rot, axis=0)
    deltas = np.concatenate([d_trans, d_rot], axis=1).astype(np.float32)
    last = np.full((1, 6), np.nan, dtype=np.float32)
    return np.concatenate([deltas, last], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-radius", type=float, default=200)
    parser.add_argument("--max-items", type=int, default=2722)
    parser.add_argument("--output", default="true_motion_vitb16emb.h5")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--profile", default="rd-ireland")
    args = parser.parse_args()

    cache = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load encoder
    print(f"Loading ViT-B/16 from {args.checkpoint}")
    encoder = load_vjepa_encoder(args.checkpoint, patch_size=16, img_size=224)
    encoder = encoder.to(device).eval()
    encoder.requires_grad_(False)
    embed_dim = encoder.embed_dim
    print(f"  embed_dim={embed_dim}")

    # AWS clients
    session = boto3.Session(profile_name=args.profile, region_name="eu-west-1")
    dynamodb = session.resource("dynamodb", region_name="eu-west-1")
    s3 = session.client("s3", region_name="eu-west-1")

    # Get filtered renderings
    items = get_filtered_renderings(dynamodb, args.min_radius, args.max_items)
    if not items:
        print("No renderings found"); return

    # Process streaming: video → encode → write
    output_path = os.path.join(cache, args.output)
    ep_embeddings = []
    ep_actions = []
    ep_lengths = []

    for i, item in enumerate(items):
        rid = item.get("rendering_id", "?")
        print(f"[{i+1}/{len(items)}] {rid[:50]}...", end=" ")

        # Decode video
        frames = decode_video_frames(s3, rid)
        if frames is None or len(frames) < 4:
            print("SKIP (no video)")
            continue

        # Extract actions from NPZ
        actions = extract_actions(s3, item["features_uri"])
        if actions is None:
            print("SKIP (no poses)")
            continue

        # Align lengths
        n = min(len(frames), len(actions))
        if n < 4:
            print("SKIP (too short)")
            continue
        frames = frames[:n]
        actions = actions[:n]

        # Encode frames → embeddings
        embs = encode_frames(encoder, frames, device, args.batch_size)

        ep_embeddings.append(embs)
        ep_actions.append(actions)
        ep_lengths.append(n)
        print(f"OK ({n} frames)")

    if not ep_lengths:
        print("No episodes processed"); return

    # Write HDF5
    total = sum(ep_lengths)
    offsets = np.concatenate([[0], np.cumsum(ep_lengths[:-1])]).astype(np.int32)

    print(f"\nWriting {len(ep_lengths)} episodes, {total} frames → {output_path}")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("ep_len", data=np.array(ep_lengths, dtype=np.int32))
        f.create_dataset("ep_offset", data=offsets)
        emb_ds = f.create_dataset("embedding", shape=(total, embed_dim), dtype=np.float32)
        act_ds = f.create_dataset("action", shape=(total, 6), dtype=np.float32)
        idx = 0
        for embs, acts in zip(ep_embeddings, ep_actions):
            n = len(embs)
            emb_ds[idx:idx+n] = embs
            act_ds[idx:idx+n] = acts
            idx += n

    print(f"Done. {total:,} frames, {os.path.getsize(output_path)/1e9:.1f} GB")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sync to H200 and run in tmux**

```bash
rsync -avz scripts/fetch_and_encode_true_motion.py RD-H100-2:~/le-wm/scripts/
ssh -x RD-H100-2 'tmux new-session -d -s truemotion "cd ~/le-wm && source .venv/bin/activate && export STABLEWM_HOME=\$HOME/.stable_worldmodel && export HDF5_PLUGIN_PATH=\$(python3 -c \"import hdf5plugin; print(hdf5plugin.PLUGINS_PATH)\") && python scripts/fetch_and_encode_true_motion.py --min-radius 200 --max-items 2722 --output true_motion_vitb16emb.h5 --checkpoint \$STABLEWM_HOME/vjepa_checkpoints/vitb16_pretrained.pt 2>&1 | tee ~/le-wm/true_motion_fetch.log"'
```

Expected: ~1.5h. Output: `$STABLEWM_HOME/true_motion_vitb16emb.h5` (~5GB, 1.2M frames × 768d embeddings + 6d actions)

- [ ] **Step 3: Verify the dataset**

```bash
ssh -x RD-H100-2 'cd ~/le-wm && source .venv/bin/activate && python3 -c "
import h5py
f = h5py.File(\"/home/ubuntu/.stable_worldmodel/true_motion_vitb16emb.h5\", \"r\")
for k in f.keys():
    print(f\"  {k}: {f[k].shape} {f[k].dtype}\")
print(f\"Episodes: {len(f[\"ep_len\"][:])}\")
print(f\"Total frames: {f[\"ep_len\"][:].sum()}\")
f.close()
"'
```

Expected: ~2700 episodes, ~1.2M frames, embedding (N, 768) float32, action (N, 6) float32

- [ ] **Step 4: Commit**

```bash
git add scripts/fetch_and_encode_true_motion.py
git commit -m "feat: add streaming fetch+encode pipeline for filtered SDP10 data"
```

---

### Task 2: Train Probe with SIGReg Sweep

**Files:**
- None new (uses existing `train_probe.py`)

- [ ] **Step 1: Run main probe (sigreg=0.5, 100 epochs, save every 10)**

```bash
ssh -x RD-H100-2 'tmux new-session -d -s tm_probe "cd ~/le-wm && source .venv/bin/activate && export STABLEWM_HOME=\$HOME/.stable_worldmodel && export HDF5_PLUGIN_PATH=\$(python3 -c \"import hdf5plugin; print(hdf5plugin.PLUGINS_PATH)\") && python train_probe.py --dataset true_motion_vitb16emb --embed-dim 768 --proj-dim 192 --epochs 100 --batch-size 256 --lr 5e-4 --sigreg-weight 0.5 --frameskip 5 --action-dim 6 --save-every 10 --task-name true-motion-probe-sr05 2>&1 | tee ~/le-wm/tm_probe_sr05.log && for SR in 0.01 0.1 0.25 1.0 2.0; do echo \"=== sigreg=\$SR ===\" && python train_probe.py --dataset true_motion_vitb16emb --embed-dim 768 --proj-dim 192 --epochs 50 --batch-size 256 --lr 5e-4 --sigreg-weight \$SR --frameskip 5 --action-dim 6 --task-name true-motion-probe-sr\$SR 2>&1 | tail -5; done"'
```

Expected: ~30 min total. Main probe: 100 epochs with checkpoints at 10, 20, ..., 100. Sweep: 5 configs × 50 epochs.

- [ ] **Step 2: Verify results**

```bash
ssh -x RD-H100-2 'tail -5 ~/le-wm/tm_probe_sr05.log && ls ~/.stable_worldmodel/true-motion-probe-*'
```

Expected: Final copy ratio < 0.5, multiple probe checkpoint files.

---

### Task 3: Assemble Checkpoint + CEM Evaluation

**Files:**
- Create: `config/train/data/true_motion.yaml`

- [ ] **Step 1: Create data config**

```yaml
dataset:
  num_steps: ${eval:'${wm.num_preds} + ${wm.history_size}'}
  frameskip: 5
  name: true_motion_vitb16emb
  keys_to_load:
    - embedding
    - action
  keys_to_cache:
    - action
```

- [ ] **Step 2: Assemble best probe into JEPA checkpoint**

```bash
ssh -x RD-H100-2 'cd ~/le-wm && source .venv/bin/activate && export STABLEWM_HOME=$HOME/.stable_worldmodel && python scripts/assemble_checkpoint.py \
  --probe-components $STABLEWM_HOME/true-motion-probe-sr05_probe_components.pt \
  --encoder-checkpoint $STABLEWM_HOME/vjepa_checkpoints/vitb16_pretrained.pt \
  --output $STABLEWM_HOME/true_motion_assembled_object.ckpt'
```

- [ ] **Step 3: Evaluate copy ratio at each saved epoch (10, 20, ..., 100)**

Write a small script that loads each epoch checkpoint and computes copy ratio on val set — this gives us the training curve.

```bash
ssh -x RD-H100-2 'cd ~/le-wm && source .venv/bin/activate && export STABLEWM_HOME=$HOME/.stable_worldmodel && export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGINS_PATH)") && python3 -c "
import torch, os, json, numpy as np, h5py
from train_probe import EmbeddingDataset
from module import ARPredictor, Embedder, MLP, SIGReg

cache = os.environ[\"STABLEWM_HOME\"]
device = \"cuda\"

# Val set
val_ds = EmbeddingDataset(
    os.path.join(cache, \"true_motion_vitb16emb.h5\"),
    num_steps=4, frameskip=5, split=\"val\"
)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4)

curve = []
for epoch in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    path = os.path.join(cache, f\"true-motion-probe-sr05_epoch{epoch}_probe_components.pt\")
    if not os.path.exists(path):
        path = os.path.join(cache, \"true-motion-probe-sr05_probe_components.pt\")
        if epoch != 100: continue

    probe = torch.load(path, map_location=\"cpu\")
    cfg = probe[\"config\"]
    ead = cfg[\"frameskip\"] * cfg[\"action_dim\"]

    proj = MLP(input_dim=768, output_dim=192, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d).to(device)
    pred = ARPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0).to(device)
    pp = MLP(input_dim=192, output_dim=192, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d).to(device)
    ae = Embedder(input_dim=ead, smoothed_dim=max(ead, 10), emb_dim=192).to(device)
    proj.load_state_dict(probe[\"projector\"]); pred.load_state_dict(probe[\"predictor\"])
    pp.load_state_dict(probe[\"pred_proj\"]); ae.load_state_dict(probe[\"action_encoder\"])
    for m in [proj, pred, pp, ae]: m.eval()
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)

    tp, tc, ts, n = 0, 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            e = batch[\"embedding\"].to(device); a = torch.nan_to_num(batch[\"action\"].to(device), 0)
            B, T, D = e.shape
            emb = proj(e.reshape(B*T, D)).reshape(B, T, -1)
            act = ae(a)
            ctx_e, ctx_a = emb[:, :3], act[:, :3]
            tgt = emb[:, 1:]
            pr = pp(pred(ctx_e, ctx_a)[:, :3].reshape(-1, 192)).reshape(B, 3, 192)
            tp += (pr - tgt).pow(2).mean().item()
            tc += (ctx_e[:, -1:].expand_as(tgt) - tgt).pow(2).mean().item()
            ts += sigreg(emb.transpose(0, 1)).item()
            n += 1

    ratio = (tp/n) / (tc/n + 1e-8)
    curve.append({\"epoch\": epoch, \"pred_loss\": tp/n, \"copy_ratio\": ratio, \"sigreg\": ts/n})
    print(f\"  Epoch {epoch}: ratio={ratio:.4f}, pred={tp/n:.6f}, sreg={ts/n:.3f}\")

with open(os.path.join(cache, \"true_motion_training_curve.json\"), \"w\") as f:
    json.dump(curve, f)
print(\"Saved training curve\")
"'
```

- [ ] **Step 4: Commit**

```bash
git add config/train/data/true_motion.yaml
git commit -m "feat: add true motion data config and training curve evaluation"
```

---

### Task 4: Generate Rollout Visualization Data

**Files:**
- None new — reuse patterns from `scripts/humanoid_viz_data.py` and `scripts/record_pusht_viz.py`

- [ ] **Step 1: Generate dataset episode frames + rollout embeddings at multiple horizons**

Since we don't have raw pixels stored, we need to re-download a small sample of videos for visualization. Fetch ~8 episodes worth of frames (just for display), compute rollouts at H=3,5,10,15.

```bash
# This combines: sample frames for viz + compute embedding rollouts at multiple horizons
# Output: true_motion_viz_data.json (~1MB)
```

The script should:
1. Pick 8 random episodes from the embedding dataset
2. For each, download the original video frames (small subset, just for viz)
3. Compute rollout embeddings at horizons 3, 5, 10, 15
4. Compare rollout MSE vs copy baseline
5. Save as JSON with base64 frames + rollout metrics

- [ ] **Step 2: Run and verify**

Expected: `$STABLEWM_HOME/true_motion_viz_data.json` with 8 episodes, 4 horizons each.

---

### Task 5: Build HTML Explorer

**Files:**
- Uses: `scripts/build_explorers.py`

- [ ] **Step 1: Prepare results JSON in the format build_explorers.py expects**

Combine training curves, sigreg sweep results, rollout data, and dataset episodes into the expected format.

- [ ] **Step 2: Generate explorer**

```bash
python scripts/build_explorers.py --results-dir $STABLEWM_HOME --out-dir docs/
```

Or manually create `docs/truemotion-explorer.html` from the template.

- [ ] **Step 3: Verify with Playwright MCP**

Navigate, check all 4 tabs render, check training curve SVGs show the copy ratio dropping over epochs.

- [ ] **Step 4: Commit**

```bash
git add docs/truemotion-explorer.html
git commit -m "docs: add true motion explorer with training curves and rollouts"
```

---

### Task 6: Master Orchestrator

**Files:**
- Create: `scripts/run_true_motion_experiment.sh`

- [ ] **Step 1: Write orchestrator that chains everything**

```bash
#!/bin/bash
# True Motion Experiment — filtered SDP10 data (animation_radius > 200cm)
# Usage: tmux new-session -s truemotion "bash scripts/run_true_motion_experiment.sh"

set -euo pipefail
cd ~/le-wm
source .venv/bin/activate
export STABLEWM_HOME=$HOME/.stable_worldmodel
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGINS_PATH)")

LOG=~/le-wm/true_motion_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1

echo "=== True Motion Experiment (radius > 200cm) ==="
echo "Started: $(date)"

# Phase 1: Fetch + encode (~1.5h)
echo ">>> Phase 1: Streaming fetch + encode..."
python scripts/fetch_and_encode_true_motion.py \
    --min-radius 200 --max-items 2722 \
    --output true_motion_vitb16emb.h5 \
    --checkpoint $STABLEWM_HOME/vjepa_checkpoints/vitb16_pretrained.pt

# Phase 2: Probe training + sweep (~30 min)
echo ">>> Phase 2: Probe training..."
python train_probe.py \
    --dataset true_motion_vitb16emb --embed-dim 768 --proj-dim 192 \
    --epochs 100 --batch-size 256 --lr 5e-4 \
    --sigreg-weight 0.5 --frameskip 5 --action-dim 6 \
    --save-every 10 --task-name true-motion-probe-sr05

for SR in 0.01 0.1 0.25 1.0 2.0; do
    echo "  Sweep sigreg=$SR..."
    python train_probe.py \
        --dataset true_motion_vitb16emb --embed-dim 768 --proj-dim 192 \
        --epochs 50 --batch-size 256 --lr 5e-4 \
        --sigreg-weight $SR --frameskip 5 --action-dim 6 \
        --task-name true-motion-probe-sr$SR || true
done

# Phase 3: Assemble checkpoint
echo ">>> Phase 3: Assemble checkpoint..."
python scripts/assemble_checkpoint.py \
    --probe-components $STABLEWM_HOME/true-motion-probe-sr05_probe_components.pt \
    --encoder-checkpoint $STABLEWM_HOME/vjepa_checkpoints/vitb16_pretrained.pt \
    --output $STABLEWM_HOME/true_motion_assembled_object.ckpt

# Phase 4: Training curve evaluation (~10 min)
echo ">>> Phase 4: Evaluate training curve..."
# (inline python from Task 3 Step 3)

echo ""
echo "=== Experiment Complete: $(date) ==="
```

- [ ] **Step 2: Launch on H200**

```bash
rsync -avz scripts/ RD-H100-2:~/le-wm/scripts/
ssh -x RD-H100-2 'tmux new-session -d -s truemotion "cd ~/le-wm && bash scripts/run_true_motion_experiment.sh"'
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_true_motion_experiment.sh
git commit -m "feat: add true motion experiment orchestrator"
```

---

## Expected Results

| Metric | Estimate |
|--------|----------|
| Dataset | ~2,700 episodes, ~1.2M frames, animation_radius > 200cm |
| Embedding file | ~5 GB (768d × 1.2M frames) |
| Best probe copy ratio | < 0.3 (based on 200-episode results of 0.33) |
| Training curve | Copy ratio dropping from ~1.0 → < 0.3 over 100 epochs |
| SIGReg sweep | 6 configs (0.01, 0.1, 0.25, 0.5, 1.0, 2.0) |
| Visualization | HTML explorer with rollouts at H=3,5,10,15 |
| Total compute | ~3h on H200 |
