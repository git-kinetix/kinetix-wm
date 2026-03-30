# Multi-Environment LeWM Training Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train LeWM world models across all 5 environments (PushT, TwoRoom, Cube, Reacher, Humanoid) with 3 architectural variations each, log training dynamics, generate CEM planning rollouts, compare against pretrained baselines, and build per-environment HTML explorers.

**Architecture:** Each environment gets 3 training runs: (A) ViT-Tiny from scratch (paper baseline), (B) Frozen ViT-B/16 probe with sigreg=0.5, (C) Frozen ViT-B/16 probe with best sigreg from sweep. Pretrained LeWM checkpoints from HuggingFace serve as reference. CEM planning rollouts are captured at epochs 10, 50, 100 and rendered into interactive HTML explorers with loss curves, copy ratio, and playable videos.

**Tech Stack:** PyTorch Lightning, stable_worldmodel, stable_pretraining, Hydra, ClearML (self-hosted), MuJoCo (for DMControl), Playwright (verification)

**Compute:** H200 GPU (144 GB VRAM), 12h budget = 720 minutes

---

## Compute Budget Allocation

| Phase | Environments | Est. Time | Details |
|-------|-------------|-----------|---------|
| 1. Data download & prep | All 5 | 30 min | Download HF datasets, collect humanoid, precompute embeddings |
| 2. ViT-Tiny training | PushT, TwoRoom | 4h | 30 epochs each (~38 min/epoch for PushT) |
| 3. ViT-Tiny training | Cube, Reacher, Humanoid | 1.5h | 100 epochs each (smaller datasets) |
| 4. Precompute ViT-B/16 embeddings | All 5 | 30 min | 512 batch, one pass per dataset |
| 5. Probe training (sigreg=0.5) | All 5 | 30 min | 100 epochs each, ~5 min per env |
| 6. Probe sigreg sweep (best) | All 5 | 1.5h | 5 sigreg values × 50 epochs × 5 envs |
| 7. Download pretrained baselines | PushT, TwoRoom, Cube, Reacher | 15 min | HuggingFace models |
| 8. Eval + CEM rollouts | All 5 × 3 configs | 1h | 6 episodes per config |
| 9. Build HTML explorers | All 5 | 30 min | Automated script |
| 10. Final comparison eval | All configs | 30 min | Copy ratio on val set |
| **Total** | | **~10.5h** | 1.5h buffer |

---

## File Structure

### New files to create:
```
scripts/
  multi_env_train.sh          # Master orchestrator (runs everything in tmux)
  download_all_datasets.py    # Download all HF datasets + collect humanoid
  precompute_all_embeddings.py # Precompute ViT-B/16 for all datasets
  eval_all.py                 # Evaluate all checkpoints, capture rollouts
  build_explorers.py          # Generate per-environment HTML explorers

config/train/data/
  tworoom.yaml                # Already exists
  reacher.yaml                # Already exists (as dmc.yaml)
  cube.yaml                   # Already exists (as ogb.yaml)
  humanoid.yaml               # Already exists

config/eval/
  humanoid.yaml               # Already created
  (pusht.yaml, tworoom.yaml, cube.yaml, reacher.yaml already exist)

docs/
  pusht-explorer.html         # Already exists
  humanoid-explorer.html      # Already exists
  tworoom-explorer.html       # To create
  cube-explorer.html          # To create
  reacher-explorer.html       # To create
  comparison.html             # Cross-environment comparison dashboard
```

### Files to modify:
```
train_probe.py               # Add --save-every flag for checkpoint at epochs 10, 50, 100
scripts/precompute_embeddings.py  # Add batch mode for multiple datasets
```

---

## Task 1: Data Download & Preparation

**Files:**
- Create: `scripts/download_all_datasets.py`
- Modify: none

- [ ] **Step 1: Write download script**

```python
#!/usr/bin/env python3
"""Download all LeWM datasets from HuggingFace + collect humanoid from DMControl."""

import os, subprocess, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

CACHE = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))

DATASETS = {
    "pusht_expert_train": {"repo": "quentinll/lewm-pusht", "file": "pusht_expert_train.h5.zst", "ext": ".h5"},
    "tworoom": {"repo": "quentinll/lewm-tworooms", "file": "tworoom.tar.zst", "ext": ".tar"},
    "cube_single_expert": {"repo": "quentinll/lewm-cube", "file": "cube_single_expert.tar.zst", "ext": ".tar"},
    "reacher": {"repo": "quentinll/lewm-reacher", "file": "reacher.tar.zst", "ext": ".tar"},
}

def download_dataset(name, info):
    h5_path = os.path.join(CACHE, name + ".h5")
    if os.path.exists(h5_path):
        print(f"  {name}: already exists ({os.path.getsize(h5_path)/1e9:.1f} GB)")
        return

    print(f"  {name}: downloading from {info['repo']}...")
    path = hf_hub_download(info["repo"], info["file"], repo_type="dataset", local_dir=CACHE)

    if info["ext"] == ".h5":
        out = os.path.join(CACHE, name + ".h5")
        subprocess.run(["zstd", "-d", path, "-o", out], check=True)
        os.remove(path)
    elif info["ext"] == ".tar":
        subprocess.run(["tar", "--zstd", "-xvf", path, "-C", CACHE], check=True)
        os.remove(path)

    print(f"  {name}: done")

def collect_humanoid():
    h5_path = os.path.join(CACHE, "humanoid.h5")
    if os.path.exists(h5_path):
        print(f"  humanoid: already exists ({os.path.getsize(h5_path)/1e9:.1f} GB)")
        return

    print("  humanoid: collecting 200 episodes from DMControl...")
    os.environ["MUJOCO_GL"] = "egl"
    import stable_worldmodel as swm
    world = swm.World(
        env_name="swm/HumanoidDMControl-v0", num_envs=1,
        image_shape=(224, 224), max_episode_steps=200, goal_conditioned=False,
    )
    policy = swm.policy.RandomPolicy()
    world.set_policy(policy)
    world.record_dataset(dataset_name="humanoid", episodes=200, seed=42)
    print("  humanoid: done")

if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    for name, info in DATASETS.items():
        download_dataset(name, info)
    collect_humanoid()
    print("\nAll datasets ready.")
```

- [ ] **Step 2: Run on H200**

```bash
ssh RD-H100-2 'tmux new-session -d -s data "cd ~/le-wm && source .venv/bin/activate && export STABLEWM_HOME=\$HOME/.stable_worldmodel && python scripts/download_all_datasets.py 2>&1 | tee ~/le-wm/data_download.log"'
```
Expected: ~30 min. All 5 datasets in `$STABLEWM_HOME/`.

- [ ] **Step 3: Verify all datasets**

```bash
ssh RD-H100-2 'ls -lhS ~/.stable_worldmodel/*.h5'
```
Expected: pusht_expert_train.h5, tworoom.h5, cube_single_expert.h5 (or similar), reacher.h5, humanoid.h5

- [ ] **Step 4: Commit**

```bash
git add scripts/download_all_datasets.py
git commit -m "feat: add multi-dataset download script"
```

---

## Task 2: Precompute ViT-B/16 Embeddings for All Datasets

**Files:**
- Create: `scripts/precompute_all_embeddings.py`

- [ ] **Step 1: Write batch precompute script**

```python
#!/usr/bin/env python3
"""Precompute ViT-B/16 embeddings for all datasets."""

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

CACHE = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))
CKPT = os.path.join(CACHE, "vjepa_checkpoints", "vitb16_pretrained.pt")

DATASETS = {
    "pusht_expert_train": {"keys": ["pixels", "action", "proprio", "state"]},
    "tworoom": {"keys": ["pixels", "action", "proprio"]},
    "reacher": {"keys": ["pixels", "action", "observation"]},
    "humanoid": {"keys": ["pixels", "action", "observation"]},
    # cube may need keys_to_merge — check after download
}

def main():
    # Ensure ViT-B/16 checkpoint exists
    if not os.path.exists(CKPT):
        print("Creating ViT-B/16 checkpoint...")
        import timm, torch
        os.makedirs(os.path.dirname(CKPT), exist_ok=True)
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        torch.save({"model": model.state_dict()}, CKPT)

    for name in DATASETS:
        out = os.path.join(CACHE, f"{name}_vitb16emb.h5")
        if os.path.exists(out):
            print(f"{name}: embeddings already exist")
            continue

        h5_path = os.path.join(CACHE, f"{name}.h5")
        if not os.path.exists(h5_path):
            print(f"{name}: dataset not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Precomputing embeddings for {name}...")
        os.system(f"python scripts/precompute_embeddings.py "
                  f"--checkpoint {CKPT} --dataset {name} "
                  f"--output {name}_vitb16emb.h5 --batch-size 512")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on H200**

```bash
ssh RD-H100-2 'tmux new-session -d -s embed "cd ~/le-wm && source .venv/bin/activate && export STABLEWM_HOME=\$HOME/.stable_worldmodel && export HDF5_PLUGIN_PATH=\$(python3 -c \"import hdf5plugin; print(hdf5plugin.PLUGINS_PATH)\") && python scripts/precompute_all_embeddings.py 2>&1 | tee ~/le-wm/embed.log"'
```
Expected: ~30 min total. PushT (~15 min for 2.3M frames), others faster.

- [ ] **Step 3: Commit**

```bash
git add scripts/precompute_all_embeddings.py
git commit -m "feat: add batch embedding precomputation"
```

---

## Task 3: ViT-Tiny Training (Paper Baseline)

**Files:**
- Create: `scripts/multi_env_train.sh`

- [ ] **Step 1: Write master training script**

```bash
#!/bin/bash
# Multi-environment training orchestrator
# Runs in tmux, logs to individual files

set -e
cd ~/le-wm
source .venv/bin/activate
export STABLEWM_HOME=$HOME/.stable_worldmodel
export MUJOCO_GL=egl
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGINS_PATH)")

# Phase A: ViT-Tiny from scratch
echo "=== Phase A: ViT-Tiny Training ==="

# Large datasets (fewer epochs due to time)
for ENV in pusht tworoom; do
    EPOCHS=30
    echo "[$(date)] Starting ViT-Tiny $ENV ($EPOCHS epochs)..."
    python train.py data=$ENV trainer.max_epochs=$EPOCHS \
        wandb.enabled=false clearml.enabled=false \
        loader.batch_size=128 loader.num_workers=8 seed=42 \
        output_model_name=lewm_${ENV}_tiny \
        2>&1 | tee ~/le-wm/train_${ENV}_tiny.log || true
    echo "[$(date)] Finished ViT-Tiny $ENV"
done

# Small datasets (more epochs)
for ENV_CFG in "humanoid:+wm.action_dim=21" "dmc:" "ogb:"; do
    ENV=$(echo $ENV_CFG | cut -d: -f1)
    EXTRA=$(echo $ENV_CFG | cut -d: -f2)
    EPOCHS=100
    echo "[$(date)] Starting ViT-Tiny $ENV ($EPOCHS epochs)..."
    python train.py data=$ENV trainer.max_epochs=$EPOCHS $EXTRA \
        wandb.enabled=false clearml.enabled=false \
        loader.batch_size=128 loader.num_workers=8 seed=42 \
        output_model_name=lewm_${ENV}_tiny \
        2>&1 | tee ~/le-wm/train_${ENV}_tiny.log || true
    echo "[$(date)] Finished ViT-Tiny $ENV"
done

echo "=== Phase A Complete ==="
```

- [ ] **Step 2: Launch on H200 in tmux**

```bash
rsync -avz scripts/multi_env_train.sh RD-H100-2:~/le-wm/scripts/
ssh RD-H100-2 'chmod +x ~/le-wm/scripts/multi_env_train.sh && tmux new-session -d -s train "bash ~/le-wm/scripts/multi_env_train.sh 2>&1 | tee ~/le-wm/multi_train.log"'
```

- [ ] **Step 3: Commit**

```bash
git add scripts/multi_env_train.sh
git commit -m "feat: add multi-environment training orchestrator"
```

---

## Task 4: Probe Training (All Environments)

**Files:**
- Modify: `train_probe.py` (add `--save-every` flag)
- Create: `scripts/train_all_probes.sh`

- [ ] **Step 1: Add `--save-every` to train_probe.py**

Add argument and checkpoint saving at specified epoch intervals (for rollout visualization at different training stages).

```python
parser.add_argument("--save-every", type=int, default=0,
                    help="Save checkpoint every N epochs (0=only final)")
```

In the training loop, after each epoch:
```python
if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
    ep_path = os.path.join(cache_dir, f"{args.task_name}_epoch{epoch+1}_probe_components.pt")
    torch.save({...}, ep_path)
```

- [ ] **Step 2: Write probe training sweep script**

```bash
#!/bin/bash
# Train probes for all environments with sigreg=0.5 and sweep

cd ~/le-wm
source .venv/bin/activate
export STABLEWM_HOME=$HOME/.stable_worldmodel
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGINS_PATH)")

ENVS=("pusht_expert_train:2:5" "tworoom:2:5" "reacher:2:5" "humanoid:21:5")
# Format: dataset:action_dim:frameskip

# (cube needs special handling for action dim — check after download)

for ENV_CFG in "${ENVS[@]}"; do
    DS=$(echo $ENV_CFG | cut -d: -f1)
    ADIM=$(echo $ENV_CFG | cut -d: -f2)
    FS=$(echo $ENV_CFG | cut -d: -f3)

    EMB="${DS}_vitb16emb"

    if [ ! -f "$STABLEWM_HOME/${EMB}.h5" ]; then
        echo "Skipping $DS — no embeddings found"
        continue
    fi

    # Probe with sigreg=0.5 (known good)
    echo "[$(date)] Probe $DS sigreg=0.5..."
    python train_probe.py \
        --dataset $EMB --embed-dim 768 --proj-dim 192 \
        --epochs 100 --batch-size 256 --lr 5e-4 \
        --sigreg-weight 0.5 --frameskip $FS --action-dim $ADIM \
        --save-every 10 \
        --task-name "${DS}-probe-sr05" \
        2>&1 | tee ~/le-wm/probe_${DS}_sr05.log || true

    # Sigreg sweep: 0.01, 0.1, 0.5, 1.0, 2.0
    for SR in 0.01 0.1 1.0 2.0; do
        echo "[$(date)] Probe $DS sigreg=$SR..."
        python train_probe.py \
            --dataset $EMB --embed-dim 768 --proj-dim 192 \
            --epochs 50 --batch-size 256 --lr 5e-4 \
            --sigreg-weight $SR --frameskip $FS --action-dim $ADIM \
            --task-name "${DS}-probe-sr${SR}" \
            2>&1 | tee ~/le-wm/probe_${DS}_sr${SR}.log || true
    done
done

echo "=== All probes complete ==="
```

- [ ] **Step 3: Launch after embeddings are done**

```bash
ssh RD-H100-2 'tmux new-session -d -s probes "bash ~/le-wm/scripts/train_all_probes.sh"'
```

- [ ] **Step 4: Commit**

```bash
git add train_probe.py scripts/train_all_probes.sh
git commit -m "feat: add multi-env probe training with sweep"
```

---

## Task 5: Download Pretrained Baselines

**Files:**
- Create: `scripts/download_pretrained.py`

- [ ] **Step 1: Write download script**

Download the 4 pretrained LeWM checkpoints from HuggingFace and convert to format `AutoCostModel` expects.

```python
#!/usr/bin/env python3
"""Download pretrained LeWM checkpoints from HuggingFace."""

from huggingface_hub import hf_hub_download
import os, torch

CACHE = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))

MODELS = {
    "pusht/lewm": "quentinll/lewm-pusht",
    "tworoom/lewm": "quentinll/lewm-tworooms",
    "cube/lewm": "quentinll/lewm-cube",
    "reacher/lewm": "quentinll/lewm-reacher",
}

for policy_path, repo_id in MODELS.items():
    out_dir = os.path.join(CACHE, os.path.dirname(policy_path))
    out_file = os.path.join(out_dir, os.path.basename(policy_path) + "_object.ckpt")

    if os.path.exists(out_file):
        print(f"  {policy_path}: already exists")
        continue

    os.makedirs(out_dir, exist_ok=True)

    # Download weights
    weights_path = hf_hub_download(repo_id, "weights.pt", local_dir=out_dir)
    config_path = hf_hub_download(repo_id, "config.json", local_dir=out_dir)

    print(f"  {policy_path}: downloaded weights + config")

    # The AutoCostModel expects *_object.ckpt — a pickled model object
    # We need to reconstruct the model from weights + config
    # For now, just note the weights location
    print(f"    Weights at: {weights_path}")
    print(f"    Config at: {config_path}")
```

- [ ] **Step 2: Run and commit**

```bash
ssh RD-H100-2 'cd ~/le-wm && source .venv/bin/activate && export STABLEWM_HOME=$HOME/.stable_worldmodel && python scripts/download_pretrained.py'
git add scripts/download_pretrained.py
git commit -m "feat: add pretrained model downloader"
```

---

## Task 6: Evaluation + CEM Rollouts (All Environments)

**Files:**
- Create: `scripts/eval_all.py`

- [ ] **Step 1: Write evaluation script**

For each environment × architecture config:
1. Load the model checkpoint
2. Run `eval.py` with CEM planning
3. Capture rollout videos/frames
4. Compute copy ratio on val set
5. Save results JSON

This produces per-environment result files with:
- Copy ratio at each checkpoint epoch
- CEM rollout frames
- Loss curves (from training logs)
- Comparison with pretrained baseline

- [ ] **Step 2: Run evaluations**

```bash
ssh RD-H100-2 'tmux new-session -d -s eval "cd ~/le-wm && source .venv/bin/activate && export STABLEWM_HOME=\$HOME/.stable_worldmodel && export MUJOCO_GL=egl && python scripts/eval_all.py 2>&1 | tee ~/le-wm/eval_all.log"'
```

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_all.py
git commit -m "feat: add comprehensive evaluation pipeline"
```

---

## Task 7: Build HTML Explorers (All Environments)

**Files:**
- Create: `scripts/build_explorers.py`
- Create: `docs/tworoom-explorer.html`, `docs/cube-explorer.html`, `docs/reacher-explorer.html`
- Create: `docs/comparison.html`

- [ ] **Step 1: Write explorer generator**

Templated HTML builder that takes a results JSON and produces a Distill-style explorer with:
- **Tab 1: CEM Rollouts** — playable frame sequences from CEM planning
- **Tab 2: Training Curves** — loss, copy ratio, sigreg loss vs epoch (inline SVG charts)
- **Tab 3: Architecture Comparison** — side-by-side ViT-Tiny vs Probe vs Pretrained
- **Tab 4: Dataset Viewer** — playable episodes from the dataset

- [ ] **Step 2: Build comparison dashboard**

`docs/comparison.html` — cross-environment view:
- Table: environment × architecture → copy ratio
- Bar charts comparing all 15 configurations (5 envs × 3 archs)
- Best config per environment highlighted

- [ ] **Step 3: Generate all explorers**

```bash
python scripts/build_explorers.py --env pusht --env tworoom --env cube --env reacher --env humanoid
```

- [ ] **Step 4: Verify with Playwright MCP**

Navigate to each explorer, take screenshots, verify all tabs work.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_explorers.py docs/*-explorer.html docs/comparison.html
git commit -m "feat: add per-environment explorers and comparison dashboard"
```

---

## Task 8: Master Orchestrator

**Files:**
- Create: `scripts/run_full_experiment.sh`

- [ ] **Step 1: Write the 12h orchestrator**

Single script that runs everything in sequence within a tmux session:

```bash
#!/bin/bash
# Full 12h experiment run
# Usage: tmux new-session -d -s experiment "bash scripts/run_full_experiment.sh"

set -e
LOG=~/le-wm/experiment_$(date +%Y%m%d_%H%M).log
exec > >(tee -a $LOG) 2>&1

echo "=== LeWM Multi-Environment Experiment ==="
echo "Started: $(date)"
echo "Compute budget: 12h"
echo ""

# Phase 1: Data (30 min)
echo ">>> Phase 1: Data download..."
python scripts/download_all_datasets.py

# Phase 2: Embeddings (30 min)
echo ">>> Phase 2: Precompute embeddings..."
python scripts/precompute_all_embeddings.py

# Phase 3: ViT-Tiny training (5.5h)
echo ">>> Phase 3: ViT-Tiny training..."
bash scripts/multi_env_train.sh

# Phase 4: Probe training (2h)
echo ">>> Phase 4: Probe training + sweep..."
bash scripts/train_all_probes.sh

# Phase 5: Download pretrained (15 min)
echo ">>> Phase 5: Download pretrained baselines..."
python scripts/download_pretrained.py

# Phase 6: Evaluation (1h)
echo ">>> Phase 6: Evaluation + CEM rollouts..."
python scripts/eval_all.py

# Phase 7: Build visualizations (30 min)
echo ">>> Phase 7: Build explorers..."
python scripts/build_explorers.py

echo ""
echo "=== Experiment Complete ==="
echo "Finished: $(date)"
echo "Results in docs/*-explorer.html"
```

- [ ] **Step 2: Launch**

```bash
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='.git' --exclude='*.ckpt' /home/omar/Repos/le-wm/ RD-H100-2:~/le-wm/
ssh RD-H100-2 'tmux new-session -d -s experiment "cd ~/le-wm && source .venv/bin/activate && export STABLEWM_HOME=\$HOME/.stable_worldmodel && export MUJOCO_GL=egl && export HDF5_PLUGIN_PATH=\$(python3 -c \"import hdf5plugin; print(hdf5plugin.PLUGINS_PATH)\") && bash scripts/run_full_experiment.sh"'
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_full_experiment.sh
git commit -m "feat: add 12h master experiment orchestrator"
```

---

## Expected Final Results

### Per-Environment Copy Ratios (target)

| Environment | ViT-Tiny (scratch) | Probe (sr=0.5) | Probe (best sr) | Pretrained |
|------------|-------------------|----------------|-----------------|------------|
| **PushT** | ~0.21 (1ep) | ~0.15 | ~0.10 | TBD |
| **TwoRoom** | TBD | TBD | TBD | TBD |
| **Cube** | TBD | TBD | TBD | TBD |
| **Reacher** | TBD | TBD | TBD | TBD |
| **Humanoid** | ~1.94 (8ep) | 0.12 | TBD | N/A |

### Deliverables
- 5 interactive HTML explorers (`docs/{env}-explorer.html`)
- 1 comparison dashboard (`docs/comparison.html`)
- 15 trained model checkpoints (5 envs × 3 architectures)
- 30+ CEM planning rollout videos
- Training curve data for all runs
- Git branch with organized commits
