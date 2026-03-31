#!/bin/bash
# =============================================================================
# LeWM Multi-Environment Experiment (12h compute budget)
# Run inside tmux: tmux new-session -s experiment "bash scripts/run_full_experiment.sh"
# =============================================================================

set -euo pipefail
cd ~/le-wm
source .venv/bin/activate
export STABLEWM_HOME=$HOME/.stable_worldmodel
export MUJOCO_GL=egl
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGINS_PATH)")

LOG=~/le-wm/experiment_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "  LeWM Multi-Environment Experiment"
echo "  Started: $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"
echo ""

# ── Phase 1: Data Download ──────────────────────────────────────
echo ">>> [$(date +%H:%M)] Phase 1: Download datasets..."
python scripts/download_all_datasets.py
echo ">>> [$(date +%H:%M)] Phase 1 complete."
echo ""

# ── Phase 2: Precompute Embeddings ──────────────────────────────
echo ">>> [$(date +%H:%M)] Phase 2: Precompute ViT-B/16 embeddings..."
python scripts/precompute_all_embeddings.py
echo ">>> [$(date +%H:%M)] Phase 2 complete."
echo ""

# ── Phase 3: ViT-Tiny Training ──────────────────────────────────
echo ">>> [$(date +%H:%M)] Phase 3: ViT-Tiny training (all envs)..."

# PushT (large dataset — 30 epochs)
echo "  [$(date +%H:%M)] PushT ViT-Tiny (30 epochs)..."
python train.py data=pusht trainer.max_epochs=30 \
    wandb.enabled=false clearml.enabled=false \
    loader.batch_size=128 loader.num_workers=8 seed=42 \
    output_model_name=lewm_pusht_tiny \
    2>&1 | tail -5 || echo "  PushT training failed/exited"

# TwoRoom (30 epochs)
echo "  [$(date +%H:%M)] TwoRoom ViT-Tiny (30 epochs)..."
python train.py data=tworoom trainer.max_epochs=30 \
    wandb.enabled=false clearml.enabled=false \
    loader.batch_size=128 loader.num_workers=8 seed=42 \
    output_model_name=lewm_tworoom_tiny \
    2>&1 | tail -5 || echo "  TwoRoom training failed/exited"

# Reacher (100 epochs — DMControl, smaller)
echo "  [$(date +%H:%M)] Reacher ViT-Tiny (100 epochs)..."
python train.py data=dmc trainer.max_epochs=100 \
    wandb.enabled=false clearml.enabled=false \
    loader.batch_size=128 loader.num_workers=8 seed=42 \
    output_model_name=lewm_reacher_tiny \
    2>&1 | tail -5 || echo "  Reacher training failed/exited"

# Cube (100 epochs — OGBench)
echo "  [$(date +%H:%M)] Cube ViT-Tiny (100 epochs)..."
python train.py data=ogb trainer.max_epochs=100 \
    wandb.enabled=false clearml.enabled=false \
    loader.batch_size=128 loader.num_workers=8 seed=42 \
    output_model_name=lewm_cube_tiny \
    2>&1 | tail -5 || echo "  Cube training failed/exited"

# Humanoid (100 epochs)
echo "  [$(date +%H:%M)] Humanoid ViT-Tiny (100 epochs)..."
python train.py data=humanoid +wm.action_dim=21 trainer.max_epochs=100 \
    wandb.enabled=false clearml.enabled=false \
    loader.batch_size=128 loader.num_workers=8 seed=42 \
    output_model_name=lewm_humanoid_tiny \
    2>&1 | tail -5 || echo "  Humanoid training failed/exited"

echo ">>> [$(date +%H:%M)] Phase 3 complete."
echo ""

# ── Phase 4: Probe Training + Sweep ─────────────────────────────
echo ">>> [$(date +%H:%M)] Phase 4: Probe training..."
bash scripts/train_all_probes.sh
echo ">>> [$(date +%H:%M)] Phase 4 complete."
echo ""

# ── Phase 5: Download Pretrained ─────────────────────────────────
echo ">>> [$(date +%H:%M)] Phase 5: Download pretrained baselines..."
python scripts/download_pretrained.py || echo "  Pretrained download failed"
echo ">>> [$(date +%H:%M)] Phase 5 complete."
echo ""

# ── Phase 6: Evaluation ─────────────────────────────────────────
echo ">>> [$(date +%H:%M)] Phase 6: Evaluation + rollouts..."
python scripts/eval_all.py || echo "  Eval failed"
echo ">>> [$(date +%H:%M)] Phase 6 complete."
echo ""

# ── Phase 7: Build Visualizations ────────────────────────────────
echo ">>> [$(date +%H:%M)] Phase 7: Build HTML explorers..."
python scripts/build_explorers.py || echo "  Build failed"
echo ">>> [$(date +%H:%M)] Phase 7 complete."
echo ""

echo "============================================================"
echo "  Experiment Complete!"
echo "  Finished: $(date)"
echo "  Log: $LOG"
echo "  Results: docs/*-explorer.html"
echo "============================================================"
