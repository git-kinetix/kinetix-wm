#!/bin/bash
# Sweep probe training variations on precomputed ViT-B/16 embeddings.
# Each trial takes ~5 min, so this whole sweep should be ~2 hours.
#
# Usage: setsid bash scripts/sweep_probe.sh > ~/le-wm/probe_sweep.log 2>&1 < /dev/null &

cd ~/le-wm
source .venv/bin/activate
export STABLEWM_HOME=$HOME/.stable_worldmodel

DATASET="prod_beta0_200ep_vitb16emb"
EPOCHS=100
EMBED=768

echo "=== Probe Sweep: precomputed ViT-B/16 embeddings ==="
echo "Dataset: $DATASET"
echo ""

run_probe() {
    local name="$1"; shift
    echo "[$(date)] Starting: $name"
    python train_probe.py \
        --dataset $DATASET \
        --embed-dim $EMBED \
        --epochs $EPOCHS \
        --num-workers 8 \
        --clearml \
        --project lewm/probe-sweep \
        --task-name "$name" \
        "$@" 2>&1 | grep -E "Epoch.*(0/|9/|19/|29/|49/|79/|99/|best)" | tail -15
    echo "[$(date)] Finished: $name"
    echo ""
}

# --- 1. SIGReg weight ---
echo "=== Group 1: SIGReg weight ==="
run_probe "sigreg=0.001"  --sigreg-weight 0.001 --proj-dim 192 --lr 5e-4 --batch-size 256
run_probe "sigreg=0.01"   --sigreg-weight 0.01  --proj-dim 192 --lr 5e-4 --batch-size 256
run_probe "sigreg=0.05"   --sigreg-weight 0.05  --proj-dim 192 --lr 5e-4 --batch-size 256
run_probe "sigreg=0.09"   --sigreg-weight 0.09  --proj-dim 192 --lr 5e-4 --batch-size 256
run_probe "sigreg=0.2"    --sigreg-weight 0.2   --proj-dim 192 --lr 5e-4 --batch-size 256
run_probe "sigreg=0.5"    --sigreg-weight 0.5   --proj-dim 192 --lr 5e-4 --batch-size 256

# --- 2. Projection dimension ---
echo "=== Group 2: Projection dimension ==="
run_probe "proj=64"   --proj-dim 64  --sigreg-weight 0.09 --lr 5e-4 --batch-size 256
run_probe "proj=128"  --proj-dim 128 --sigreg-weight 0.09 --lr 5e-4 --batch-size 256
run_probe "proj=192"  --proj-dim 192 --sigreg-weight 0.09 --lr 5e-4 --batch-size 256
run_probe "proj=384"  --proj-dim 384 --sigreg-weight 0.09 --lr 5e-4 --batch-size 256
run_probe "proj=768"  --proj-dim 768 --sigreg-weight 0.09 --lr 5e-4 --batch-size 256

# --- 3. Learning rate ---
echo "=== Group 3: Learning rate ==="
run_probe "lr=1e-4"   --lr 1e-4 --proj-dim 192 --sigreg-weight 0.09 --batch-size 256
run_probe "lr=3e-4"   --lr 3e-4 --proj-dim 192 --sigreg-weight 0.09 --batch-size 256
run_probe "lr=5e-4"   --lr 5e-4 --proj-dim 192 --sigreg-weight 0.09 --batch-size 256
run_probe "lr=1e-3"   --lr 1e-3 --proj-dim 192 --sigreg-weight 0.09 --batch-size 256
run_probe "lr=3e-3"   --lr 3e-3 --proj-dim 192 --sigreg-weight 0.09 --batch-size 256

# --- 4. Batch size ---
echo "=== Group 4: Batch size ==="
run_probe "bs=64"    --batch-size 64  --proj-dim 192 --sigreg-weight 0.09 --lr 5e-4
run_probe "bs=128"   --batch-size 128 --proj-dim 192 --sigreg-weight 0.09 --lr 5e-4
run_probe "bs=256"   --batch-size 256 --proj-dim 192 --sigreg-weight 0.09 --lr 5e-4
run_probe "bs=512"   --batch-size 512 --proj-dim 192 --sigreg-weight 0.09 --lr 5e-4

# --- 5. Frameskip ---
echo "=== Group 5: Frameskip ==="
run_probe "fs=1"   --frameskip 1  --proj-dim 192 --sigreg-weight 0.09 --lr 5e-4 --batch-size 256
run_probe "fs=3"   --frameskip 3  --proj-dim 192 --sigreg-weight 0.09 --lr 5e-4 --batch-size 256
run_probe "fs=5"   --frameskip 5  --proj-dim 192 --sigreg-weight 0.09 --lr 5e-4 --batch-size 256
run_probe "fs=10"  --frameskip 10 --proj-dim 192 --sigreg-weight 0.09 --lr 5e-4 --batch-size 256
run_probe "fs=15"  --frameskip 15 --proj-dim 192 --sigreg-weight 0.09 --lr 5e-4 --batch-size 256

# --- 6. Best combo candidates ---
echo "=== Group 6: Best combo candidates ==="
run_probe "best-v1_sr001-proj384-lr3e4" --sigreg-weight 0.001 --proj-dim 384 --lr 3e-4 --batch-size 256
run_probe "best-v2_sr005-proj384-lr5e4" --sigreg-weight 0.005 --proj-dim 384 --lr 5e-4 --batch-size 256
run_probe "best-v3_sr01-proj768-lr3e4"  --sigreg-weight 0.01  --proj-dim 768 --lr 3e-4 --batch-size 256
run_probe "best-v4_sr005-proj192-lr1e3" --sigreg-weight 0.005 --proj-dim 192 --lr 1e-3 --batch-size 256
run_probe "best-v5_sr001-proj768-lr5e4-fs10" --sigreg-weight 0.001 --proj-dim 768 --lr 5e-4 --batch-size 256 --frameskip 10

echo "=== Probe Sweep Complete: $(date) ==="
