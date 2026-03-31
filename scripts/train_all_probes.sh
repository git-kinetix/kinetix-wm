#!/bin/bash
# Train probes for all environments with sigreg=0.5 and sweep best sigreg.
#
# For each env:
#   1. Train probe with sigreg=0.5 for 100 epochs (save every 10)
#   2. Sweep sigreg values: 0.01, 0.1, 1.0, 2.0 at 50 epochs each
#
# Usage:
#   cd ~/le-wm && source .venv/bin/activate
#   export STABLEWM_HOME=$HOME/.stable_worldmodel
#   export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGINS_PATH)")
#   bash scripts/train_all_probes.sh 2>&1 | tee ~/le-wm/probe_all.log

set -o pipefail
cd ~/le-wm
source .venv/bin/activate
export STABLEWM_HOME=${STABLEWM_HOME:-$HOME/.stable_worldmodel}
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGINS_PATH)")

echo "=== Multi-Environment Probe Training ==="
echo "STABLEWM_HOME=$STABLEWM_HOME"
echo "Started: $(date)"
echo ""

# Format: dataset_name:action_dim:frameskip
ENVS=(
    "pusht_expert_train:2:5"
    "tworoom:2:5"
    "reacher:2:5"
    "humanoid:21:5"
    # cube: action_dim TBD after dataset inspection; uncomment when known
    # "cube_single_expert:??:5"
)

# Common probe training parameters
EMBED_DIM=768
PROJ_DIM=192
BATCH_SIZE=256
LR=5e-4

for ENV_CFG in "${ENVS[@]}"; do
    DS=$(echo "$ENV_CFG" | cut -d: -f1)
    ADIM=$(echo "$ENV_CFG" | cut -d: -f2)
    FS=$(echo "$ENV_CFG" | cut -d: -f3)

    EMB="${DS}_vitb16emb"

    if [ ! -f "$STABLEWM_HOME/${EMB}.h5" ]; then
        echo "[$(date)] SKIP $DS -- no embeddings found at $STABLEWM_HOME/${EMB}.h5"
        echo ""
        continue
    fi

    echo "============================================================"
    echo "  Dataset:    $DS"
    echo "  Embeddings: ${EMB}.h5"
    echo "  Action dim: $ADIM   Frameskip: $FS"
    echo "============================================================"
    echo ""

    # ------------------------------------------------------------------
    # Phase 1: Probe with sigreg=0.5 (100 epochs, save every 10)
    # ------------------------------------------------------------------
    echo "[$(date)] Probe $DS sigreg=0.5 (100 epochs, save-every 10)..."
    python train_probe.py \
        --dataset "$EMB" \
        --embed-dim $EMBED_DIM \
        --proj-dim $PROJ_DIM \
        --epochs 100 \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --sigreg-weight 0.5 \
        --frameskip "$FS" \
        --action-dim "$ADIM" \
        --save-every 10 \
        --task-name "${DS}-probe-sr05" \
        2>&1 | tee ~/le-wm/probe_${DS}_sr05.log || true

    echo "[$(date)] Done: $DS sigreg=0.5"
    echo ""

    # ------------------------------------------------------------------
    # Phase 2: Sigreg sweep (50 epochs each)
    # ------------------------------------------------------------------
    for SR in 0.01 0.1 1.0 2.0; do
        # Build a task name safe for filenames (replace . with nothing)
        SR_TAG=$(echo "$SR" | tr -d '.')
        echo "[$(date)] Probe $DS sigreg=$SR (50 epochs)..."
        python train_probe.py \
            --dataset "$EMB" \
            --embed-dim $EMBED_DIM \
            --proj-dim $PROJ_DIM \
            --epochs 50 \
            --batch-size $BATCH_SIZE \
            --lr $LR \
            --sigreg-weight "$SR" \
            --frameskip "$FS" \
            --action-dim "$ADIM" \
            --task-name "${DS}-probe-sr${SR}" \
            2>&1 | tee ~/le-wm/probe_${DS}_sr${SR_TAG}.log || true

        echo "[$(date)] Done: $DS sigreg=$SR"
        echo ""
    done
done

echo ""
echo "=== All probes complete: $(date) ==="
