#!/bin/bash
# Overnight sweep: train probe variants, assemble JEPA checkpoints, run planning.
# ~12 hours total on H200.
#
# Usage: setsid bash scripts/overnight_sweep.sh > ~/le-wm/overnight.log 2>&1 < /dev/null &

cd ~/le-wm
source .venv/bin/activate
export STABLEWM_HOME=$HOME/.stable_worldmodel
export PATH="$HOME/.local/bin:$PATH"

DATASET="prod_beta0_200ep_vitb16emb"
ENCODER_CKPT="$HOME/.stable_worldmodel/vjepa_checkpoints/vitb16_pretrained.pt"
EMBED=768
EPOCHS=100
PLAN_EPS=16
PLAN_HORIZONS="3,5,10,15"
CEM_SAMPLES=256
CEM_ITERS=5

echo "=== Overnight Sweep: $(date) ==="
echo ""

# Define all configs: name, sigreg, proj_dim, lr, batch_size, frameskip
CONFIGS=(
    # Group A: SIGReg sweep (best range)
    "sr0.09-p192    0.09 192 5e-4 256 5"
    "sr0.2-p192     0.2  192 5e-4 256 5"
    "sr0.5-p192     0.5  192 5e-4 256 5"
    "sr1.0-p192     1.0  192 5e-4 256 5"

    # Group B: Projection dim with best sigreg
    "sr0.5-p64      0.5  64  5e-4 256 5"
    "sr0.5-p128     0.5  128 5e-4 256 5"
    "sr0.5-p384     0.5  384 5e-4 256 5"
    "sr0.5-p768     0.5  768 5e-4 256 5"

    # Group C: Learning rate with best sigreg
    "sr0.5-lr1e4    0.5  192 1e-4 256 5"
    "sr0.5-lr1e3    0.5  192 1e-3 256 5"
    "sr0.5-lr3e3    0.5  192 3e-3 256 5"

    # Group D: Frameskip
    "sr0.5-fs1      0.5  192 5e-4 256 1"
    "sr0.5-fs3      0.5  192 5e-4 256 3"
    "sr0.5-fs10     0.5  192 5e-4 256 10"

    # Group E: Best combos
    "sr0.5-p384-lr1e3       0.5  384 1e-3 256 5"
    "sr0.5-p768-lr3e4       0.5  768 3e-4 256 5"
    "sr1.0-p384             1.0  384 5e-4 256 5"
    "sr0.2-p384             0.2  384 5e-4 256 5"
    "sr0.5-p384-fs10        0.5  384 5e-4 256 10"
    "sr0.5-p768-lr1e3-fs10  0.5  768 1e-3 256 10"

    # Group F: Batch size
    "sr0.5-bs64     0.5  192 5e-4 64  5"
    "sr0.5-bs512    0.5  192 5e-4 512 5"

    # Group G: Higher sigreg combos
    "sr2.0-p192     2.0  192 5e-4 256 5"
    "sr1.0-p768     1.0  768 5e-4 256 5"
)

TOTAL=${#CONFIGS[@]}
echo "Total configs: $TOTAL"
echo "Estimated time: ~$((TOTAL * 20)) minutes (~$((TOTAL * 20 / 60)) hours)"
echo ""

for i in "${!CONFIGS[@]}"; do
    IFS=' ' read -r NAME SR PROJ LR BS FS <<< "${CONFIGS[$i]}"
    echo "================================================================"
    echo "[$((i+1))/$TOTAL] $NAME (sigreg=$SR proj=$PROJ lr=$LR bs=$BS fs=$FS)"
    echo "================================================================"

    # Step 1: Train probe (~5 min)
    echo "[$(date)] Training probe..."
    python train_probe.py \
        --dataset $DATASET \
        --embed-dim $EMBED \
        --proj-dim $PROJ \
        --epochs $EPOCHS \
        --batch-size $BS \
        --lr $LR \
        --sigreg-weight $SR \
        --frameskip $FS \
        --action-dim 6 \
        --num-workers 8 \
        --task-name "$NAME" 2>&1 | grep -E "Epoch.*(0/|49/|99/|best)" | tail -5

    COMP="$STABLEWM_HOME/${NAME}_probe_components.pt"
    if [ ! -f "$COMP" ]; then
        echo "[$(date)] ERROR: probe components not saved, skipping"
        continue
    fi

    # Step 2: Assemble JEPA checkpoint (~10s)
    echo "[$(date)] Assembling checkpoint..."
    CKPT="$STABLEWM_HOME/lewm_${NAME}_object.ckpt"
    python scripts/assemble_checkpoint.py \
        --probe-components "$COMP" \
        --encoder-checkpoint "$ENCODER_CKPT" \
        --output "$CKPT" 2>&1 | tail -3

    if [ ! -f "$CKPT" ]; then
        echo "[$(date)] ERROR: checkpoint assembly failed, skipping"
        continue
    fi

    # Step 3: Run planning with GT comparison (~10 min)
    echo "[$(date)] Running planning..."
    python scripts/plan_batch_v2.py \
        --checkpoint "$CKPT" \
        --name "$NAME" \
        --num-episodes $PLAN_EPS \
        --horizons $PLAN_HORIZONS \
        --frameskip $FS \
        --cem-samples $CEM_SAMPLES \
        --cem-iters $CEM_ITERS \
        --output-dir ~/le-wm/overnight_results 2>&1 | grep -E "Summary|h=|Saved"

    # Clean up large checkpoint to save disk (keep probe components)
    rm -f "$CKPT"

    echo "[$(date)] Done: $NAME"
    echo ""
done

# Merge all results and build viewer
echo "================================================================"
echo "Building combined viewer..."
echo "================================================================"

python3 -c "
import json, glob, os
results_dir = os.path.expanduser('~/le-wm/overnight_results')
all_results = []
for f in sorted(glob.glob(os.path.join(results_dir, 'results_*.json'))):
    with open(f) as fh:
        all_results.extend(json.load(fh))
with open(os.path.join(results_dir, 'all_results.json'), 'w') as fh:
    json.dump(all_results, fh)
print(f'Merged {len(all_results)} plans from {len(glob.glob(os.path.join(results_dir, \"results_*.json\")))} configs')

# Summary table
from collections import defaultdict
by_model = defaultdict(list)
for r in all_results:
    by_model[r['model']].append(r)

print()
print('%-30s %8s %8s %8s %8s' % ('Model', 'GT_dist', 'CEM_dist', 'GT_cos', 'CEM_cos'))
print('-' * 75)
for m in sorted(by_model.keys()):
    rs = by_model[m]
    print('%-30s %8.3f %8.3f %8.4f %8.4f' % (
        m,
        sum(r['gt_goal_dist'] for r in rs)/len(rs),
        sum(r['cem_goal_dist'] for r in rs)/len(rs),
        sum(r['gt_cos_sim'] for r in rs)/len(rs),
        sum(r['cem_cos_sim'] for r in rs)/len(rs),
    ))
" 2>&1

echo ""
echo "=== Overnight Sweep Complete: $(date) ==="
