#!/bin/bash
# Bayesian-ish sweep over SIGReg weight on the H200.
# Runs trials sequentially (single GPU), logs each to ClearML.
#
# Usage: setsid bash scripts/sweep_sigreg.sh > ~/le-wm/sweep.log 2>&1 &

set -e
cd ~/le-wm
source .venv/bin/activate
export STABLEWM_HOME=$HOME/.stable_worldmodel

# SIGReg weights to try — coarse-to-fine grid
WEIGHTS=(0.001 0.005 0.01 0.03 0.05 0.09 0.15 0.25 0.4)
EPOCHS=30
BS=128

echo "=== SIGReg Sweep: ${#WEIGHTS[@]} trials, $EPOCHS epochs each ==="
echo "Weights: ${WEIGHTS[*]}"
echo ""

for w in "${WEIGHTS[@]}"; do
    TAG="sigreg_${w}"
    echo "[$(date)] Starting trial: sigreg_weight=$w"

    python train.py \
        data=prod_beta0_fs5 \
        +wm.action_dim=6 \
        trainer.max_epochs=$EPOCHS \
        wandb.enabled=false \
        clearml.enabled=true \
        clearml.project=lewm/hpo \
        clearml.task_name="sweep-sigreg-${w}" \
        '+clearml.tags=[hpo,sigreg-sweep]' \
        loader.batch_size=$BS \
        loader.num_workers=8 \
        seed=42 \
        loss.sigreg.weight=$w \
        output_model_name="lewm_sweep_sigreg_${w}" \
        2>&1 | tail -5

    echo "[$(date)] Finished trial: sigreg_weight=$w"
    echo ""
done

echo "=== Sweep complete ==="
