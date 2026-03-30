#!/bin/bash
# Compare frozen vs unfrozen ViT-B/16 pretrained encoder with original sigreg=0.09
#
# Usage: setsid bash scripts/run_vjepa_comparison.sh > ~/le-wm/vjepa_comparison.log 2>&1 < /dev/null &

cd ~/le-wm
source .venv/bin/activate
export STABLEWM_HOME=$HOME/.stable_worldmodel
export PATH="$HOME/.local/bin:$PATH"

CKPT_PATH="$HOME/.stable_worldmodel/vjepa_checkpoints/vitb16_pretrained.pt"
EPOCHS=30

echo "=== ViT-B/16 Pretrained: Frozen vs Unfrozen ==="
echo "Checkpoint: $CKPT_PATH"
echo "SIGReg weight: 0.09 (original)"
echo "Epochs: $EPOCHS"
echo ""

# --- Trial 1: Frozen encoder ---
echo "[$(date)] Starting: ViT-B/16 FROZEN"

python train.py \
    data=prod_beta0_fs5 \
    +wm.action_dim=6 \
    encoder_type=vjepa \
    vjepa_checkpoint="$CKPT_PATH" \
    encoder_frozen=true \
    patch_size=16 \
    trainer.max_epochs=$EPOCHS \
    wandb.enabled=false \
    clearml.enabled=true \
    clearml.project=lewm \
    clearml.task_name="vitb16-frozen-sigreg0.09" \
    '+clearml.tags=[vitb16,frozen,comparison]' \
    loader.batch_size=64 \
    loader.num_workers=8 \
    seed=42 \
    loss.sigreg.weight=0.09 \
    output_model_name=lewm_vitb16_frozen \
    2>&1 | tail -5 || echo "[$(date)] Frozen trial exited with code $?"

echo "[$(date)] Finished: ViT-B/16 FROZEN"
echo ""

# --- Trial 2: Unfrozen encoder (end-to-end fine-tuning) ---
echo "[$(date)] Starting: ViT-B/16 UNFROZEN (end-to-end)"

python train.py \
    data=prod_beta0_fs5 \
    +wm.action_dim=6 \
    encoder_type=vjepa \
    vjepa_checkpoint="$CKPT_PATH" \
    encoder_frozen=false \
    patch_size=16 \
    trainer.max_epochs=$EPOCHS \
    wandb.enabled=false \
    clearml.enabled=true \
    clearml.project=lewm \
    clearml.task_name="vitb16-unfrozen-sigreg0.09" \
    '+clearml.tags=[vitb16,unfrozen,comparison]' \
    loader.batch_size=32 \
    loader.num_workers=8 \
    seed=42 \
    loss.sigreg.weight=0.09 \
    optimizer.lr=1e-5 \
    output_model_name=lewm_vitb16_unfrozen \
    2>&1 | tail -5 || echo "[$(date)] Unfrozen trial exited with code $?"

echo "[$(date)] Finished: ViT-B/16 UNFROZEN"
echo ""
echo "=== Comparison complete ==="
