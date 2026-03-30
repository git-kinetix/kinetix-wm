#!/bin/bash
# VJEPA sweep: get pretrained ViT-B/16, then sweep sigreg weights with frozen encoder.
#
# Usage: setsid bash scripts/sweep_vjepa.sh > ~/le-wm/vjepa_sweep.log 2>&1 &

cd ~/le-wm
source .venv/bin/activate
export STABLEWM_HOME=$HOME/.stable_worldmodel
export PATH="$HOME/.local/bin:$PATH"

CKPT_DIR="$HOME/.stable_worldmodel/vjepa_checkpoints"
CKPT_PATH="$CKPT_DIR/vitb16_pretrained.pt"

# --- Step 1: Get ViT-B/16 pretrained checkpoint via timm ---
uv pip install timm 2>&1 | tail -2

if [ ! -f "$CKPT_PATH" ]; then
    echo "[$(date)] Creating ViT-B/16 pretrained checkpoint via timm..."
    mkdir -p "$CKPT_DIR"
    python -c "
import timm, torch
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
torch.save({'model': model.state_dict()}, '$CKPT_PATH')
print('Saved timm ViT-B/16 pretrained (ImageNet-21k)')
print('embed_dim=%d, params=%d' % (model.embed_dim, sum(p.numel() for p in model.parameters())))
" 2>&1
    echo "[$(date)] Checkpoint ready: $(ls -lh $CKPT_PATH)"
else
    echo "[$(date)] Checkpoint already exists: $CKPT_PATH"
fi

# Verify
python -c "
import torch
ckpt = torch.load('$CKPT_PATH', map_location='cpu', weights_only=False)
sd = ckpt.get('model', ckpt)
print('Keys: %d, first few: %s' % (len(sd), list(sd.keys())[:3]))
for k in sd:
    if 'patch_embed' in k and 'weight' in k:
        print('embed_dim=%d' % sd[k].shape[0])
        break
" 2>&1

# --- Step 2: Sweep sigreg weights with frozen ViT-B/16 ---
WEIGHTS=(0.001 0.01 0.05 0.09 0.15 0.3)
EPOCHS=30
BS=64

echo ""
echo "=== VJEPA Frozen Sweep: ${#WEIGHTS[@]} trials, $EPOCHS epochs each ==="
echo "Checkpoint: $CKPT_PATH"
echo "Weights: ${WEIGHTS[*]}"
echo ""

for w in "${WEIGHTS[@]}"; do
    echo "[$(date)] Starting VJEPA trial: sigreg_weight=$w"

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
        clearml.project=lewm/hpo \
        clearml.task_name="sweep-vjepa-frozen-sigreg-${w}" \
        '+clearml.tags=[hpo,vjepa,frozen,sigreg-sweep]' \
        loader.batch_size=$BS \
        loader.num_workers=8 \
        seed=42 \
        loss.sigreg.weight=$w \
        output_model_name="lewm_vjepa_frozen_sigreg_${w}" \
        2>&1 | tail -5 || echo "[$(date)] Trial sigreg=$w exited with code $?"

    echo "[$(date)] Finished VJEPA trial: sigreg_weight=$w"
    echo ""
done

echo "=== VJEPA Sweep complete ==="
