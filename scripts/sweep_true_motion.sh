#!/bin/bash
# Complete architecture + sigreg sweep on true motion dataset (10 epochs each)
# Uploads all checkpoints + training logs to S3

set -uo pipefail
cd ~/le-wm
source .venv/bin/activate
export STABLEWM_HOME=$HOME/.stable_worldmodel
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGINS_PATH)")
export PYTHONUNBUFFERED=1

DATASET=true_motion_vitb16emb
EMBED_DIM=768
EPOCHS=10
BS=256
S3_PREFIX="s3://kinetix-rd-storage/lewm/sweep_results"

LOG=~/le-wm/sweep_true_motion_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "  True Motion Sweep: Architecture × SIGReg"
echo "  Dataset: $DATASET (2722 episodes, 1.37M frames)"
echo "  Epochs per run: $EPOCHS"
echo "  Started: $(date)"
echo "============================================================"

RUN=0

# 4 proj × 5 sigreg × 2 lr = 40 runs × ~10 min = ~7h
for PROJ_DIM in 128 192 384 768; do
  for SIGREG in 0.01 0.1 0.5 1.0 2.0; do
    for LR in 5e-4 1e-3; do
      RUN=$((RUN + 1))
      NAME="tm-p${PROJ_DIM}-sr${SIGREG}-lr${LR}"

      echo ""
      echo ">>> [$RUN] $NAME (proj=$PROJ_DIM sigreg=$SIGREG lr=$LR)"
      echo "    Started: $(date)"

      # Train
      python -u train_probe.py \
        --dataset $DATASET \
        --embed-dim $EMBED_DIM \
        --proj-dim $PROJ_DIM \
        --epochs $EPOCHS \
        --batch-size $BS \
        --lr $LR \
        --sigreg-weight $SIGREG \
        --frameskip 5 \
        --action-dim 6 \
        --save-every 5 \
        --task-name "$NAME" \
        2>&1 || echo "    FAILED: $NAME"

      echo "    Finished: $(date)"

      # Upload checkpoint to S3
      for f in $STABLEWM_HOME/${NAME}*probe_components.pt; do
        if [ -f "$f" ]; then
          aws s3 cp "$f" "$S3_PREFIX/checkpoints/$(basename $f)" --quiet && echo "    Uploaded: $(basename $f)"
        fi
      done
    done
  done
done

echo ""
echo "============================================================"
echo "  Sweep complete: $RUN runs"
echo "  Finished: $(date)"
echo "============================================================"

# Upload the full log to S3
aws s3 cp "$LOG" "$S3_PREFIX/logs/$(basename $LOG)" --quiet
echo "Log uploaded to S3"

# Also generate a summary CSV
python3 -u -c "
import os, json, glob

cache = os.environ['STABLEWM_HOME']
results = []

for f in sorted(glob.glob(os.path.join(cache, 'tm-*_probe_components.pt'))):
    import torch
    d = torch.load(f, map_location='cpu', weights_only=False)
    cfg = d.get('config', {})
    name = os.path.basename(f).replace('_probe_components.pt', '')
    results.append({
        'name': name,
        'proj_dim': cfg.get('proj_dim', '?'),
        'sigreg': cfg.get('sigreg_weight', '?'),
        'lr': cfg.get('lr', '?'),
        'epochs': cfg.get('epochs', '?'),
    })

# Print summary table
print()
print('NAME'.ljust(40), 'PROJ', 'SIGREG', 'LR')
print('-' * 70)
for r in results:
    print(r['name'].ljust(40), str(r['proj_dim']).ljust(5), str(r['sigreg']).ljust(7), str(r['lr']))
print(f'Total: {len(results)} checkpoints')
"

echo "Done."
