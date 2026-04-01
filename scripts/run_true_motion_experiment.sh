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
python3 -c "
import torch, os, json, numpy as np, h5py
from train_probe import EmbeddingDataset
from module import ARPredictor, Embedder, MLP, SIGReg

cache = os.environ['STABLEWM_HOME']
device = 'cuda'

# Val set
val_ds = EmbeddingDataset(
    os.path.join(cache, 'true_motion_vitb16emb.h5'),
    num_steps=4, frameskip=5, split='val'
)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4)

curve = []
for epoch in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    path = os.path.join(cache, f'true-motion-probe-sr05_epoch{epoch}_probe_components.pt')
    if not os.path.exists(path):
        path = os.path.join(cache, 'true-motion-probe-sr05_probe_components.pt')
        if epoch != 100: continue

    probe = torch.load(path, map_location='cpu')
    cfg = probe['config']
    ead = cfg['frameskip'] * cfg['action_dim']

    proj = MLP(input_dim=768, output_dim=192, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d).to(device)
    pred = ARPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0).to(device)
    pp = MLP(input_dim=192, output_dim=192, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d).to(device)
    ae = Embedder(input_dim=ead, smoothed_dim=max(ead, 10), emb_dim=192).to(device)
    proj.load_state_dict(probe['projector']); pred.load_state_dict(probe['predictor'])
    pp.load_state_dict(probe['pred_proj']); ae.load_state_dict(probe['action_encoder'])
    for m in [proj, pred, pp, ae]: m.eval()
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)

    tp, tc, ts, n = 0, 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            e = batch['embedding'].to(device); a = torch.nan_to_num(batch['action'].to(device), 0)
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
    curve.append({'epoch': epoch, 'pred_loss': tp/n, 'copy_ratio': ratio, 'sigreg': ts/n})
    print(f'  Epoch {epoch}: ratio={ratio:.4f}, pred={tp/n:.6f}, sreg={ts/n:.3f}')

with open(os.path.join(cache, 'true_motion_training_curve.json'), 'w') as f:
    json.dump(curve, f)
print('Saved training curve')
"

echo ""
echo "=== Experiment Complete: $(date) ==="
