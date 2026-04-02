# VJEPA2 True Motion Embedding Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace our per-frame timm ViT-B/16 embeddings with proper VJEPA2 clip-level embeddings matching the original repo's configuration, then retrain probes and measure the difference.

**Architecture:** VJEPA2 ViT-Large (or ViT-Giant) processes **64-frame clips** through a **3D tubelet embedding** (tubelet_size=2, patch_size=16, crop=256px). This produces one embedding per clip, not per frame. The SDP10 videos at 30fps yield ~7 clips per 451-frame rendering. We encode all 2722 true-motion renderings via SkyPilot A10G spot instances, store clip-level embeddings + temporally-aligned actions to S3, then train probes on the H200.

**Tech Stack:** VJEPA2 via `torch.hub.load('facebookresearch/vjepa2')`, SkyPilot (5× A10G spot), decord (video loading), PyAV (fallback), h5py

---

## Key Differences: Current vs Correct Pipeline

| Aspect | Current (WRONG) | Correct (VJEPA2) |
|--------|-----------------|-------------------|
| **Model** | timm ViT-B/16 (86M, ImageNet) | VJEPA2 ViT-Large (307M, video SSL) |
| **Input** | Single frame (B, C, H, W) | 64-frame clip (B, C, T, H, W) |
| **Resolution** | 224×224 | 256×256 |
| **Patch embed** | 2D Conv (16×16) | 3D Conv (2×16×16 tubelet) |
| **Temporal** | None — frames independent | Tubelet merges 2 frames → 32 temporal tokens |
| **Output** | 768d per frame | 1024d per clip (mean-pooled patches) |
| **Granularity** | 451 embeddings per rendering | ~7 clip embeddings per rendering |
| **Normalization** | Resize 224 + ImageNet norm | Resize+CenterCrop 256 + ImageNet norm (×255 for uint8) |
| **Action alignment** | 1 action per frame | Aggregate actions per 64-frame clip |

---

## File Structure

### New files:
```
sky/vjepa2_worker.py            # SkyPilot worker: VJEPA2 clip encoding
sky/vjepa2_task.yaml            # SkyPilot job config (5× A10G)
scripts/merge_vjepa2_shards.py  # Merge per-rendering NPZs → single HDF5
tests/test_vjepa2_alignment.py  # Verify clip shapes, action alignment, reproducibility
```

### Modified files:
```
train_probe.py                  # Support variable embed_dim (1024 for ViT-L)
```

---

### Task 1: Write VJEPA2 SkyPilot Worker

**Files:**
- Create: `sky/vjepa2_worker.py`

- [ ] **Step 1: Write the worker**

The worker must:
1. Load VJEPA2 ViT-Large via `torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')`
   - Returns `(encoder, predictor)` — use only `encoder`
   - `encoder.embed_dim = 1024`, `tubelet_size=2`, `patch_size=16`, `num_frames=64`
2. For each rendering:
   a. Download video from S3 (compressed MP4, same region)
   b. Decode ALL frames with decord: `VideoReader` → numpy `(T, H, W, 3)` uint8
   c. Split into **non-overlapping 64-frame clips**: `clips[i] = frames[i*64:(i+1)*64]`
      - Pad last clip with repeated last frame if `T % 64 != 0`
      - Discard if `T < 64` (too short)
   d. For each clip:
      - Resize to 256 (short side) + CenterCrop 256×256 (matching eval transform)
      - Normalize: `(pixel / 255.0 - mean) / std` with ImageNet stats
      - Shape: `(1, 3, 64, 256, 256)` — `(B, C, T, H, W)`
   e. Forward through encoder → `(1, N_patches, 1024)` → mean pool → `(1, 1024)`
   f. Compute **clip-level actions**: for each 64-frame clip, aggregate the 6D root delta poses
      - Sum deltas within the clip → net displacement over 64 frames
      - Or keep all 64 per-frame deltas and let the probe handle aggregation
      - **Decision**: Store per-frame actions + clip boundaries so probe can apply any frameskip
3. Upload per-rendering NPZ: `embedding=(n_clips, 1024)`, `action=(T, 6)`, `clip_indices=(n_clips, 2)` [start, end frame for each clip]

Key code pattern (from `parallel-processing/shared/vjepa_optimized.py`):
```python
# Load model
encoder, predictor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large', trust_repo=True)
encoder = encoder.to(device).eval().half()

# Preprocessing (matches eval config)
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Process clip: (64, H, W, 3) uint8 → (1, 3, 64, 256, 256) float
clip_tensors = torch.stack([transform(Image.fromarray(f)) for f in clip_frames])  # (64, 3, 256, 256)
clip_input = clip_tensors.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device).half()   # (1, 3, 64, 256, 256)

with torch.no_grad():
    features = encoder(clip_input)  # (1, N_patches, 1024)
    pooled = features.mean(dim=1)   # (1, 1024)
```

- [ ] **Step 2: Write tests locally before deploying**

```python
# tests/test_vjepa2_alignment.py
# Test 1: Model loads and produces correct output shape
# Test 2: Clip splitting produces correct number of clips
# Test 3: Transform matches eval config (256 crop, ImageNet norm)
# Test 4: Action alignment — clip boundaries match frame indices
# Test 5: Deterministic — same clip → same embedding
# Test 6: Two batch sizes → same result
```

- [ ] **Step 3: Run tests locally (needs GPU + vjepa2 repo)**

```bash
pytest tests/test_vjepa2_alignment.py -v -s --timeout=120
```

- [ ] **Step 4: Commit**

```bash
git add sky/vjepa2_worker.py tests/test_vjepa2_alignment.py
git commit -m "feat: add VJEPA2 clip-level encoding worker with alignment tests"
```

---

### Task 2: Write SkyPilot Job Config

**Files:**
- Create: `sky/vjepa2_task.yaml`

- [ ] **Step 1: Write YAML**

```yaml
name: lewm-vjepa2

resources:
  cloud: aws
  region: eu-west-1
  accelerators: A10G:1
  use_spot: true
  disk_size: 80

num_nodes: 5

file_mounts:
  /task/vjepa2_worker.py: ./sky/vjepa2_worker.py

setup: |
  pip install uv 2>/dev/null || true
  uv pip install --system "torch==2.1.2+cu121" "torchvision==0.16.2+cu121" --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3
  uv pip install --system "numpy<2" timm einops decord av boto3 Pillow h5py 2>&1 | tail -3

run: |
  echo "Node rank: $SKYPILOT_NODE_RANK / $SKYPILOT_NUM_NODES"
  echo "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
  python /task/vjepa2_worker.py
```

**Important**: VJEPA2 needs `einops` and the model downloads from `dl.fbaipublicfiles.com` (~1.2GB for ViT-Large). The first node will be slow (download), subsequent nodes may cache if on same AMI.

- [ ] **Step 2: Test with 1 node first**

```bash
sky jobs launch sky/vjepa2_task.yaml --name vjepa2-test -y
# Wait 5 min, check logs
sky jobs logs <id> --no-follow | tail -20
# Check S3 output
aws s3 ls s3://kinetix-rd-storage/lewm/vjepa2_embeddings/ | head -5
```

- [ ] **Step 3: Verify compute vs IO ratio**

The worker prints `dl=Xs enc=Xs ratio=X.XX (COMPUTE|IO-bound)` every 10 renderings. VJEPA2 ViT-Large on A10G should be more compute-bound than ViT-B/16 since:
- Model is 3.5× larger (307M vs 86M)
- Input is 64 frames × 256×256 (vs 1 frame × 224×224)
- 3D convolutions in patch embed

Expected: ratio > 0.5 (compute-bound) → good GPU utilization

- [ ] **Step 4: Scale to 5 nodes if test passes**

```bash
sky jobs launch sky/vjepa2_task.yaml --name vjepa2-full -y
```

- [ ] **Step 5: Commit**

```bash
git add sky/vjepa2_task.yaml
git commit -m "feat: add VJEPA2 SkyPilot job config"
```

---

### Task 3: Merge Shards + Build HDF5

**Files:**
- Create: `scripts/merge_vjepa2_shards.py`

- [ ] **Step 1: Write merge script**

Downloads all per-rendering NPZ files from S3, concatenates into a single HDF5 with:
- `embedding`: `(total_clips, 1024)` float32
- `action`: `(total_frames, 6)` float32 — raw per-frame actions
- `clip_start`: `(total_clips,)` int32 — frame index where each clip starts
- `clip_end`: `(total_clips,)` int32 — frame index where each clip ends
- `ep_len`: `(n_episodes,)` — clips per episode
- `ep_offset`: `(n_episodes,)` — cumulative clip offset

This differs from the per-frame HDF5: the probe's `EmbeddingDataset` needs to be updated to sample **clip sequences** (not frame sequences).

- [ ] **Step 2: Run on H200**

```bash
ssh -x RD-H100-2 'tmux new-session -d -s merge "cd ~/le-wm && source .venv/bin/activate && python scripts/merge_vjepa2_shards.py"'
```

- [ ] **Step 3: Commit**

```bash
git add scripts/merge_vjepa2_shards.py
git commit -m "feat: add VJEPA2 shard merger for clip-level HDF5"
```

---

### Task 4: Adapt Probe Training for Clip-Level Embeddings

**Files:**
- Modify: `train_probe.py` — add `--clip-level` mode

- [ ] **Step 1: Adapt `EmbeddingDataset`**

In clip-level mode:
- Each "frame" in the sequence is actually a clip embedding (1024d)
- `frameskip` now means "skip N clips" (each clip = 64 raw frames = 2.13s at 30fps)
- Actions between clips: aggregate the 64 per-frame deltas into one clip-level action
  - Sum translations, compose rotations (or just sum axis-angle deltas as approximation)
- `num_steps=4` with `frameskip=1` means 4 consecutive clips ≈ 8.5 seconds

- [ ] **Step 2: Run probe sweep (same grid as before)**

```bash
# 4 proj_dim × 5 sigreg × 2 lr = 40 runs × 10 epochs
bash scripts/sweep_true_motion.sh  # with --clip-level and --embed-dim 1024
```

- [ ] **Step 3: Commit**

```bash
git add train_probe.py
git commit -m "feat: support clip-level embeddings in probe training"
```

---

### Task 5: Alignment Tests

**Files:**
- Create: `tests/test_vjepa2_alignment.py`

- [ ] **Step 1: Write comprehensive tests**

```python
class TestVJEPA2Pipeline:
    def test_model_output_shape(self):
        """VJEPA2 ViT-L: (1, 3, 64, 256, 256) → (1, N_patches, 1024)"""
        
    def test_clip_splitting(self):
        """451 frames / 64 = 7 clips (last padded from 3 frames)"""
        
    def test_eval_transform_matches_repo(self):
        """Resize(256, bicubic) + CenterCrop(256) + ToTensor + ImageNet norm"""
        
    def test_tubelet_temporal_reduction(self):
        """64 frames / tubelet_size=2 = 32 temporal positions"""
        
    def test_clip_action_aggregation(self):
        """Sum of 64 per-frame deltas = clip-level action"""
        
    def test_s3_vs_local_exact_match(self):
        """SkyPilot output matches local single-process computation"""
        
    def test_deterministic_across_batch_sizes(self):
        """batch_size=1 vs batch_size=4 → same results"""
        
    def test_no_information_leakage_between_clips(self):
        """Embedding of clip[i] independent of clip[i+1]"""
```

- [ ] **Step 2: Run on H200**

```bash
pytest tests/test_vjepa2_alignment.py -v -s
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_vjepa2_alignment.py
git commit -m "test: comprehensive VJEPA2 pipeline alignment verification"
```

---

### Task 6: Compare Results

- [ ] **Step 1: Evaluate both embedding types on the same probe architecture**

| Config | timm ViT-B/16 (per-frame) | VJEPA2 ViT-L (per-clip) |
|--------|---------------------------|-------------------------|
| embed_dim | 768 | 1024 |
| temporal | None | 64-frame context |
| n_embeddings per rendering | ~451 | ~7 |
| Training samples | 1.19M | ~19k |

- [ ] **Step 2: Build comparison visualization**

Update `docs/comparison.html` with VJEPA2 results alongside timm ViT-B/16.

---

## Compute Budget

| Phase | Time | Cost |
|-------|------|------|
| VJEPA2 encoding (5× A10G spot) | ~2h (model is 3.5× larger) | ~$6 |
| Merge to HDF5 | ~10 min | Free (H200) |
| Probe sweep (40 configs × 10 epochs) | ~2h | Free (H200) |
| Tests | ~10 min | Free |
| **Total** | **~4.5h** | **~$6** |

## Expected Outcome

VJEPA2 clip embeddings should significantly outperform per-frame ViT-B/16 because:
1. **Temporal context**: 64 frames of motion encoded jointly (vs independent frames)
2. **Video SSL pretraining**: VJEPA2 was trained to predict video — our data IS video
3. **Larger model**: 307M vs 86M params, 1024d vs 768d embeddings
4. **More meaningful representation**: Each embedding captures 2 seconds of motion, not a single static frame
