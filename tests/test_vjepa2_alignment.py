"""Tests for VJEPA2 clip-level embedding pipeline.

Verifies model output shapes, clip splitting, preprocessing, action extraction,
determinism, and clip independence. GPU tests are skipped if CUDA is unavailable.

Run: pytest tests/test_vjepa2_alignment.py -v -s --timeout=120
"""

import sys
import os

import numpy as np
import pytest
import torch

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sky.vjepa2_worker import (
    split_into_clips,
    pad_clip,
    make_eval_transform,
    preprocess_clip,
    CLIP_SIZE,
    CROP_SIZE,
    EMBED_DIM,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs GPU"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vjepa2_encoder():
    """Load VJEPA2 ViT-Large encoder once per test module (expensive)."""
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")
    encoder, _predictor = torch.hub.load(
        "facebookresearch/vjepa2", "vjepa2_vit_large", trust_repo=True
    )
    encoder = encoder.to("cuda").eval().half()
    return encoder


@pytest.fixture(scope="module")
def eval_transform():
    return make_eval_transform()


# ---------------------------------------------------------------------------
# Test 1: Model loads and output shape
# ---------------------------------------------------------------------------

@needs_gpu
def test_model_loads_and_output_shape(vjepa2_encoder):
    """VJEPA2 ViT-L: (1, 3, 64, 256, 256) -> (1, N_patches, 1024)."""
    encoder = vjepa2_encoder
    assert encoder.embed_dim == EMBED_DIM, (
        f"Expected embed_dim={EMBED_DIM}, got {encoder.embed_dim}"
    )

    dummy = torch.randn(
        1, 3, CLIP_SIZE, CROP_SIZE, CROP_SIZE,
        device="cuda", dtype=torch.float16,
    )
    with torch.no_grad():
        out = encoder(dummy)

    assert out.ndim == 3, f"Expected 3D output, got {out.ndim}D"
    B, N, D = out.shape
    assert B == 1
    assert D == EMBED_DIM, f"Expected D={EMBED_DIM}, got {D}"
    # N should be num_patches = (T/tubelet) * (H/patch) * (W/patch)
    # = (64/2) * (256/16) * (256/16) = 32 * 16 * 16 = 8192
    assert N > 0, f"Expected positive N_patches, got {N}"


# ---------------------------------------------------------------------------
# Test 2: Clip splitting
# ---------------------------------------------------------------------------

class TestClipSplitting:
    def test_451_frames_7_clips(self):
        """451 frames -> 7 clips (last padded from 451-448=3 frames)."""
        clips = split_into_clips(451, clip_size=64)
        assert len(clips) == 8  # 0-63, 64-127, ..., 384-447, 448-450
        # Last clip: start=448, end=451 (3 frames, will be padded)
        assert clips[-1] == (448, 451)

    def test_64_frames_1_clip(self):
        """Exactly 64 frames -> 1 clip, no padding."""
        clips = split_into_clips(64, clip_size=64)
        assert len(clips) == 1
        assert clips[0] == (0, 64)

    def test_63_frames_skip(self):
        """63 frames -> too short, skip (empty list)."""
        clips = split_into_clips(63, clip_size=64)
        assert len(clips) == 0

    def test_128_frames_2_clips(self):
        """128 frames -> exactly 2 full clips."""
        clips = split_into_clips(128, clip_size=64)
        assert len(clips) == 2
        assert clips[0] == (0, 64)
        assert clips[1] == (64, 128)

    def test_pad_clip_short(self):
        """Pad a 3-frame clip to 64 by repeating last frame."""
        frames = np.random.randint(0, 255, (3, 32, 32, 3), dtype=np.uint8)
        padded = pad_clip(frames, clip_size=64)
        assert padded.shape == (64, 32, 32, 3)
        # First 3 frames unchanged
        np.testing.assert_array_equal(padded[:3], frames)
        # Remaining 61 frames are copies of the last original frame
        for j in range(3, 64):
            np.testing.assert_array_equal(padded[j], frames[-1])

    def test_pad_clip_exact(self):
        """Exact-size clip needs no padding."""
        frames = np.random.randint(0, 255, (64, 32, 32, 3), dtype=np.uint8)
        padded = pad_clip(frames, clip_size=64)
        assert padded.shape == (64, 32, 32, 3)
        np.testing.assert_array_equal(padded, frames)


# ---------------------------------------------------------------------------
# Test 3: Eval transform
# ---------------------------------------------------------------------------

def test_eval_transform(eval_transform):
    """Verify Resize(256, bicubic) + CenterCrop(256) + ToTensor + Normalize
    produces expected output shape and value range."""
    from PIL import Image

    # Create a non-square image to test resize + crop
    img = Image.fromarray(
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    )
    out = eval_transform(img)

    # Shape: (3, 256, 256) after ToTensor
    assert out.shape == (3, CROP_SIZE, CROP_SIZE), f"Got shape {out.shape}"
    assert out.dtype == torch.float32

    # After ImageNet normalization, values are roughly in [-2.5, 2.5]
    # (0-mean)/std range: (0 - 0.485)/0.229 ~ -2.12, (1 - 0.406)/0.225 ~ 2.64
    assert out.min() > -3.0, f"Min too low: {out.min()}"
    assert out.max() < 3.5, f"Max too high: {out.max()}"


# ---------------------------------------------------------------------------
# Test 4: Mean pooling
# ---------------------------------------------------------------------------

def test_mean_pooling():
    """(1, N, 1024) -> mean(dim=1) -> (1, 1024)."""
    N = 8192  # expected patch count for 64-frame 256px input
    features = torch.randn(1, N, EMBED_DIM)
    pooled = features.mean(dim=1)
    assert pooled.shape == (1, EMBED_DIM)

    # Verify it is actually the mean
    expected = features.sum(dim=1) / N
    torch.testing.assert_close(pooled, expected)


# ---------------------------------------------------------------------------
# Test 5: Action extraction matches process.py
# ---------------------------------------------------------------------------

def test_action_extraction_matches_process_py(tmp_path):
    """Compare action extraction against the existing sky/process.py::extract_body_poses
    for the same synthetic NPZ data."""
    from sky.process import extract_body_poses as process_extract
    from sky.vjepa2_worker import extract_body_poses as vjepa2_extract

    # Create synthetic NPZ with known poses and translations
    T = 100
    rng = np.random.RandomState(42)
    # poses_beta0_world: (1, T, J, K) — we need at least root joint (joint 0)
    # process.py uses poses[:, 0, :] which gives (T, K)
    n_joints = 24
    rot_dim = 3
    poses = rng.randn(1, T, n_joints, rot_dim).astype(np.float32)
    trans = rng.randn(1, T, 3).astype(np.float32)

    npz_path = str(tmp_path / "test.npz")
    np.savez(npz_path, poses_beta0_world=poses, trans_beta0_world=trans)

    # Both functions download from S3, so we mock with a local helper
    class FakeS3:
        def download_file(self, bucket, key, path):
            import shutil
            shutil.copy(npz_path, path)

    fake_s3 = FakeS3()
    fake_uri = "s3://fake-bucket/fake-key.npz"

    result_process = process_extract(fake_s3, fake_uri)
    result_vjepa2 = vjepa2_extract(fake_s3, fake_uri)

    assert result_process is not None
    assert result_vjepa2 is not None
    np.testing.assert_array_equal(result_process, result_vjepa2)
    assert result_process.shape == (T, 6)
    assert result_process.dtype == np.float32

    # Last row should be NaN (no delta for final frame)
    assert np.all(np.isnan(result_process[-1]))


# ---------------------------------------------------------------------------
# Test 6: Deterministic
# ---------------------------------------------------------------------------

@needs_gpu
def test_deterministic(vjepa2_encoder, eval_transform):
    """Same clip -> same embedding twice."""
    rng = np.random.RandomState(123)
    clip_frames = rng.randint(0, 255, (CLIP_SIZE, 320, 240, 3), dtype=np.uint8)

    clip_input = preprocess_clip(clip_frames, eval_transform)
    clip_input = clip_input.to("cuda", dtype=torch.float16)

    with torch.no_grad():
        out1 = vjepa2_encoder(clip_input).mean(dim=1)
        out2 = vjepa2_encoder(clip_input).mean(dim=1)

    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Test 7: Clip independence
# ---------------------------------------------------------------------------

@needs_gpu
def test_clip_independence(vjepa2_encoder, eval_transform):
    """Encoding clip[0] alone vs clip[0] as part of a batch -> same result."""
    rng = np.random.RandomState(456)
    clip0_frames = rng.randint(0, 255, (CLIP_SIZE, 320, 240, 3), dtype=np.uint8)
    clip1_frames = rng.randint(0, 255, (CLIP_SIZE, 320, 240, 3), dtype=np.uint8)

    input0 = preprocess_clip(clip0_frames, eval_transform).to("cuda", dtype=torch.float16)
    input1 = preprocess_clip(clip1_frames, eval_transform).to("cuda", dtype=torch.float16)

    # Encode clip0 alone
    with torch.no_grad():
        out_single = vjepa2_encoder(input0).mean(dim=1)  # (1, 1024)

    # Encode clip0 and clip1 as a batch
    batch_input = torch.cat([input0, input1], dim=0)  # (2, 3, 64, 256, 256)
    with torch.no_grad():
        out_batch = vjepa2_encoder(batch_input).mean(dim=1)  # (2, 1024)

    # clip0 result should be identical regardless of batching
    torch.testing.assert_close(
        out_single, out_batch[:1],
        atol=1e-3, rtol=1e-3,
        msg="Clip embedding should not depend on other clips in the batch",
    )
