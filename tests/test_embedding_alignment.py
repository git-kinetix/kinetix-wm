"""Verify that SkyPilot-produced embeddings match local single-process computation.

Downloads a few sample renderings, runs the exact same pipeline locally
(same model, same preprocessing, same fp16 path), and compares against
the NPZ files produced by the distributed workers on S3.

Run: pytest tests/test_embedding_alignment.py -v -s
"""

import io
import os
import tempfile

import boto3
import numpy as np
import pytest
import torch

BUCKET = "kinetix-rd-storage"
MANIFEST_KEY = "lewm/true_motion_manifest.json"
CHECKPOINT_KEY = "lewm/vitb16_pretrained.pt"
OUTPUT_PREFIX = "lewm/embeddings_sky/"
REGION = "eu-west-1"

# Tolerance: fp16 has ~1e-3 relative error, but we accumulate through layers
# Use generous tolerance since we're comparing fp16-computed values
EMB_ATOL = 1e-2  # absolute tolerance for embedding comparison
EMB_RTOL = 5e-3  # relative tolerance
ACTION_ATOL = 1e-6  # actions are computed in float64, should be exact


@pytest.fixture(scope="module")
def s3():
    return boto3.client("s3", region_name=REGION)


@pytest.fixture(scope="module")
def manifest(s3):
    resp = s3.get_object(Bucket=BUCKET, Key=MANIFEST_KEY)
    import json
    return json.loads(resp["Body"].read())


@pytest.fixture(scope="module")
def encoder(s3):
    """Load ViT-B/16 exactly as the SkyPilot worker does."""
    import timm

    ckpt_path = "/tmp/test_vitb16.pt"
    if not os.path.exists(ckpt_path):
        s3.download_file(BUCKET, CHECKPOINT_KEY, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    model.load_state_dict(state_dict, strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.to(device).eval().half()
    else:
        model = model.to(device).eval()

    return model, device


def decode_video_local(s3, video_key, img_size=224):
    """Exact same decode as sky/process.py::download_and_decode_video."""
    import av
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        s3.download_file(BUCKET, video_key, tmp.name)
        container = av.open(tmp.name)
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_image().convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            frames.append(np.asarray(img))
        container.close()
    return np.stack(frames) if frames else None


def extract_actions_local(s3, features_uri):
    """Exact same extraction as sky/process.py::extract_body_poses."""
    parts = features_uri.replace("s3://", "").split("/", 1)
    npz_bucket, npz_key = parts[0], parts[1]

    with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
        s3.download_file(npz_bucket, npz_key, tmp.name)
        npz = np.load(tmp.name, allow_pickle=True)

    poses = npz.get("poses_beta0_world")
    trans = npz.get("trans_beta0_world")
    if poses is None or trans is None:
        return None
    if poses.ndim == 4:
        poses = poses[0]
    if trans.ndim == 3:
        trans = trans[0]

    root_rot = poses[:, 0, :]
    d_trans = np.diff(trans, axis=0)
    d_rot = np.diff(root_rot, axis=0)
    deltas = np.concatenate([d_trans, d_rot], axis=1).astype(np.float32)
    last = np.full((1, 6), np.nan, dtype=np.float32)
    return np.concatenate([deltas, last], axis=0)


def encode_frames_local(model, device, frames_np, batch_size=128):
    """Exact same encoding as sky/process.py — fp16 on GPU, float32 output."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    if device == "cuda":
        mean = mean.half()
        std = std.half()

    all_embs = []
    with torch.no_grad():
        if device == "cuda":
            ctx = torch.amp.autocast("cuda", dtype=torch.float16)
        else:
            ctx = torch.no_grad()  # no autocast on CPU

        with ctx:
            for j in range(0, len(frames_np), batch_size):
                batch = frames_np[j:j + batch_size]
                if device == "cuda":
                    t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device, dtype=torch.float16) / 255.0
                else:
                    t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
                t = (t - mean) / std
                emb = model(t)
                all_embs.append(emb.float().cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def download_s3_npz(s3, rendering_id):
    """Download the SkyPilot-produced NPZ from S3."""
    key = f"{OUTPUT_PREFIX}{rendering_id}.npz"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        data = np.load(io.BytesIO(resp["Body"].read()))
        return data["embedding"], data["action"]
    except Exception as e:
        pytest.skip(f"S3 NPZ not found for {rendering_id}: {e}")


class TestEmbeddingAlignment:
    """Compare locally-computed embeddings against SkyPilot S3 output."""

    @pytest.fixture(autouse=True)
    def _setup(self, s3, manifest, encoder):
        self.s3 = s3
        self.items = manifest["items"]
        self.model, self.device = encoder

    def _test_rendering(self, item):
        """Full pipeline test for a single rendering."""
        rid = item["rendering_id"]

        # 1. Download S3 reference
        s3_emb, s3_act = download_s3_npz(self.s3, rid)

        # 2. Reproduce locally
        frames = decode_video_local(self.s3, item["video_key"])
        assert frames is not None, f"Video decode failed for {rid}"

        actions = extract_actions_local(self.s3, item["features_uri"])
        assert actions is not None, f"Action extraction failed for {rid}"

        # 3. Align (same truncation as worker)
        n = min(len(frames), len(actions))
        frames = frames[:n]
        actions = actions[:n]

        # 4. Encode
        local_emb = encode_frames_local(self.model, self.device, frames)

        # 5. Compare shapes
        assert local_emb.shape == s3_emb.shape, (
            f"Shape mismatch: local {local_emb.shape} vs S3 {s3_emb.shape}"
        )
        assert actions.shape == s3_act.shape, (
            f"Action shape mismatch: local {actions.shape} vs S3 {s3_act.shape}"
        )

        # 6. Compare embeddings
        # fp16 path introduces rounding — use tolerant comparison
        close_mask = np.isclose(local_emb, s3_emb, atol=EMB_ATOL, rtol=EMB_RTOL)
        pct_close = close_mask.mean()

        max_diff = np.abs(local_emb - s3_emb).max()
        mean_diff = np.abs(local_emb - s3_emb).mean()

        print(f"\n  {rid[:50]}")
        print(f"  Frames: {n}, Emb shape: {local_emb.shape}")
        print(f"  Embedding: {pct_close:.4%} within tol, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        assert pct_close > 0.99, (
            f"Only {pct_close:.2%} of embedding values match. "
            f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
        )

        # 7. Compare actions (should be exact — same float64 computation)
        # NaN at last position — compare non-NaN values
        valid = ~np.isnan(actions) & ~np.isnan(s3_act)
        if valid.any():
            np.testing.assert_allclose(
                actions[valid], s3_act[valid], atol=ACTION_ATOL,
                err_msg=f"Action mismatch for {rid}"
            )

        # Check NaN alignment
        assert np.array_equal(np.isnan(actions), np.isnan(s3_act)), (
            f"NaN positions differ for {rid}"
        )

        return {"rid": rid, "n_frames": n, "pct_close": pct_close,
                "max_diff": max_diff, "mean_diff": mean_diff}

    def test_first_rendering(self):
        """Test the first rendering in the manifest."""
        self._test_rendering(self.items[0])

    def test_middle_rendering(self):
        """Test a rendering from the middle of the manifest."""
        self._test_rendering(self.items[len(self.items) // 2])

    def test_last_rendering(self):
        """Test the last rendering in the manifest."""
        self._test_rendering(self.items[-1])

    def test_batch_size_invariance(self):
        """Verify that different batch sizes produce identical results."""
        item = self.items[0]
        frames = decode_video_local(self.s3, item["video_key"])
        assert frames is not None

        emb_bs32 = encode_frames_local(self.model, self.device, frames[:64], batch_size=32)
        emb_bs64 = encode_frames_local(self.model, self.device, frames[:64], batch_size=64)

        np.testing.assert_allclose(
            emb_bs32, emb_bs64, atol=1e-5,
            err_msg="Batch size should not affect results"
        )

    def test_deterministic_encoding(self):
        """Verify encoding the same frames twice gives identical results."""
        item = self.items[0]
        frames = decode_video_local(self.s3, item["video_key"])
        assert frames is not None

        emb1 = encode_frames_local(self.model, self.device, frames[:32])
        emb2 = encode_frames_local(self.model, self.device, frames[:32])

        np.testing.assert_array_equal(
            emb1, emb2,
            err_msg="Same input should produce identical output"
        )

    def test_action_computation_deterministic(self):
        """Verify action extraction is deterministic."""
        item = self.items[0]
        act1 = extract_actions_local(self.s3, item["features_uri"])
        act2 = extract_actions_local(self.s3, item["features_uri"])

        assert act1 is not None and act2 is not None
        # NaN-aware comparison
        valid = ~np.isnan(act1) & ~np.isnan(act2)
        np.testing.assert_array_equal(act1[valid], act2[valid])
        np.testing.assert_array_equal(np.isnan(act1), np.isnan(act2))

    def test_frame_count_alignment(self):
        """Verify video frames and NPZ poses have compatible lengths."""
        for item in self.items[:5]:
            frames = decode_video_local(self.s3, item["video_key"])
            actions = extract_actions_local(self.s3, item["features_uri"])
            if frames is None or actions is None:
                continue
            # Frames and actions should be close in length
            # (actions derived from NPZ which may have slightly different frame count)
            diff = abs(len(frames) - len(actions))
            assert diff <= 5, (
                f"{item['rendering_id']}: frame/action length mismatch "
                f"({len(frames)} vs {len(actions)}, diff={diff})"
            )

    def test_embedding_statistics(self):
        """Check that embeddings have reasonable statistics (not collapsed/exploded)."""
        item = self.items[0]
        s3_emb, _ = download_s3_npz(self.s3, item["rendering_id"])

        mean_norm = np.linalg.norm(s3_emb, axis=1).mean()
        var_per_dim = s3_emb.var(axis=0).mean()
        effective_dim = s3_emb.shape[1]

        print(f"\n  Embedding stats: mean_norm={mean_norm:.2f}, var={var_per_dim:.4f}")

        assert mean_norm > 1.0, f"Embeddings too small: mean_norm={mean_norm}"
        assert mean_norm < 100.0, f"Embeddings too large: mean_norm={mean_norm}"
        assert var_per_dim > 0.01, f"Embeddings collapsed: var={var_per_dim}"

    def test_nan_only_at_boundaries(self):
        """Verify NaN only appears at the last action of each rendering."""
        for item in self.items[:5]:
            _, s3_act = download_s3_npz(self.s3, item["rendering_id"])
            nan_rows = np.isnan(s3_act).any(axis=1)

            # Only the last row should be NaN
            assert nan_rows[-1], f"Last action should be NaN for {item['rendering_id']}"
            assert not nan_rows[:-1].any(), (
                f"NaN found before last action for {item['rendering_id']}: "
                f"NaN at positions {np.where(nan_rows)[0].tolist()}"
            )
