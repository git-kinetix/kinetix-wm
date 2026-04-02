"""SkyPilot VJEPA2 ViT-Large clip-level embedding + body pose extraction.

Based on sky/process.py pattern, adapted for VJEPA2 video model.
Each node processes renderings[rank::num_nodes], uploads per-rendering NPZ to S3.

Key differences from process.py (ViT-B/16):
- Model: VJEPA2 ViT-Large (307M params, 1024d) via torch.hub
- Input: 64-frame clips (B, 3, 64, 256, 256) not single frames (B, 3, 224, 224)
- Resolution: 256px with Resize+CenterCrop (matching eval config)
- Output: clip-level embeddings (n_clips, 1024) not per-frame (T, 768)
- Video loading: decord (faster bulk frame access) instead of PyAV
- Preprocessing: per-frame Resize(256,BICUBIC)+CenterCrop(256)+ToTensor+Normalize
"""

import io
import json
import os
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor

import boto3
import botocore.config
import numpy as np

BUCKET = "kinetix-rd-storage"
MANIFEST_KEY = "lewm/true_motion_manifest.json"
OUTPUT_PREFIX = "lewm/vjepa2_embeddings/"
CLIP_SIZE = 64
EMBED_DIM = 1024
CROP_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_s3_client():
    cfg = botocore.config.Config(
        max_pool_connections=64,
        connect_timeout=5,
        read_timeout=10,
        retries={"max_attempts": 3, "mode": "adaptive"},
        tcp_keepalive=True,
    )
    return boto3.client("s3", region_name="eu-west-1", config=cfg)


def download_and_decode_video(s3, video_key):
    """Download video from S3 and decode all frames with decord.

    Returns numpy array (T, H, W, 3) uint8, or None on failure.
    """
    from decord import VideoReader, cpu

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        s3.download_file(BUCKET, video_key, tmp.name)
        vr = VideoReader(tmp.name, ctx=cpu(0))
        if len(vr) == 0:
            return None
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()
    return frames


def split_into_clips(n_frames, clip_size=CLIP_SIZE):
    """Split T frames into non-overlapping clips of clip_size.

    Returns list of (start, end) tuples. Last clip is padded by repeating
    the final frame if needed. Returns empty list if n_frames < clip_size.
    """
    if n_frames < clip_size:
        return []
    clips = []
    for start in range(0, n_frames, clip_size):
        end = min(start + clip_size, n_frames)
        clips.append((start, end))
    return clips


def pad_clip(frames, clip_size=CLIP_SIZE):
    """Pad a clip shorter than clip_size by repeating the last frame."""
    if len(frames) >= clip_size:
        return frames[:clip_size]
    pad_count = clip_size - len(frames)
    padding = np.repeat(frames[-1:], pad_count, axis=0)
    return np.concatenate([frames, padding], axis=0)


def make_eval_transform():
    """Build the VJEPA2 eval transform: Resize(256, BICUBIC) + CenterCrop(256) + ToTensor + Normalize."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(CROP_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_clip(clip_frames, transform):
    """Preprocess a clip of numpy frames into model input tensor.

    Args:
        clip_frames: numpy (64, H, W, 3) uint8
        transform: torchvision transform (Resize+CenterCrop+ToTensor+Normalize)

    Returns:
        tensor: (1, 3, 64, 256, 256) float16 on CPU
    """
    import torch
    from PIL import Image

    frame_tensors = []
    for frame in clip_frames:
        img = Image.fromarray(frame)
        frame_tensors.append(transform(img))
    # Stack → (64, 3, 256, 256), then rearrange to (1, 3, 64, 256, 256)
    clip_tensor = torch.stack(frame_tensors)  # (T, C, H, W)
    clip_input = clip_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
    return clip_input


def extract_body_poses(s3, features_uri):
    """Download NPZ and extract 6D root delta poses.

    Identical to sky/process.py::extract_body_poses.
    Returns (T, 6) float32 array of per-frame deltas, or None.
    """
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


def main():
    import torch

    t_start = time.time()

    rank = int(os.environ.get("SKYPILOT_NODE_RANK", 0))
    num_nodes = int(os.environ.get("SKYPILOT_NUM_NODES", 1))
    print(f"[Node {rank}/{num_nodes}] Starting VJEPA2 worker", flush=True)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    s3 = make_s3_client()

    # Download manifest
    print(f"[Node {rank}] Downloading manifest...", flush=True)
    resp = s3.get_object(Bucket=BUCKET, Key=MANIFEST_KEY)
    manifest = json.loads(resp["Body"].read())
    all_items = manifest["items"]

    my_items = all_items[rank::num_nodes]
    print(f"[Node {rank}] Processing {len(my_items)}/{len(all_items)} renderings", flush=True)

    # Load VJEPA2 ViT-Large
    print(f"[Node {rank}] Loading VJEPA2 ViT-Large...", flush=True)
    t_model = time.time()

    # Fix: the vjepa2 main branch has localhost:8300 hardcoded for testing.
    # Download the repo source first, patch the URL, then load the model.
    import torch.hub as _hub
    _repo_dir = _hub._get_cache_or_reload(
        "facebookresearch/vjepa2", force_reload=False, trust_repo=True, calling_fn="load"
    )
    _backbones_path = os.path.join(_repo_dir, "src", "hub", "backbones.py")
    if os.path.exists(_backbones_path):
        with open(_backbones_path) as f:
            src = f.read()
        if "localhost:8300" in src:
            src = src.replace(
                'VJEPA_BASE_URL = "http://localhost:8300"',
                'VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"',
            )
            with open(_backbones_path, "w") as f:
                f.write(src)
            print(f"[Node {rank}] Patched VJEPA_BASE_URL", flush=True)

    encoder, predictor = torch.hub.load(
        "facebookresearch/vjepa2", "vjepa2_vit_large", trust_repo=True
    )
    encoder = encoder.to("cuda").eval().half()
    del predictor  # not needed for embedding extraction

    assert encoder.embed_dim == EMBED_DIM, (
        f"Expected embed_dim={EMBED_DIM}, got {encoder.embed_dim}"
    )

    # Build eval transform
    transform = make_eval_transform()

    # Warmup
    with torch.no_grad():
        dummy = torch.randn(1, 3, CLIP_SIZE, CROP_SIZE, CROP_SIZE,
                            device="cuda", dtype=torch.float16)
        _ = encoder(dummy)
    print(f"[Node {rank}] Model ready in {time.time()-t_model:.1f}s", flush=True)

    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(
        f"[Node {rank}] GPU: {torch.cuda.get_device_name(0)}, "
        f"{gpu_mem:.0f}GB",
        flush=True,
    )

    # Process renderings
    total_clips = 0
    total_dl = 0.0
    total_enc = 0.0
    n_success = 0
    errors = []

    for i, item in enumerate(my_items):
        rid = item["rendering_id"]
        try:
            # Download video + NPZ concurrently
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=2) as pool:
                vid_future = pool.submit(download_and_decode_video, s3, item["video_key"])
                npz_future = pool.submit(extract_body_poses, s3, item["features_uri"])
                frames = vid_future.result()
                actions = npz_future.result()

            if frames is None or actions is None:
                continue

            n = min(len(frames), len(actions))
            frames = frames[:n]
            actions = actions[:n]
            t_dl = time.time() - t0

            # Split into clips, skip if too short
            clip_ranges = split_into_clips(n, CLIP_SIZE)
            if not clip_ranges:
                continue

            # Encode clips
            t1 = time.time()
            all_embs = []
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                for start, end in clip_ranges:
                    clip_frames = frames[start:end]
                    clip_frames = pad_clip(clip_frames, CLIP_SIZE)
                    clip_input = preprocess_clip(clip_frames, transform)
                    clip_input = clip_input.to("cuda", dtype=torch.float16)

                    features = encoder(clip_input)  # (1, N_patches, 1024)
                    pooled = features.mean(dim=1)    # (1, 1024)
                    all_embs.append(pooled.float().cpu().numpy())

            clip_embeddings = np.concatenate(all_embs, axis=0)  # (n_clips, 1024)
            t_enc = time.time() - t1

            # Upload per-rendering NPZ to S3
            buf = io.BytesIO()
            np.savez_compressed(
                buf,
                clip_embeddings=clip_embeddings.astype(np.float32),
                action=actions.astype(np.float32),
                n_clips=len(clip_ranges),
                clip_size=CLIP_SIZE,
            )
            buf.seek(0)
            s3.put_object(
                Bucket=BUCKET,
                Key=f"{OUTPUT_PREFIX}{rid}.npz",
                Body=buf.getvalue(),
            )

            total_clips += len(clip_ranges)
            total_dl += t_dl
            total_enc += t_enc
            n_success += 1

            if (i + 1) % 5 == 0 or (i + 1) == len(my_items):
                elapsed = time.time() - t_start
                cps = total_clips / elapsed if elapsed > 0 else 0
                ratio = total_enc / (total_dl + 1e-8)
                print(
                    f"[Node {rank}] {i+1}/{len(my_items)} | "
                    f"{n_success} done, {total_clips} clips, {cps:.1f} clips/s | "
                    f"dl={total_dl:.0f}s enc={total_enc:.0f}s ratio={ratio:.2f} "
                    f"({'COMPUTE' if ratio > 0.5 else 'IO'}-bound)",
                    flush=True,
                )

        except Exception as e:
            errors.append(f"{rid[:30]}: {e}")

    wall_clock = time.time() - t_start

    # Save node results
    result = {
        "node_rank": rank,
        "num_nodes": num_nodes,
        "wall_clock": wall_clock,
        "n_success": n_success,
        "n_assigned": len(my_items),
        "total_clips": total_clips,
        "clips_per_sec": total_clips / wall_clock if wall_clock > 0 else 0,
        "total_dl": total_dl,
        "total_enc": total_enc,
        "errors": errors[:10],
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTPUT_PREFIX}results_node_{rank}.json",
        Body=json.dumps(result, indent=2),
    )
    print(
        f"[Node {rank}] DONE: {n_success} renderings, {total_clips} clips "
        f"in {wall_clock:.0f}s ({result['clips_per_sec']:.1f} clips/s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
