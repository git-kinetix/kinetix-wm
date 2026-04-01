"""SkyPilot ViT-B/16 embedding + body pose extraction — optimized for multi-node.

Based on ~/Repos/parallel-processing/methods/skypilot/process.py pattern.
Each node processes renderings[rank::num_nodes], uploads per-rendering NPZ to S3.

Key optimizations:
- Video download (single file) instead of 450 frame downloads
- torch.compile for faster ViT-B/16 inference
- fp16 throughout
- Concurrent NPZ + video download
"""

import io
import json
import os
import sys
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor

import boto3
import botocore.config
import numpy as np

BUCKET = "kinetix-rd-storage"
MANIFEST_KEY = "lewm/true_motion_manifest.json"
CHECKPOINT_KEY = "lewm/vitb16_pretrained.pt"
OUTPUT_PREFIX = "lewm/embeddings_sky/"


def make_s3_client():
    cfg = botocore.config.Config(
        max_pool_connections=64,
        connect_timeout=5,
        read_timeout=10,
        retries={"max_attempts": 3, "mode": "adaptive"},
        tcp_keepalive=True,
    )
    return boto3.client("s3", region_name="eu-west-1", config=cfg)


def download_and_decode_video(s3, video_key, img_size=224):
    """Download video from S3 and decode to numpy frames."""
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


def extract_body_poses(s3, features_uri):
    """Download NPZ and extract 6D root delta poses."""
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
    import timm

    t_start = time.time()

    rank = int(os.environ.get("SKYPILOT_NODE_RANK", 0))
    num_nodes = int(os.environ.get("SKYPILOT_NUM_NODES", 1))
    print(f"[Node {rank}/{num_nodes}] Starting", flush=True)

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

    # Load ViT-B/16
    print(f"[Node {rank}] Loading ViT-B/16...", flush=True)
    t_model = time.time()

    # Download checkpoint from S3
    ckpt_path = "/tmp/vitb16.pt"
    if not os.path.exists(ckpt_path):
        s3.download_file(BUCKET, CHECKPOINT_KEY, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    model.load_state_dict(state_dict, strict=False)
    model = model.to("cuda").eval().half()

    try:
        model = torch.compile(model)
        print(f"[Node {rank}] torch.compile OK", flush=True)
    except Exception as e:
        print(f"[Node {rank}] torch.compile failed: {e}", flush=True)

    embed_dim = model.embed_dim  # 768

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1).half()
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1).half()

    # Warmup
    with torch.no_grad():
        _ = model(torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float16))
    print(f"[Node {rank}] Model ready in {time.time()-t_model:.1f}s", flush=True)

    # Auto batch size
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    batch_size = 128 if gpu_mem >= 20 else 64 if gpu_mem >= 14 else 32
    print(f"[Node {rank}] GPU: {torch.cuda.get_device_name(0)}, {gpu_mem:.0f}GB, bs={batch_size}", flush=True)

    # Process renderings
    total_frames = 0
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

            if frames is None or actions is None or len(frames) < 4:
                continue

            n = min(len(frames), len(actions))
            frames = frames[:n]
            actions = actions[:n]
            t_dl = time.time() - t0

            # Encode
            t1 = time.time()
            all_embs = []
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                for j in range(0, n, batch_size):
                    batch = frames[j:j+batch_size]
                    t = torch.from_numpy(batch).permute(0, 3, 1, 2).to("cuda", dtype=torch.float16) / 255.0
                    t = (t - mean) / std
                    emb = model(t)
                    all_embs.append(emb.float().cpu().numpy())
            embeddings = np.concatenate(all_embs, axis=0)
            t_enc = time.time() - t1

            # Upload per-rendering NPZ to S3
            buf = io.BytesIO()
            np.savez_compressed(buf, embedding=embeddings, action=actions)
            buf.seek(0)
            s3.put_object(Bucket=BUCKET, Key=f"{OUTPUT_PREFIX}{rid}.npz", Body=buf.getvalue())

            total_frames += n
            total_dl += t_dl
            total_enc += t_enc
            n_success += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(my_items):
                elapsed = time.time() - t_start
                fps = total_frames / elapsed if elapsed > 0 else 0
                ratio = total_enc / (total_dl + 1e-8)
                print(
                    f"[Node {rank}] {i+1}/{len(my_items)} | "
                    f"{n_success} done, {total_frames} frames, {fps:.0f} fps | "
                    f"dl={total_dl:.0f}s enc={total_enc:.0f}s ratio={ratio:.2f} "
                    f"({'COMPUTE' if ratio > 0.5 else 'IO'}-bound)",
                    flush=True,
                )

        except Exception as e:
            errors.append(f"{rid[:30]}: {e}")

    wall_clock = time.time() - t_start

    # Save node results
    result = {
        "node_rank": rank, "num_nodes": num_nodes,
        "wall_clock": wall_clock,
        "n_success": n_success, "n_assigned": len(my_items),
        "total_frames": total_frames,
        "fps": total_frames / wall_clock if wall_clock > 0 else 0,
        "total_dl": total_dl, "total_enc": total_enc,
        "errors": errors[:10],
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTPUT_PREFIX}results_node_{rank}.json",
        Body=json.dumps(result, indent=2),
    )
    print(
        f"[Node {rank}] DONE: {n_success} renderings, {total_frames} frames "
        f"in {wall_clock:.0f}s ({result['fps']:.0f} fps)",
        flush=True,
    )


if __name__ == "__main__":
    main()
