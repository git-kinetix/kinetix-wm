#!/usr/bin/env python3
"""SkyPilot worker: download videos from S3, encode with ViT-B/16, upload embeddings.

Designed to be compute-bound on GPU when running in the same AWS region as the S3 bucket.
Each worker processes a shard of the manifest (round-robin by worker_id).

Usage:
    python sky/encode_worker.py \
        --manifest-s3 s3://bucket/manifest.json \
        --output-s3 s3://bucket/embeddings/ \
        --checkpoint-s3 s3://bucket/vitb16_pretrained.pt \
        --worker-id 0 --num-workers 4 --batch-size 512
"""

import argparse
import json
import os
import tempfile
import time

import boto3
import h5py
import numpy as np
import torch
import av
from PIL import Image
from urllib.parse import urlparse

# ImageNet normalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_s3(uri):
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


def download_s3(s3, uri, local_path):
    bucket, key = parse_s3(uri)
    s3.download_file(bucket, key, local_path)


def upload_s3(s3, local_path, uri):
    bucket, key = parse_s3(uri)
    s3.upload_file(local_path, bucket, key)


def load_encoder(s3, checkpoint_s3, device):
    """Download and load ViT-B/16 checkpoint."""
    import timm

    # Download checkpoint
    local_ckpt = "/tmp/vitb16.pt"
    if not os.path.exists(local_ckpt):
        print(f"Downloading checkpoint from {checkpoint_s3}...")
        download_s3(s3, checkpoint_s3, local_ckpt)

    ckpt = torch.load(local_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    # Detect architecture
    embed_key = [k for k in state_dict if "patch_embed" in k and "weight" in k]
    embed_dim = int(state_dict[embed_key[0]].shape[0]) if embed_key else 768
    depth = len({k.split(".")[1] for k in state_dict if k.startswith("blocks.")})
    num_heads = {768: 12, 1024: 16, 1280: 16}.get(embed_dim, 12)

    encoder = timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=0,
        embed_dim=embed_dim, depth=depth or 12, num_heads=num_heads,
    )
    encoder.load_state_dict(state_dict, strict=False)
    encoder = encoder.to(device).eval()
    encoder.requires_grad_(False)
    print(f"Encoder loaded: embed_dim={embed_dim}, depth={depth}")
    return encoder, embed_dim


def decode_video(s3, video_s3_key, bucket, img_size=224):
    """Download and decode video to numpy frames."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        s3.download_file(bucket, video_s3_key, tmp.name)
        container = av.open(tmp.name)
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_image().convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            frames.append(np.asarray(img))
        container.close()
    return np.stack(frames) if frames else None


def encode_frames(encoder, frames_np, device, batch_size=512):
    """Encode (N, H, W, 3) uint8 → (N, D) float32."""
    all_embs = []
    for i in range(0, len(frames_np), batch_size):
        batch = frames_np[i:i + batch_size]
        t = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
        t = (t - MEAN) / STD
        t = t.to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = encoder(t)
            emb = out if out.ndim == 2 else out.mean(dim=1)
        all_embs.append(emb.float().cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def extract_actions_from_npz(s3, npz_s3_uri):
    """Download NPZ, extract 6D root body delta poses."""
    bucket, key = parse_s3(npz_s3_uri)
    with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
        try:
            s3.download_file(bucket, key, tmp.name)
            npz = np.load(tmp.name, allow_pickle=True)
        except Exception:
            return None

    if "poses_beta0_world" not in npz or "trans_beta0_world" not in npz:
        return None

    poses = npz["poses_beta0_world"]
    trans = npz["trans_beta0_world"]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-s3", required=True)
    parser.add_argument("--output-s3", required=True)
    parser.add_argument("--checkpoint-s3", required=True)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    s3 = boto3.client("s3", region_name="eu-west-1")

    # Load manifest
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        download_s3(s3, args.manifest_s3, tmp.name)
        with open(tmp.name) as f:
            manifest = json.load(f)

    items = manifest["items"]
    print(f"Manifest: {len(items)} renderings total")

    # Shard: this worker processes items[worker_id::num_workers]
    my_items = items[args.worker_id::args.num_workers]
    print(f"Worker {args.worker_id}/{args.num_workers}: processing {len(my_items)} renderings")

    # Load encoder
    encoder, embed_dim = load_encoder(s3, args.checkpoint_s3, device)

    # Process each rendering
    ep_embeddings = []
    ep_actions = []
    ep_lengths = []
    ep_rids = []

    t_download = 0
    t_encode = 0

    for i, item in enumerate(my_items):
        rid = item["rendering_id"]
        video_key = item["video_key"]
        npz_uri = item["features_uri"]
        bucket = item.get("bucket", "kinetix-rd-storage")

        # Download + decode video
        t0 = time.time()
        frames = decode_video(s3, video_key, bucket)
        if frames is None or len(frames) < 4:
            print(f"  [{i+1}/{len(my_items)}] {rid[:40]}... SKIP (no video)")
            continue

        actions = extract_actions_from_npz(s3, npz_uri)
        if actions is None:
            print(f"  [{i+1}/{len(my_items)}] {rid[:40]}... SKIP (no poses)")
            continue
        t_download += time.time() - t0

        # Align
        n = min(len(frames), len(actions))
        frames = frames[:n]
        actions = actions[:n]

        # Encode
        t0 = time.time()
        embs = encode_frames(encoder, frames, device, args.batch_size)
        t_encode += time.time() - t0

        ep_embeddings.append(embs)
        ep_actions.append(actions)
        ep_lengths.append(n)
        ep_rids.append(rid)

        if (i + 1) % 10 == 0:
            total_frames = sum(ep_lengths)
            ratio = t_encode / (t_download + 1e-8)
            print(f"  [{i+1}/{len(my_items)}] {len(ep_lengths)} eps, {total_frames} frames | "
                  f"dl={t_download:.0f}s enc={t_encode:.0f}s ratio={ratio:.2f} "
                  f"({'COMPUTE-BOUND' if ratio > 0.5 else 'IO-BOUND'})")

    if not ep_lengths:
        print("No episodes processed")
        return

    # Write HDF5
    total = sum(ep_lengths)
    offsets = np.concatenate([[0], np.cumsum(ep_lengths[:-1])]).astype(np.int32)
    local_h5 = f"/tmp/shard_{args.worker_id}.h5"

    print(f"\nWriting {len(ep_lengths)} episodes, {total} frames → {local_h5}")
    with h5py.File(local_h5, "w") as f:
        f.create_dataset("ep_len", data=np.array(ep_lengths, dtype=np.int32))
        f.create_dataset("ep_offset", data=offsets)
        emb_ds = f.create_dataset("embedding", shape=(total, embed_dim), dtype=np.float32)
        act_ds = f.create_dataset("action", shape=(total, 6), dtype=np.float32)
        idx = 0
        for embs, acts in zip(ep_embeddings, ep_actions):
            n = len(embs)
            emb_ds[idx:idx + n] = embs
            act_ds[idx:idx + n] = acts
            idx += n

    # Upload to S3
    output_key = f"{parse_s3(args.output_s3)[1]}shard_{args.worker_id}.h5"
    output_uri = f"s3://{parse_s3(args.output_s3)[0]}/{output_key}"
    print(f"Uploading to {output_uri}...")
    upload_s3(s3, local_h5, output_uri)

    print(f"\nDone. Worker {args.worker_id}: {total} frames, "
          f"dl={t_download:.0f}s enc={t_encode:.0f}s "
          f"({'COMPUTE-BOUND' if t_encode > t_download else 'IO-BOUND'})")


if __name__ == "__main__":
    main()
