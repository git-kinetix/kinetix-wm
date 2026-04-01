#!/usr/bin/env python3
"""Streaming fetch + encode: download SDP10 videos, encode with ViT-B/16, save embeddings.

Filters Prod_Renderings by animation_radius > threshold via Dev_Animations.
Processes each rendering as: video → frames → ViT-B/16 → 768d embeddings.
Stores only embeddings + actions (no raw pixels) → ~5GB for 1.2M frames.

Usage:
    python scripts/fetch_and_encode_true_motion.py \
        --min-radius 200 \
        --max-items 2722 \
        --output true_motion_vitb16emb.h5 \
        --checkpoint $STABLEWM_HOME/vjepa_checkpoints/vitb16_pretrained.pt
"""

import sys, os, argparse, tempfile
from pathlib import Path
from io import BytesIO

import boto3
import h5py
import numpy as np
import torch
import av
from PIL import Image
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import load_vjepa_encoder

# ImageNet normalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_s3_uri(uri):
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


def get_filtered_renderings(dynamodb, min_radius, max_items):
    """Get Prod_Renderings items whose animation has radius > min_radius."""
    # Step 1: Get qualifying animation_ids
    anim_table = dynamodb.Table("Dev_Animations")
    qualifying_aids = set()
    scan_kw = {}
    print(f"Scanning Dev_Animations for radius > {min_radius}...")
    while True:
        resp = anim_table.scan(**scan_kw)
        for item in resp.get("Items", []):
            r = item.get("animation_radius")
            if r is not None:
                try:
                    if float(r) > min_radius:
                        qualifying_aids.add(item["animation_id"])
                except (ValueError, TypeError):
                    pass
        if "LastEvaluatedKey" not in resp:
            break
        scan_kw["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    print(f"  {len(qualifying_aids)} qualifying animations")

    # Step 2: Get renderings for those animations
    prod_table = dynamodb.Table("Prod_Renderings")
    items = []
    scan_kw = {}
    print(f"Scanning Prod_Renderings (max {max_items})...")
    while True:
        resp = prod_table.scan(**scan_kw)
        for item in resp.get("Items", []):
            if (item.get("animation_id") in qualifying_aids
                and item.get("features_uri") and item.get("frames_uri")):
                items.append(item)
                if len(items) >= max_items:
                    print(f"  {len(items)} renderings collected")
                    return items
        if "LastEvaluatedKey" not in resp:
            break
        scan_kw["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    print(f"  {len(items)} renderings collected")
    return items


def decode_video_frames(s3, rendering_id, img_size=224, bucket="kinetix-rd-storage"):
    """Download video from S3 and decode to numpy array."""
    prefix = f"preprocessed_datasets/SDP_10_static_camera/videos/{rendering_id}/"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=5)
    mp4_keys = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".mp4")]
    if not mp4_keys:
        return None
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        s3.download_file(bucket, mp4_keys[0], tmp.name)
        container = av.open(tmp.name)
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_image().convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            frames.append(np.asarray(img))
        container.close()
    return np.stack(frames) if frames else None


def encode_frames(encoder, frames_np, device, batch_size=256):
    """Encode numpy frames (N, H, W, 3) uint8 → (N, 768) float32 embeddings."""
    all_embs = []
    for i in range(0, len(frames_np), batch_size):
        batch = frames_np[i:i+batch_size]
        # Convert to tensor: (B, 3, H, W) float, ImageNet normalized
        t = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
        t = (t - MEAN) / STD
        t = t.to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = encoder(t)
            if torch.is_tensor(out):
                emb = out if out.ndim == 2 else out.mean(dim=1)
            elif hasattr(out, "last_hidden_state"):
                emb = out.last_hidden_state.mean(dim=1)
            else:
                emb = out
        all_embs.append(emb.float().cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def extract_actions(s3, features_uri):
    """Download NPZ, extract root body delta poses (6D)."""
    bucket, key = parse_s3_uri(features_uri)
    try:
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
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
    parser.add_argument("--min-radius", type=float, default=200)
    parser.add_argument("--max-items", type=int, default=2722)
    parser.add_argument("--output", default="true_motion_vitb16emb.h5")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--profile", default="rd-ireland")
    args = parser.parse_args()

    cache = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load encoder
    print(f"Loading ViT-B/16 from {args.checkpoint}")
    encoder = load_vjepa_encoder(args.checkpoint, patch_size=16, img_size=224)
    encoder = encoder.to(device).eval()
    encoder.requires_grad_(False)
    embed_dim = encoder.embed_dim
    print(f"  embed_dim={embed_dim}")

    # AWS clients
    session = boto3.Session(profile_name=args.profile, region_name="eu-west-1")
    dynamodb = session.resource("dynamodb", region_name="eu-west-1")
    s3 = session.client("s3", region_name="eu-west-1")

    # Get filtered renderings
    items = get_filtered_renderings(dynamodb, args.min_radius, args.max_items)
    if not items:
        print("No renderings found"); return

    # Process streaming: video → encode → write
    output_path = os.path.join(cache, args.output)
    ep_embeddings = []
    ep_actions = []
    ep_lengths = []

    for i, item in enumerate(items):
        rid = item.get("rendering_id", "?")
        print(f"[{i+1}/{len(items)}] {rid[:50]}...", end=" ")

        # Decode video
        frames = decode_video_frames(s3, rid)
        if frames is None or len(frames) < 4:
            print("SKIP (no video)")
            continue

        # Extract actions from NPZ
        actions = extract_actions(s3, item["features_uri"])
        if actions is None:
            print("SKIP (no poses)")
            continue

        # Align lengths
        n = min(len(frames), len(actions))
        if n < 4:
            print("SKIP (too short)")
            continue
        frames = frames[:n]
        actions = actions[:n]

        # Encode frames → embeddings
        embs = encode_frames(encoder, frames, device, args.batch_size)

        ep_embeddings.append(embs)
        ep_actions.append(actions)
        ep_lengths.append(n)
        print(f"OK ({n} frames)")

    if not ep_lengths:
        print("No episodes processed"); return

    # Write HDF5
    total = sum(ep_lengths)
    offsets = np.concatenate([[0], np.cumsum(ep_lengths[:-1])]).astype(np.int32)

    print(f"\nWriting {len(ep_lengths)} episodes, {total} frames → {output_path}")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("ep_len", data=np.array(ep_lengths, dtype=np.int32))
        f.create_dataset("ep_offset", data=offsets)
        emb_ds = f.create_dataset("embedding", shape=(total, embed_dim), dtype=np.float32)
        act_ds = f.create_dataset("action", shape=(total, 6), dtype=np.float32)
        idx = 0
        for embs, acts in zip(ep_embeddings, ep_actions):
            n = len(embs)
            emb_ds[idx:idx+n] = embs
            act_ds[idx:idx+n] = acts
            idx += n

    print(f"Done. {total:,} frames, {os.path.getsize(output_path)/1e9:.1f} GB")


if __name__ == "__main__":
    main()
