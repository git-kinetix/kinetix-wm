#!/usr/bin/env python3
"""Fetch data from Prod_Renderings DynamoDB + S3 and build HDF5 for LeWM training.

Queries the Prod_Renderings DynamoDB table, downloads frames (from frames_uri)
and NPZ camera data (from features_uri), computes delta poses from consecutive
camera extrinsics, and writes the HDF5 dataset for stable_worldmodel.

Usage:
    python scripts/fetch_prod_renderings.py \
        --max-items 100 \
        --output prod_beta0.h5 \
        --image-size 224 \
        --rotation-repr 6d
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import boto3
import h5py
import numpy as np
from PIL import Image

# Add scripts/ to path for conversion utilities
sys.path.insert(0, str(Path(__file__).parent))
from convert_npz_to_hdf5 import (
    convert_rotation,
    rotation_matrix_to_6d,
    rotation_matrix_to_axis_angle,
    write_hdf5,
    validate_hdf5,
)


# ---------------------------------------------------------------------------
# AWS helpers
# ---------------------------------------------------------------------------

def get_aws_clients(profile: str = "rd-ireland", region: str = "eu-west-1"):
    """Create boto3 DynamoDB and S3 clients."""
    session = boto3.Session(profile_name=profile, region_name=region)
    dynamodb = session.resource("dynamodb", region_name=region)
    s3 = session.client("s3", region_name=region)
    return dynamodb, s3


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


# ---------------------------------------------------------------------------
# DynamoDB scan
# ---------------------------------------------------------------------------

def scan_prod_renderings(dynamodb, max_items: int | None = None) -> list[dict]:
    """Scan Prod_Renderings table and return items with both frames_uri and features_uri."""
    table = dynamodb.Table("Prod_Renderings")
    items = []
    scan_kwargs = {}

    print(f"Scanning Prod_Renderings (max_items={max_items}) ...")
    while True:
        response = table.scan(**scan_kwargs)
        for item in response.get("Items", []):
            # Only keep items that have both frames and NPZ data
            if item.get("frames_uri") and item.get("features_uri"):
                items.append(item)
                if max_items and len(items) >= max_items:
                    print(f"  Found {len(items)} items with frames + NPZ")
                    return items
        if "LastEvaluatedKey" not in response:
            break
        scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

    print(f"  Found {len(items)} items with frames + NPZ")
    return items


# ---------------------------------------------------------------------------
# S3 frame download
# ---------------------------------------------------------------------------

def download_frames_from_video(s3, rendering_id: str, image_size: int = 224,
                               bucket: str = "kinetix-rd-storage") -> np.ndarray | None:
    """Download a compressed video from S3 and decode frames.

    Videos are at: s3://{bucket}/preprocessed_datasets/SDP_10_static_camera/videos/{rendering_id}/*.mp4
    """
    import av

    prefix = f"preprocessed_datasets/SDP_10_static_camera/videos/{rendering_id}/"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=10)
    mp4_keys = [obj["Key"] for obj in resp.get("Contents", []) if obj["Key"].endswith(".mp4")]

    if not mp4_keys:
        return None

    # Download the video to a temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        s3.download_file(bucket, mp4_keys[0], tmp.name)

        container = av.open(tmp.name)
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_image().convert("RGB")
            img = img.resize((image_size, image_size), Image.BILINEAR)
            frames.append(np.asarray(img))
        container.close()

    if not frames:
        return None
    return np.stack(frames)


def download_frames_from_images(s3, frames_uri: str, image_size: int = 224) -> np.ndarray | None:
    """Fallback: download individual frame images from S3 directory."""
    bucket, prefix = parse_s3_uri(frames_uri)
    if not prefix.endswith("/"):
        prefix += "/"

    keys = []
    continuation_token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        response = s3.list_objects_v2(**kwargs)
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".png", ".jpg", ".jpeg")):
                keys.append(key)
        if not response.get("IsTruncated"):
            break
        continuation_token = response["NextContinuationToken"]

    keys = sorted(keys)
    if not keys:
        return None

    frames = []
    for key in keys:
        resp = s3.get_object(Bucket=bucket, Key=key)
        img = Image.open(BytesIO(resp["Body"].read())).convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        frames.append(np.asarray(img))

    return np.stack(frames)


# ---------------------------------------------------------------------------
# NPZ download + delta pose extraction
# ---------------------------------------------------------------------------

def download_npz(s3, features_uri: str) -> dict | None:
    """Download NPZ from S3 and return as dict of arrays."""
    bucket, key = parse_s3_uri(features_uri)
    try:
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            s3.download_file(bucket, key, tmp.name)
            return dict(np.load(tmp.name, allow_pickle=True))
    except Exception as e:
        print(f"  [warn] Failed to download NPZ {features_uri}: {e}")
        return None


def compute_delta_poses_from_body(
    poses: np.ndarray,
    trans: np.ndarray,
    mode: str = "root",
) -> np.ndarray:
    """Compute frame-to-frame delta poses from SMPL beta-0 human body data.

    Parameters
    ----------
    poses : ndarray, shape (T, J, 3)
        Per-joint axis-angle rotations (J joints, e.g. 102 for SMPL-X).
    trans : ndarray, shape (T, 3)
        Root body translation per frame.
    mode : str
        - "root": delta of root translation + root joint rotation (action_dim=6)
        - "full": delta of root translation + all joint rotations (action_dim=3+J*3)

    Returns
    -------
    actions : ndarray, shape (T, action_dim), float32
        Delta poses. Last row is NaN (episode boundary marker).
    """
    T = poses.shape[0]

    if mode == "root":
        # Root joint rotation (joint 0) + root translation
        root_rot = poses[:, 0, :]  # (T, 3) axis-angle
        d_trans = np.diff(trans, axis=0)  # (T-1, 3)
        d_rot = np.diff(root_rot, axis=0)  # (T-1, 3)
        deltas = np.concatenate([d_trans, d_rot], axis=1)  # (T-1, 6)
    elif mode == "full":
        # All joints flattened + root translation
        all_flat = poses.reshape(T, -1)  # (T, J*3)
        d_trans = np.diff(trans, axis=0)  # (T-1, 3)
        d_joints = np.diff(all_flat, axis=0)  # (T-1, J*3)
        deltas = np.concatenate([d_trans, d_joints], axis=1)  # (T-1, 3+J*3)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Pad last frame with NaN (episode boundary)
    action_dim = deltas.shape[1]
    last_row = np.full((1, action_dim), np.nan, dtype=np.float32)
    actions = np.concatenate([deltas.astype(np.float32), last_row], axis=0)  # (T, action_dim)

    return actions


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_rendering(s3, item: dict, image_size: int, pose_mode: str) -> dict | None:
    """Process a single rendering item into an episode dict."""
    rendering_id = item.get("rendering_id", "unknown")
    frames_uri = item["frames_uri"]
    features_uri = item["features_uri"]

    # Download frames — try compressed video first, fall back to individual images
    pixels = download_frames_from_video(s3, rendering_id, image_size)
    if pixels is None:
        pixels = download_frames_from_images(s3, frames_uri, image_size)
    if pixels is None or len(pixels) < 4:
        print(f"  [skip] {rendering_id}: not enough frames ({0 if pixels is None else len(pixels)})")
        return None

    # Download NPZ
    npz_data = download_npz(s3, features_uri)
    if npz_data is None:
        return None

    # Extract beta-0 human body poses
    if "poses_beta0_world" not in npz_data or "trans_beta0_world" not in npz_data:
        print(f"  [skip] {rendering_id}: missing poses_beta0_world or trans_beta0_world")
        return None

    poses = npz_data["poses_beta0_world"]  # (1, T, J, 3)
    trans = npz_data["trans_beta0_world"]  # (1, T, 3)

    # Remove leading batch dim if present
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses[0]  # (T, J, 3)
    if trans.ndim == 3 and trans.shape[0] == 1:
        trans = trans[0]  # (T, 3)

    # Align frame count (pixels, poses, trans)
    n = min(len(pixels), len(poses), len(trans))
    if n < 4:
        print(f"  [skip] {rendering_id}: too few aligned frames ({n})")
        return None

    pixels = pixels[:n]
    poses = poses[:n]
    trans = trans[:n]

    # Compute delta poses from body motion
    actions = compute_delta_poses_from_body(poses, trans, mode=pose_mode)

    print(f"  [ok] {rendering_id}: {n} frames, action_dim={actions.shape[1]}")
    return {"pixels": pixels, "actions": actions}


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Prod_Renderings data and build HDF5 for LeWM training."
    )
    parser.add_argument("--max-items", type=int, default=100,
                        help="Max renderings to fetch from DynamoDB (default: 100)")
    parser.add_argument("--output", type=str, default="prod_beta0.h5",
                        help="Output HDF5 filename (default: prod_beta0.h5)")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Resize frames to NxN (default: 224)")
    parser.add_argument("--pose-mode", choices=["root", "full"], default="root",
                        help="Pose delta mode: 'root' (6D: root trans+rot) or 'full' (3+J*3: all joints) (default: root)")
    parser.add_argument("--profile", type=str, default="rd-ireland",
                        help="AWS profile (default: rd-ireland)")
    parser.add_argument("--stablewm-home", type=str, default=None,
                        help="Output directory (default: ~/.stable_worldmodel/)")
    args = parser.parse_args()

    # Determine output path
    if args.stablewm_home:
        out_dir = Path(args.stablewm_home)
    else:
        out_dir = Path.home() / ".stable_worldmodel"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / args.output

    # Connect to AWS
    dynamodb, s3 = get_aws_clients(profile=args.profile)

    # Scan DynamoDB
    items = scan_prod_renderings(dynamodb, max_items=args.max_items)
    if not items:
        print("No renderings found. Exiting.")
        sys.exit(1)

    # Process each rendering into episodes
    episodes = []
    for i, item in enumerate(items):
        rid = item.get("rendering_id", f"item_{i}")
        print(f"[{i+1}/{len(items)}] Processing {rid} ...")
        ep = process_rendering(s3, item, args.image_size, args.pose_mode)
        if ep is not None:
            episodes.append(ep)

    if not episodes:
        print("No valid episodes produced. Exiting.")
        sys.exit(1)

    print(f"\nWriting {len(episodes)} episodes to {output_path} ...")
    write_hdf5(episodes, output_path)

    print("Validating ...")
    validate_hdf5(output_path)

    print(f"\nDone. Dataset ready at: {output_path}")
    print(f"To train: python train.py data=prod_beta0 encoder_type=vjepa ...")


if __name__ == "__main__":
    main()
