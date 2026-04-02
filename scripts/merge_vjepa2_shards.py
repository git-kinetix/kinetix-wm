#!/usr/bin/env python3
"""Merge per-rendering VJEPA2 NPZ shards from S3 into a single clip-level HDF5.

Downloads all NPZ files from s3://kinetix-rd-storage/lewm/vjepa2_embeddings/,
concatenates clip embeddings and aggregated clip-level actions into:
    $STABLEWM_HOME/true_motion_vjepa2emb.h5

HDF5 layout:
    embedding : (total_clips, 1024) float32  — VJEPA2 clip embeddings
    action    : (total_clips, 6)    float32  — aggregated clip-level actions
    ep_len    : (n_episodes,)       int32    — clips per episode
    ep_offset : (n_episodes,)       int32    — cumulative clip offset

Action aggregation per clip:
    Each clip covers 64 frames with 6D per-frame deltas (dx, dy, dz, dr1, dr2, dr3).
    Translation: sum 64 (dx, dy, dz) -> net displacement.
    Rotation: sum 64 (dr1, dr2, dr3) -> net rotation delta (axis-angle sum approximation).
    If any frame's action is NaN, the clip action is NaN.

Usage:
    python scripts/merge_vjepa2_shards.py
"""

import io
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import boto3
import botocore.config
import h5py
import hdf5plugin  # noqa: F401 — registers HDF5 compression codecs
import numpy as np

BUCKET = "kinetix-rd-storage"
PREFIX = "lewm/vjepa2_embeddings/"
EMBED_DIM = 1024
ACTION_DIM = 6
CLIP_SIZE = 64


def make_s3_client():
    cfg = botocore.config.Config(
        max_pool_connections=64,
        connect_timeout=5,
        read_timeout=30,
        retries={"max_attempts": 3, "mode": "adaptive"},
        tcp_keepalive=True,
    )
    return boto3.client("s3", region_name="eu-west-1", config=cfg)


def list_npz_keys(s3):
    """List all .npz keys under PREFIX, excluding results_node_*.json."""
    keys = []
    continuation = {}
    while True:
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX, **continuation)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".npz"):
                keys.append(key)
        if not resp.get("IsTruncated"):
            break
        continuation["ContinuationToken"] = resp["NextContinuationToken"]
    return sorted(keys)


def aggregate_clip_actions(actions, n_clips, clip_size):
    """Aggregate per-frame 6D deltas into clip-level actions.

    Args:
        actions: (T, 6) float32 per-frame deltas
        n_clips: number of clips
        clip_size: frames per clip (64)

    Returns:
        (n_clips, 6) float32 clip-level actions.
        If any frame within a clip has NaN, the entire clip action is NaN.
    """
    clip_actions = np.empty((n_clips, ACTION_DIM), dtype=np.float32)
    for c in range(n_clips):
        start = c * clip_size
        end = min(start + clip_size, len(actions))
        chunk = actions[start:end]  # (<=64, 6)
        if np.any(np.isnan(chunk)):
            clip_actions[c] = np.nan
        else:
            clip_actions[c] = chunk.sum(axis=0)
    return clip_actions


def main():
    t_start = time.time()
    cache_dir = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))
    os.makedirs(cache_dir, exist_ok=True)
    output_path = os.path.join(cache_dir, "true_motion_vjepa2emb.h5")

    s3 = make_s3_client()

    # List all NPZ files
    print("Listing NPZ shards from S3...")
    keys = list_npz_keys(s3)
    print(f"  Found {len(keys)} NPZ files")
    if not keys:
        print("No NPZ files found. Exiting.")
        return

    # First pass: download all shards and collect data
    all_embeddings = []
    all_actions = []
    ep_lengths = []
    n_errors = 0

    for i, key in enumerate(keys):
        try:
            with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
                s3.download_file(BUCKET, key, tmp.name)
                npz = np.load(tmp.name, allow_pickle=True)

            clip_embeddings = npz["clip_embeddings"]  # (n_clips, 1024)
            actions = npz["action"]                    # (T, 6)
            n_clips = int(npz["n_clips"])
            clip_size = int(npz["clip_size"])

            if clip_embeddings.ndim != 2 or clip_embeddings.shape[1] != EMBED_DIM:
                print(f"  WARN: {key} has unexpected embedding shape {clip_embeddings.shape}, skipping")
                n_errors += 1
                continue

            # Aggregate per-frame actions to clip-level
            clip_actions = aggregate_clip_actions(actions, n_clips, clip_size)

            all_embeddings.append(clip_embeddings.astype(np.float32))
            all_actions.append(clip_actions)
            ep_lengths.append(n_clips)

        except Exception as e:
            n_errors += 1
            if n_errors <= 10:
                print(f"  ERROR [{key}]: {e}")
            continue

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t_start
            total_clips_so_far = sum(ep_lengths)
            print(
                f"  [{i+1}/{len(keys)}] {len(ep_lengths)} episodes, "
                f"{total_clips_so_far} clips, {elapsed:.0f}s elapsed"
            )

    if not ep_lengths:
        print("No valid episodes. Exiting.")
        return

    # Build arrays
    total_clips = sum(ep_lengths)
    ep_len_arr = np.array(ep_lengths, dtype=np.int32)
    ep_offset_arr = np.concatenate([[0], np.cumsum(ep_len_arr[:-1])]).astype(np.int32)

    print(f"\nConcatenating {len(ep_lengths)} episodes, {total_clips} total clips...")
    embeddings = np.concatenate(all_embeddings, axis=0)  # (total_clips, 1024)
    actions = np.concatenate(all_actions, axis=0)         # (total_clips, 6)

    # Write HDF5
    print(f"Writing HDF5 → {output_path}")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("embedding", data=embeddings, dtype=np.float32)
        f.create_dataset("action", data=actions, dtype=np.float32)
        f.create_dataset("ep_len", data=ep_len_arr)
        f.create_dataset("ep_offset", data=ep_offset_arr)

    elapsed = time.time() - t_start
    file_size = os.path.getsize(output_path)
    n_episodes = len(ep_lengths)

    print(f"\n{'='*60}")
    print(f"  Episodes:    {n_episodes:,}")
    print(f"  Total clips: {total_clips:,}")
    print(f"  Embed shape: ({total_clips}, {EMBED_DIM})")
    print(f"  Action shape:({total_clips}, {ACTION_DIM})")
    print(f"  File size:   {file_size / 1e6:.1f} MB ({file_size / 1e9:.2f} GB)")
    print(f"  Errors:      {n_errors}")
    print(f"  Wall time:   {elapsed:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
