#!/usr/bin/env python3
"""Convert NPZ delta-pose actions + rendered frames into HDF5 for stable_worldmodel.

This script reads delta-pose trajectories stored as NPZ files and RGB frames
from a rendered-frames directory (e.g. a ``Prod_renderings`` database export),
then writes a single ``.h5`` file whose layout matches what
``stable_worldmodel.data.HDF5Dataset`` expects.

Expected input formats
----------------------
**NPZ files** (one or more of the following layouts is auto-detected):

1. *Single trajectory per file* -- arrays named ``delta_poses`` (N, 7) and,
   optionally, ``frames`` (N, H, W, 3).  If frames are absent they are read
   from ``--frames-dir``.
2. *Multiple episodes in one file* -- an ``episode_ids`` integer array of
   length N that maps every timestep to an episode, plus ``delta_poses`` and
   optionally ``frames``.
3. *Directory of NPZ files* -- each file is treated as one episode.  Files
   are sorted lexicographically so episode ordering is deterministic.

**Frames directory** (``--frames-dir``):

    <frames-dir>/
        episode_000/
            frame_0000.png
            frame_0001.png
            ...
        episode_001/
            ...

Sub-directories are sorted lexicographically and matched 1-to-1 with episodes
found in the NPZ data.  Images inside each sub-directory are also sorted.

Output HDF5 schema
-------------------
Datasets stored at the root of the ``.h5`` file:

* ``ep_len``    -- int32, shape ``(num_episodes,)``
* ``ep_offset`` -- int32, shape ``(num_episodes,)``
* ``pixels``    -- uint8,  shape ``(total_steps, H, W, 3)``
* ``action``    -- float32, shape ``(total_steps, action_dim)``

``action_dim`` depends on ``--rotation-repr``:

* ``euler``      -- 6  (dx dy dz + euler angles)
* ``quaternion`` -- 7  (dx dy dz + qw qx qy qz)
* ``6d``         -- 9  (dx dy dz + 6D rotation)
* ``axis_angle`` -- 6  (dx dy dz + axis-angle)

The last action of every episode is set to NaN to mark episode boundaries.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Rotation representation helpers
# ---------------------------------------------------------------------------

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (wxyz) to 3x3 rotation matrix.

    Parameters
    ----------
    q : ndarray, shape (..., 4)
        Quaternion in ``(w, x, y, z)`` convention.

    Returns
    -------
    R : ndarray, shape (..., 3, 3)
    """
    q = np.asarray(q, dtype=np.float64)
    # normalise
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def euler_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    """Convert extrinsic XYZ Euler angles (in radians) to rotation matrix.

    Parameters
    ----------
    euler : ndarray, shape (..., 3)
        Roll, pitch, yaw.

    Returns
    -------
    R : ndarray, shape (..., 3, 3)
    """
    euler = np.asarray(euler, dtype=np.float64)
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr
    return R


def rotation_matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """Extract the 6D rotation representation (Zhou et al., 2019).

    The representation is the first two columns of the rotation matrix,
    flattened: ``[r1, r2]`` where each ``r`` has 3 components.

    Parameters
    ----------
    R : ndarray, shape (..., 3, 3)

    Returns
    -------
    rot6d : ndarray, shape (..., 6)
    """
    return R[..., :2].reshape(R.shape[:-2] + (6,))


def rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to axis-angle (rotation vector).

    Parameters
    ----------
    R : ndarray, shape (..., 3, 3)

    Returns
    -------
    aa : ndarray, shape (..., 3)
    """
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    n = R_flat.shape[0]
    aa = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        cos_angle = (np.trace(R_flat[i]) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.acos(cos_angle)
        if abs(angle) < 1e-6:
            continue
        if abs(angle - math.pi) < 1e-6:
            # Find the column of (R + I) with largest norm
            M = R_flat[i] + np.eye(3)
            col = np.argmax(np.linalg.norm(M, axis=0))
            axis = M[:, col]
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            aa[i] = axis * angle
        else:
            skew = (R_flat[i] - R_flat[i].T) / (2.0 * math.sin(angle))
            axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
            aa[i] = axis * angle
    return aa.reshape(batch_shape + (3,))


def convert_rotation(
    rotation: np.ndarray,
    source_repr: str,
    target_repr: str,
) -> np.ndarray:
    """Convert rotation data between representations.

    Supported source formats:
        ``"quaternion"`` -- (w, x, y, z), shape (..., 4)
        ``"euler"``      -- extrinsic XYZ radians, shape (..., 3)
        ``"matrix"``     -- 3x3 rotation matrix, shape (..., 3, 3)

    Supported target formats:
        ``"6d"``, ``"euler"``, ``"quaternion"``, ``"axis_angle"``

    Parameters
    ----------
    rotation : ndarray
        Rotation data in *source_repr* format.
    source_repr : str
    target_repr : str

    Returns
    -------
    ndarray
        Rotation in *target_repr* format.
    """
    # Step 1 -- get rotation matrix
    if source_repr == "quaternion":
        R = quaternion_to_rotation_matrix(rotation)
    elif source_repr == "euler":
        R = euler_to_rotation_matrix(rotation)
    elif source_repr == "matrix":
        R = np.asarray(rotation, dtype=np.float64)
    else:
        raise ValueError(f"Unknown source representation: {source_repr}")

    # Step 2 -- convert to target
    if target_repr == "6d":
        return rotation_matrix_to_6d(R).astype(np.float32)
    elif target_repr == "axis_angle":
        return rotation_matrix_to_axis_angle(R).astype(np.float32)
    elif target_repr == "quaternion":
        # Already have quaternion if source was quaternion; otherwise compute.
        if source_repr == "quaternion":
            return rotation.astype(np.float32)
        raise NotImplementedError(
            "matrix/euler -> quaternion not yet implemented; "
            "use scipy.spatial.transform.Rotation if needed."
        )
    elif target_repr == "euler":
        if source_repr == "euler":
            return rotation.astype(np.float32)
        raise NotImplementedError(
            "matrix/quaternion -> euler not yet implemented; "
            "use scipy.spatial.transform.Rotation if needed."
        )
    else:
        raise ValueError(f"Unknown target representation: {target_repr}")


# ---------------------------------------------------------------------------
# Delta-pose formatting
# ---------------------------------------------------------------------------

_ROT_DIM = {
    "euler": 3,
    "quaternion": 4,
    "6d": 6,
    "axis_angle": 3,
}


def format_delta_pose(
    delta_poses: np.ndarray,
    rotation_repr: str = "6d",
) -> np.ndarray:
    """Convert raw delta poses into the desired rotation representation.

    The input ``delta_poses`` is expected to have shape ``(N, 7)`` where the
    first 3 columns are translational deltas ``(dx, dy, dz)`` and columns
    3--7 are the rotation as a **quaternion (wxyz)**.

    If the input has shape ``(N, 6)`` it is assumed to already carry Euler
    angles (columns 3--6) and is converted accordingly.

    Parameters
    ----------
    delta_poses : ndarray, shape (N, 6) or (N, 7)
    rotation_repr : str
        One of ``"euler"``, ``"quaternion"``, ``"6d"``, ``"axis_angle"``.

    Returns
    -------
    actions : ndarray, shape (N, 3 + rot_dim), float32
    """
    delta_poses = np.asarray(delta_poses, dtype=np.float64)
    translation = delta_poses[:, :3]

    if delta_poses.shape[1] == 7:
        source_repr = "quaternion"
        rotation_raw = delta_poses[:, 3:7]
    elif delta_poses.shape[1] == 6:
        source_repr = "euler"
        rotation_raw = delta_poses[:, 3:6]
    else:
        raise ValueError(
            f"delta_poses has {delta_poses.shape[1]} columns; expected 6 or 7."
        )

    if source_repr == rotation_repr:
        rotation_out = rotation_raw.astype(np.float32)
    else:
        rotation_out = convert_rotation(rotation_raw, source_repr, rotation_repr)

    return np.concatenate(
        [translation.astype(np.float32), rotation_out], axis=1
    )


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def load_frames_from_dir(
    episode_dir: Path,
    image_size: int | None = None,
) -> np.ndarray:
    """Load all image files in *episode_dir* as an (N, H, W, 3) uint8 array.

    Images are sorted lexicographically.  Supported extensions: png, jpg, jpeg.
    """
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(
        p for p in episode_dir.iterdir() if p.suffix.lower() in exts
    )
    if not paths:
        raise FileNotFoundError(f"No image files found in {episode_dir}")

    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if image_size is not None:
            img = img.resize((image_size, image_size), Image.BILINEAR)
        frames.append(np.asarray(img))
    return np.stack(frames)


# ---------------------------------------------------------------------------
# Episode extraction from NPZ
# ---------------------------------------------------------------------------

def _load_episodes_from_single_npz(
    npz_path: Path,
    frames_dir: Path | None,
    rotation_repr: str,
    image_size: int | None,
) -> list[dict]:
    """Return a list of episode dicts, each with ``"actions"`` and ``"pixels"``."""
    data = np.load(npz_path, allow_pickle=True)

    # Locate delta poses
    if "delta_poses" in data:
        all_poses = data["delta_poses"]
    elif "actions" in data:
        all_poses = data["actions"]
    else:
        raise KeyError(
            f"NPZ file {npz_path} has no 'delta_poses' or 'actions' key. "
            f"Available keys: {list(data.keys())}"
        )

    # Locate frames (may be None -> use frames_dir)
    all_frames: np.ndarray | None = None
    if "frames" in data:
        all_frames = data["frames"]
    elif "pixels" in data:
        all_frames = data["pixels"]

    # Detect episode structure
    if "episode_ids" in data:
        ep_ids = data["episode_ids"].astype(int)
        unique_ids = np.unique(ep_ids)
        episodes = []
        for eid in unique_ids:
            mask = ep_ids == eid
            poses = all_poses[mask]
            frames = all_frames[mask] if all_frames is not None else None
            episodes.append((poses, frames, eid))
    else:
        # Single trajectory
        episodes = [(all_poses, all_frames, 0)]

    # Build episode dicts
    result: list[dict] = []
    ep_dirs: list[Path] | None = None
    if frames_dir is not None and frames_dir.is_dir():
        ep_dirs = sorted(
            d for d in frames_dir.iterdir() if d.is_dir()
        )

    for idx, (poses, frames, _eid) in enumerate(episodes):
        actions = format_delta_pose(poses, rotation_repr)

        if frames is not None:
            pixels = np.asarray(frames, dtype=np.uint8)
            if image_size is not None:
                resized = []
                for f in pixels:
                    img = Image.fromarray(f).resize(
                        (image_size, image_size), Image.BILINEAR
                    )
                    resized.append(np.asarray(img))
                pixels = np.stack(resized)
        elif ep_dirs is not None and idx < len(ep_dirs):
            pixels = load_frames_from_dir(ep_dirs[idx], image_size)
        else:
            raise RuntimeError(
                f"No frames found for episode {idx}: NPZ has no 'frames' key "
                f"and --frames-dir does not contain enough sub-directories."
            )

        # Ensure timestep counts match (use the shorter of the two)
        n = min(len(actions), len(pixels))
        if len(actions) != len(pixels):
            print(
                f"  [warn] episode {idx}: {len(actions)} actions vs "
                f"{len(pixels)} frames -- truncating to {n}."
            )
        result.append({
            "actions": actions[:n],
            "pixels": pixels[:n],
        })

    return result


def load_episodes(
    npz_path: Path,
    frames_dir: Path | None,
    rotation_repr: str = "6d",
    image_size: int | None = None,
) -> list[dict]:
    """Load episodes from NPZ file(s) and (optionally) a frames directory.

    Parameters
    ----------
    npz_path : Path
        Path to a single ``.npz`` file **or** a directory of ``.npz`` files
        (one per episode).
    frames_dir : Path or None
        Directory of per-episode rendered frames.  Required when the NPZ data
        does not embed pixel arrays.
    rotation_repr : str
        Target rotation representation for the action vector.
    image_size : int or None
        If set, resize all frames to ``(image_size, image_size)``.

    Returns
    -------
    list of dict
        Each dict has ``"actions"`` (float32) and ``"pixels"`` (uint8).
    """
    npz_path = Path(npz_path)

    if npz_path.is_dir():
        npz_files = sorted(npz_path.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files in {npz_path}")

        ep_dirs: list[Path] | None = None
        if frames_dir is not None and frames_dir.is_dir():
            ep_dirs = sorted(d for d in frames_dir.iterdir() if d.is_dir())

        episodes: list[dict] = []
        for i, f in enumerate(npz_files):
            single_frames_dir = None
            if ep_dirs is not None and i < len(ep_dirs):
                # Create a temp dir structure that _load_episodes_from_single_npz
                # expects -- or just pass the single ep dir.
                single_frames_dir = ep_dirs[i].parent
                # Actually easier: just load frames directly here.
            ep_list = _load_episodes_from_single_npz(
                f, frames_dir, rotation_repr, image_size
            )
            # When loading a directory of NPZs, each file is one episode,
            # so take only the first result from each file.
            episodes.append(ep_list[0])
        return episodes
    else:
        return _load_episodes_from_single_npz(
            npz_path, frames_dir, rotation_repr, image_size
        )


# ---------------------------------------------------------------------------
# HDF5 writing
# ---------------------------------------------------------------------------

def write_hdf5(
    episodes: Sequence[dict],
    output_path: Path,
) -> None:
    """Write episodes to an HDF5 file in ``stable_worldmodel`` format.

    Parameters
    ----------
    episodes : list of dict
        Each dict must have ``"actions"`` (float32, (T, D)) and
        ``"pixels"`` (uint8, (T, H, W, 3)).
    output_path : Path
        Destination ``.h5`` file.
    """
    num_episodes = len(episodes)
    ep_lengths = np.array([len(ep["actions"]) for ep in episodes], dtype=np.int32)
    total_steps = int(ep_lengths.sum())
    ep_offsets = np.concatenate([[0], np.cumsum(ep_lengths[:-1])]).astype(np.int32)

    # Determine shapes from the first episode
    H, W, C = episodes[0]["pixels"].shape[1:]
    action_dim = episodes[0]["actions"].shape[1]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("ep_len", data=ep_lengths)
        hf.create_dataset("ep_offset", data=ep_offsets)

        px_ds = hf.create_dataset(
            "pixels",
            shape=(total_steps, H, W, C),
            dtype=np.uint8,
            chunks=(1, H, W, C),
        )
        act_ds = hf.create_dataset(
            "action",
            shape=(total_steps, action_dim),
            dtype=np.float32,
        )

        idx = 0
        for i, ep in enumerate(episodes):
            T = ep_lengths[i]
            px_ds[idx : idx + T] = ep["pixels"][:T]

            actions = ep["actions"][:T].copy()
            # Mark last action of the episode as NaN
            actions[-1, :] = np.nan
            act_ds[idx : idx + T] = actions

            idx += T

    print(f"Wrote {output_path}  ({total_steps} steps, {num_episodes} episodes)")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_hdf5(path: Path) -> bool:
    """Read back the HDF5 file and check invariants.

    Returns True if all checks pass.
    """
    path = Path(path)
    ok = True

    with h5py.File(path, "r") as hf:
        for key in ("ep_len", "ep_offset", "pixels", "action"):
            if key not in hf:
                print(f"  [FAIL] missing dataset '{key}'")
                ok = False

        if not ok:
            return False

        ep_len = hf["ep_len"][:]
        ep_offset = hf["ep_offset"][:]
        pixels = hf["pixels"]
        action = hf["action"]

        total_steps_from_len = int(ep_len.sum())
        num_episodes = len(ep_len)

        # Shape checks
        if pixels.shape[0] != total_steps_from_len:
            print(
                f"  [FAIL] pixels has {pixels.shape[0]} rows but "
                f"sum(ep_len) = {total_steps_from_len}"
            )
            ok = False

        if action.shape[0] != total_steps_from_len:
            print(
                f"  [FAIL] action has {action.shape[0]} rows but "
                f"sum(ep_len) = {total_steps_from_len}"
            )
            ok = False

        if len(ep_offset) != num_episodes:
            print(
                f"  [FAIL] ep_offset length {len(ep_offset)} != "
                f"num_episodes {num_episodes}"
            )
            ok = False

        # Offset consistency
        expected_offsets = np.concatenate([[0], np.cumsum(ep_len[:-1])])
        if not np.array_equal(ep_offset, expected_offsets):
            print("  [FAIL] ep_offset is not consistent with ep_len")
            ok = False

        # NaN at episode boundaries
        for i in range(num_episodes):
            boundary = int(ep_offset[i] + ep_len[i] - 1)
            act_row = action[boundary]
            if not np.all(np.isnan(act_row)):
                print(
                    f"  [FAIL] action[{boundary}] (end of episode {i}) "
                    f"is not all-NaN: {act_row}"
                )
                ok = False

        # Summary
        H, W, C = pixels.shape[1:]
        action_dim = action.shape[1]

        print(f"  episodes:    {num_episodes}")
        print(f"  total_steps: {total_steps_from_len}")
        print(f"  pixels:      ({total_steps_from_len}, {H}, {W}, {C}) uint8")
        print(f"  action:      ({total_steps_from_len}, {action_dim}) float32")
        print(f"  ep_len  min/max/mean: "
              f"{ep_len.min()} / {ep_len.max()} / {ep_len.mean():.1f}")

        if ok:
            print("  [OK] All checks passed.")

    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NPZ delta-pose data + frames to HDF5 for stable_worldmodel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--npz-path",
        type=Path,
        required=True,
        help="Path to a .npz file or a directory of .npz files (one per episode).",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help=(
            "Directory of rendered frames organised by episode. "
            "Required when the NPZ does not contain pixel data."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output .h5 file.",
    )
    parser.add_argument(
        "--rotation-repr",
        choices=["euler", "quaternion", "6d", "axis_angle"],
        default="6d",
        help="Rotation representation for the action vector (default: 6d).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Resize frames to (image_size, image_size). Default: keep original.",
    )
    parser.add_argument(
        "--stablewm-home",
        type=Path,
        default=None,
        help="If set, place output under this directory (overrides --output dirname).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    output_path = args.output
    if args.stablewm_home is not None:
        output_path = args.stablewm_home / output_path.name

    print(f"Loading episodes from {args.npz_path} ...")
    episodes = load_episodes(
        npz_path=args.npz_path,
        frames_dir=args.frames_dir,
        rotation_repr=args.rotation_repr,
        image_size=args.image_size,
    )

    print(f"Found {len(episodes)} episode(s). Writing HDF5 ...")
    write_hdf5(episodes, output_path)

    print("Validating output ...")
    validate_hdf5(output_path)


if __name__ == "__main__":
    main()
