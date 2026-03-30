#!/usr/bin/env python3
"""Upload, download, and list HDF5 datasets via ClearML Dataset management.

This script provides versioned, remote-accessible storage for the LeWM
training pipeline's HDF5 files.  It wraps ``clearml.Dataset`` to handle
the upload/download/listing workflow with a simple CLI.

HDF5 files are expected to follow the ``stable_worldmodel`` schema::

    ep_len    -- int32,   (num_episodes,)
    ep_offset -- int32,   (num_episodes,)
    pixels    -- uint8,   (total_steps, H, W, 3)
    action    -- float32, (total_steps, action_dim)

Usage examples
--------------
Upload::

    python scripts/clearml_dataset.py upload \\
        --dataset-name prod_beta0 \\
        --project lewm/data \\
        --h5-path $STABLEWM_HOME/prod_beta0.h5 \\
        --tags synthetic delta-pose beta-0

Download by ID::

    python scripts/clearml_dataset.py download \\
        --dataset-id <id> \\
        --output-dir $STABLEWM_HOME

Download by name + project::

    python scripts/clearml_dataset.py download \\
        --dataset-name prod_beta0 \\
        --dataset-project lewm/data \\
        --output-dir $STABLEWM_HOME

List datasets in a project::

    python scripts/clearml_dataset.py list --project lewm/data
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

try:
    from clearml import Dataset
except ImportError:
    print(
        "Error: the 'clearml' package is not installed.\n"
        "Install it with:\n\n"
        "    pip install clearml\n\n"
        "Then run 'clearml-init' to configure your ClearML credentials.",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def cmd_upload(args: argparse.Namespace) -> None:
    """Create a ClearML Dataset, add the HDF5 file, upload, and finalize."""
    h5_path = Path(args.h5_path).expanduser().resolve()
    if not h5_path.is_file():
        print(f"Error: HDF5 file not found: {h5_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Creating ClearML dataset '{args.dataset_name}' in project '{args.project}' ...")
    dataset = Dataset.create(
        dataset_name=args.dataset_name,
        dataset_project=args.project,
    )

    if args.tags:
        dataset.add_tags(args.tags)

    print(f"Adding file: {h5_path}")
    dataset.add_files(str(h5_path))

    print("Uploading ...")
    dataset.upload()

    print("Finalizing ...")
    dataset.finalize()

    print(f"Done. Dataset ID: {dataset.id}")


def cmd_download(args: argparse.Namespace) -> None:
    """Download a ClearML Dataset and copy the HDF5 file to the output directory."""
    if args.dataset_id:
        print(f"Fetching dataset by ID: {args.dataset_id} ...")
        dataset = Dataset.get(dataset_id=args.dataset_id)
    elif args.dataset_name and args.dataset_project:
        print(
            f"Fetching dataset '{args.dataset_name}' "
            f"from project '{args.dataset_project}' ..."
        )
        dataset = Dataset.get(
            dataset_name=args.dataset_name,
            dataset_project=args.dataset_project,
        )
    else:
        print(
            "Error: provide either --dataset-id or both --dataset-name "
            "and --dataset-project.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Downloading local copy ...")
    local_path = Path(dataset.get_local_copy())

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # The local copy is a directory; find the HDF5 file(s) inside it.
    h5_files = list(local_path.rglob("*.h5")) + list(local_path.rglob("*.hdf5"))
    if not h5_files:
        print(
            f"Warning: no HDF5 files found in the downloaded dataset at {local_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    for h5_file in h5_files:
        dest = output_dir / h5_file.name
        print(f"Copying {h5_file.name} -> {dest}")
        shutil.copy2(str(h5_file), str(dest))

    print(f"Done. Files available in: {output_dir}")


def cmd_list(args: argparse.Namespace) -> None:
    """List all datasets in a ClearML project."""
    print(f"Listing datasets in project '{args.project}' ...\n")

    datasets = Dataset.list_datasets(dataset_project=args.project)

    if not datasets:
        print("No datasets found.")
        return

    # Print a formatted table
    print(f"{'ID':<36}  {'Name':<30}  {'Tags'}")
    print("-" * 90)
    for info in datasets:
        ds_id = info.get("id", "N/A")
        ds_name = info.get("name", "N/A")
        ds_tags = ", ".join(info.get("tags", []))
        print(f"{ds_id:<36}  {ds_name:<30}  {ds_tags}")

    print(f"\nTotal: {len(datasets)} dataset(s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with upload/download/list subcommands."""
    parser = argparse.ArgumentParser(
        description="Manage LeWM HDF5 datasets with ClearML Dataset versioning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- upload --
    up = subparsers.add_parser(
        "upload",
        help="Upload an HDF5 file as a versioned ClearML Dataset.",
    )
    up.add_argument(
        "--dataset-name",
        required=True,
        help="Name for the ClearML dataset.",
    )
    up.add_argument(
        "--project",
        required=True,
        help="ClearML project path (e.g. 'lewm/data').",
    )
    up.add_argument(
        "--h5-path",
        required=True,
        help="Path to the HDF5 file to upload.",
    )
    up.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Optional tags for the dataset (e.g. 'synthetic delta-pose').",
    )

    # -- download --
    dl = subparsers.add_parser(
        "download",
        help="Download a ClearML Dataset and copy HDF5 files to a local directory.",
    )
    dl.add_argument(
        "--dataset-id",
        default=None,
        help="ClearML dataset ID. Mutually exclusive with --dataset-name/--dataset-project.",
    )
    dl.add_argument(
        "--dataset-name",
        default=None,
        help="ClearML dataset name (requires --dataset-project).",
    )
    dl.add_argument(
        "--dataset-project",
        default=None,
        help="ClearML project path (requires --dataset-name).",
    )
    dl.add_argument(
        "--output-dir",
        required=True,
        help="Local directory to copy HDF5 files into.",
    )

    # -- list --
    ls = subparsers.add_parser(
        "list",
        help="List all datasets in a ClearML project.",
    )
    ls.add_argument(
        "--project",
        required=True,
        help="ClearML project path to list datasets from.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "upload": cmd_upload,
        "download": cmd_download,
        "list": cmd_list,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
