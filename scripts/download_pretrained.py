#!/usr/bin/env python3
"""Download pretrained LeWM checkpoints from HuggingFace."""

import os
from huggingface_hub import hf_hub_download

CACHE = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))

MODELS = {
    "pusht/lewm": "quentinll/lewm-pusht",
    "tworoom/lewm": "quentinll/lewm-tworooms",
    "cube/lewm": "quentinll/lewm-cube",
    "reacher/lewm": "quentinll/lewm-reacher",
}


def main():
    for policy_path, repo_id in MODELS.items():
        out_dir = os.path.join(CACHE, os.path.dirname(policy_path))
        os.makedirs(out_dir, exist_ok=True)

        weights_file = os.path.join(out_dir, "weights.pt")
        if os.path.exists(weights_file):
            print(f"  {policy_path}: already exists")
            continue

        print(f"  {policy_path}: downloading from {repo_id}...")
        try:
            hf_hub_download(repo_id, "weights.pt", local_dir=out_dir)
            hf_hub_download(repo_id, "config.json", local_dir=out_dir)
            print(f"  {policy_path}: done")
        except Exception as e:
            print(f"  {policy_path}: FAILED — {e}")

    print("\nPretrained downloads complete.")


if __name__ == "__main__":
    main()
