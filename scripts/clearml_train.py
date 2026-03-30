#!/usr/bin/env python3
"""ClearML remote training launcher for LeWM.

Creates a ClearML Task from the LeWM training pipeline and optionally
enqueues it for remote execution on a ClearML Agent.

Usage
-----
Local tracking (runs training in this process, logs to ClearML):

    python scripts/clearml_train.py \\
        --project lewm \\
        --task-name "vjepa-frozen-beta0" \\
        --data prod_beta0 \\
        --overrides encoder_type=vjepa encoder_frozen=true

Remote execution (creates task, enqueues to agent, exits):

    python scripts/clearml_train.py \\
        --project lewm \\
        --task-name "vjepa-frozen-beta0" \\
        --queue gpu \\
        --dataset-id <clearml-dataset-id> \\
        --vjepa-checkpoint-id <clearml-artifact-id-or-path> \\
        --data prod_beta0 \\
        --overrides encoder_type=vjepa encoder_frozen=true patch_size=16 \\
                     wm.action_dim=9 trainer.max_epochs=20
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _looks_like_clearml_id(value: str) -> bool:
    """Return True if *value* resembles a ClearML hex ID (32-char hex)."""
    return bool(re.fullmatch(r"[0-9a-f]{32}", value))


def _load_hydra_config(data_name: str | None, overrides: list[str]):
    """Compose the Hydra config without launching the @hydra.main decorator.

    Returns an OmegaConf DictConfig (or plain dict when OmegaConf is missing).
    """
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError:
        # Hydra/OmegaConf not installed -- return overrides as a flat dict so
        # that the caller can still connect *something* to the ClearML task.
        cfg = {}
        for ov in overrides:
            if "=" in ov:
                k, v = ov.split("=", 1)
                cfg[k] = v
        return cfg

    config_dir = str(REPO_ROOT / "config" / "train")

    hydra_overrides = list(overrides)
    if data_name:
        hydra_overrides.insert(0, f"data={data_name}")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="lewm", overrides=hydra_overrides)

    return OmegaConf.to_container(cfg, resolve=True)


def _resolve_dataset(task, dataset_id: str | None):
    """Download a ClearML Dataset and set STABLEWM_HOME."""
    if not dataset_id:
        return

    from clearml import Dataset

    print(f"[clearml_train] Downloading ClearML Dataset {dataset_id} ...")
    dataset = Dataset.get(dataset_id=dataset_id)
    local_path = dataset.get_local_copy()
    os.environ["STABLEWM_HOME"] = local_path
    print(f"[clearml_train] STABLEWM_HOME set to {local_path}")
    task.set_parameter("General/stablewm_home", local_path)


def _resolve_vjepa_checkpoint(
    task, checkpoint_id: str | None
) -> str | None:
    """Return a local path to the VJEPA encoder checkpoint.

    If *checkpoint_id* looks like a 32-char hex ClearML ID, download it as an
    artifact from a ClearML task.  Otherwise treat it as a local/remote path.
    """
    if not checkpoint_id:
        return None

    if _looks_like_clearml_id(checkpoint_id):
        from clearml import Model

        print(
            f"[clearml_train] Downloading VJEPA checkpoint model {checkpoint_id} ..."
        )
        model = Model(model_id=checkpoint_id)
        local_path = model.get_local_copy()
        print(f"[clearml_train] VJEPA checkpoint at {local_path}")
        return local_path

    # Assume it is already a usable path (local or NFS mount).
    return checkpoint_id


def _find_latest_checkpoint(run_dir: Path, model_name: str) -> Path | None:
    """Find the latest epoch checkpoint saved by ModelObjectCallBack."""
    pattern = f"{model_name}_epoch_*_object.ckpt"
    ckpts = sorted(run_dir.glob(pattern), key=os.path.getmtime)
    if ckpts:
        return ckpts[-1]

    # Fallback: lightning weights checkpoint
    weights = run_dir / f"{model_name}_weights.ckpt"
    if weights.exists():
        return weights

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch LeWM training with ClearML tracking / remote execution.",
    )
    parser.add_argument(
        "--project",
        default="lewm",
        help="ClearML project name (default: lewm).",
    )
    parser.add_argument(
        "--task-name",
        default="lewm-train",
        help="ClearML task name (default: lewm-train).",
    )
    parser.add_argument(
        "--queue",
        default=None,
        help="ClearML queue for remote execution.  If omitted, train locally.",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="ClearML Dataset ID.  If set, dataset is downloaded and STABLEWM_HOME is configured.",
    )
    parser.add_argument(
        "--vjepa-checkpoint-id",
        default=None,
        help="ClearML model ID or local path for the VJEPA encoder checkpoint.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Hydra data config name (e.g. prod_beta0, pusht, dmc).  Maps to data=<name> override.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Extra Hydra overrides passed to train.py (key=value pairs).",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=[],
        help="Tags to apply to the ClearML task.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ------------------------------------------------------------------
    # 0.  Import ClearML
    # ------------------------------------------------------------------
    try:
        from clearml import Task
    except ImportError:
        print(
            "ERROR: clearml is not installed.  Install it with:\n"
            "    pip install clearml\n"
            "Then run `clearml-init` to configure your credentials.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1.  Create ClearML Task
    # ------------------------------------------------------------------
    task = Task.init(
        project_name=args.project,
        task_name=args.task_name,
        task_type=Task.TaskTypes.training,
    )

    if args.tags:
        task.add_tags(args.tags)

    # ------------------------------------------------------------------
    # 2.  Load Hydra config and connect to the task
    # ------------------------------------------------------------------
    cfg_dict = _load_hydra_config(args.data, args.overrides)
    task.connect(cfg_dict, name="hydra_config")

    # ------------------------------------------------------------------
    # 3.  Handle dataset download
    # ------------------------------------------------------------------
    _resolve_dataset(task, args.dataset_id)

    # ------------------------------------------------------------------
    # 4.  Handle VJEPA checkpoint
    # ------------------------------------------------------------------
    vjepa_path = _resolve_vjepa_checkpoint(task, args.vjepa_checkpoint_id)

    # ------------------------------------------------------------------
    # 5.  Remote execution gate
    #     If --queue is specified, Task.execute_remotely() will serialize
    #     the current state and enqueue the task.  On the agent, execution
    #     continues from the line *after* execute_remotely().
    # ------------------------------------------------------------------
    if args.queue:
        print(
            f"[clearml_train] Enqueuing task to queue '{args.queue}' ...\n"
            f"  Task ID  : {task.id}\n"
            f"  Task URL : {task.get_output_log_web_page()}"
        )
        task.execute_remotely(queue_name=args.queue)
        # --- everything below runs on the remote agent ---

    # ------------------------------------------------------------------
    # 6.  Build train.py command
    # ------------------------------------------------------------------
    cmd = [sys.executable, str(REPO_ROOT / "train.py")]

    if args.data:
        cmd.append(f"data={args.data}")

    for ov in args.overrides:
        cmd.append(ov)

    if vjepa_path:
        cmd.append(f"vjepa_checkpoint={vjepa_path}")

    # Disable wandb when running under ClearML to avoid double-logging.
    cmd.append("wandb.enabled=false")

    print(f"[clearml_train] Running: {' '.join(cmd)}")

    # ------------------------------------------------------------------
    # 7.  Execute training
    # ------------------------------------------------------------------
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))

    if result.returncode != 0:
        print(
            f"[clearml_train] train.py exited with code {result.returncode}",
            file=sys.stderr,
        )
        task.mark_failed(
            status_reason=f"train.py exited with code {result.returncode}",
        )
        sys.exit(result.returncode)

    # ------------------------------------------------------------------
    # 8.  Register model output
    # ------------------------------------------------------------------
    try:
        import stable_worldmodel as swm

        cache_dir = Path(swm.data.utils.get_cache_dir())
    except Exception:
        cache_dir = Path(os.environ.get("STABLEWM_HOME", "."))

    output_model_name = cfg_dict.get("output_model_name", "lewm")

    # The run dir inside the cache is named after Hydra's job id; scan for
    # the most recently modified checkpoint across all subdirectories.
    candidates: list[Path] = []
    for sub in sorted(cache_dir.iterdir()) if cache_dir.is_dir() else []:
        if sub.is_dir():
            ckpt = _find_latest_checkpoint(sub, output_model_name)
            if ckpt:
                candidates.append(ckpt)

    # Also check cache_dir root in case subdir is empty.
    root_ckpt = _find_latest_checkpoint(cache_dir, output_model_name)
    if root_ckpt:
        candidates.append(root_ckpt)

    if candidates:
        best = max(candidates, key=os.path.getmtime)
        print(f"[clearml_train] Uploading model artifact: {best}")
        task.update_output_model(
            model_path=str(best),
            model_name=f"{args.task_name}-checkpoint",
        )
    else:
        print("[clearml_train] WARNING: No checkpoint found to upload.")

    print("[clearml_train] Done.")


if __name__ == "__main__":
    main()
