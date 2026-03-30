#!/usr/bin/env python3
"""Bayesian hyperparameter optimization for SIGReg weight using ClearML HPO.

Launches a ClearML HPO controller that sweeps over sigreg weight values
using Bayesian optimization (Optuna backend), running each trial as a
ClearML task on the specified queue.

Usage:
    python scripts/hparam_search.py \
        --queue default \
        --base-task-id <task_id_of_a_completed_training_run> \
        --max-trials 20
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian HPO for SIGReg weight via ClearML"
    )
    parser.add_argument(
        "--base-task-id",
        type=str,
        default=None,
        help="ClearML task ID of a completed training run to clone from. "
             "If not provided, creates a new base task.",
    )
    parser.add_argument("--queue", type=str, default="default",
                        help="ClearML queue for trial execution (default: default)")
    parser.add_argument("--max-trials", type=int, default=20,
                        help="Maximum number of HPO trials (default: 20)")
    parser.add_argument("--max-concurrent", type=int, default=2,
                        help="Max concurrent trials (default: 2)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Epochs per trial (default: 30)")
    parser.add_argument("--project", type=str, default="lewm",
                        help="ClearML project (default: lewm)")
    parser.add_argument("--dataset", type=str, default="prod_beta0_fs5",
                        help="Hydra data config name (default: prod_beta0_fs5)")
    args = parser.parse_args()

    try:
        from clearml import Task
        from clearml.automation import (
            HyperParameterOptimizer,
            UniformParameterRange,
            DiscreteParameterRange,
        )
        from clearml.automation.optuna import OptimizerOptuna
    except ImportError:
        print("ERROR: clearml and clearml[automation] required.\n"
              "  pip install 'clearml[automation]'", file=sys.stderr)
        sys.exit(1)

    # --- Create or reuse a base task ---
    if args.base_task_id:
        base_task_id = args.base_task_id
        print(f"Using existing base task: {base_task_id}")
    else:
        # Create a template task by running train.py briefly
        print("Creating base task template...")
        task = Task.init(
            project_name=args.project,
            task_name="lewm-hpo-base-template",
            task_type=Task.TaskTypes.training,
        )
        task.set_script(
            working_dir=".",
            entry_point="train.py",
        )
        # Set the default hydra overrides as task parameters
        task.set_parameters({
            "hydra_config/data": args.dataset,
            "hydra_config/wm/action_dim": 6,
            "hydra_config/trainer/max_epochs": args.epochs,
            "hydra_config/wandb/enabled": False,
            "hydra_config/clearml/enabled": True,
            "hydra_config/clearml/project": args.project,
            "hydra_config/loader/batch_size": 128,
            "hydra_config/loader/num_workers": 8,
            "hydra_config/seed": 42,
            "hydra_config/loss/sigreg/weight": 0.09,
            "hydra_config/output_model_name": "lewm_hpo_trial",
        })
        task.close()
        base_task_id = task.id
        print(f"Created base task: {base_task_id}")

    # --- Set up the HPO controller ---
    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=[
            # SIGReg weight: explore from very low to moderate
            UniformParameterRange(
                "hydra_config/loss/sigreg/weight",
                min_value=0.001,
                max_value=0.5,
                step_size=0.001,
            ),
            # Also sweep learning rate as it interacts with sigreg weight
            UniformParameterRange(
                "hydra_config/optimizer/lr",
                min_value=1e-5,
                max_value=5e-4,
                step_size=1e-5,
            ),
            # Sweep embed_dim
            DiscreteParameterRange(
                "hydra_config/wm/embed_dim",
                values=[128, 192, 256, 384],
            ),
        ],
        objective_metric_title="validate",
        objective_metric_series="pred_loss",
        objective_metric_sign="min",
        optimizer_class=OptimizerOptuna,
        max_number_of_concurrent_tasks=args.max_concurrent,
        execution_queue=args.queue,
        total_max_jobs=args.max_trials,
        min_iteration_per_job=100,  # minimum steps before early stopping
        max_iteration_per_job=None,
        project_name=f"{args.project}/hpo",
        task_name="lewm-sigreg-hpo",
    )

    # Add tags to the HPO task
    optimizer.set_report_period(1)

    print(f"\nStarting HPO:")
    print(f"  Base task: {base_task_id}")
    print(f"  Queue: {args.queue}")
    print(f"  Max trials: {args.max_trials}")
    print(f"  Max concurrent: {args.max_concurrent}")
    print(f"  Epochs per trial: {args.epochs}")
    print(f"  Parameters:")
    print(f"    sigreg.weight: [0.001, 0.5]")
    print(f"    optimizer.lr: [1e-5, 5e-4]")
    print(f"    embed_dim: [128, 192, 256, 384]")
    print(f"  Objective: minimize validate/pred_loss")
    print()

    optimizer.start()
    print("HPO started. Monitor at ClearML dashboard.")
    print("Waiting for completion (Ctrl+C to stop)...")

    try:
        optimizer.wait()
    except KeyboardInterrupt:
        print("\nStopping HPO...")

    # Print results
    top_experiments = optimizer.get_top_experiments(top_k=5)
    print("\n=== Top 5 Experiments ===")
    for i, exp in enumerate(top_experiments):
        params = exp.get_parameters()
        sigreg_w = params.get("hydra_config/loss/sigreg/weight", "?")
        lr = params.get("hydra_config/optimizer/lr", "?")
        embed = params.get("hydra_config/wm/embed_dim", "?")
        metrics = exp.get_last_scalar_metrics()
        val_pred = "?"
        if metrics and "validate" in metrics and "pred_loss" in metrics["validate"]:
            val_pred = f"{metrics['validate']['pred_loss']['last']:.6f}"
        print(f"  #{i+1}: sigreg={sigreg_w}, lr={lr}, embed_dim={embed} -> val_pred_loss={val_pred}")

    optimizer.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
