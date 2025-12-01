#!/usr/bin/env python3
"""
W&B-wrapped training script for F5-TTS.
Initializes W&B run and calls finetune_cli.py with proper logging.
"""

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Training will run without W&B logging.")


def load_params(params_file: str = "params.yaml") -> dict:
    """Load params.yaml file."""
    params_path = Path(params_file)
    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found at {params_path}")

    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    return params


def setup_wandb(params: dict) -> None:
    """Initialize Weights & Biases run."""
    if not WANDB_AVAILABLE:
        print("[WARN] wandb not available, skipping W&B logging")
        return

    training = params["training"]
    wandb_config = params.get("wandb", {})

    if not wandb_config.get("enabled", False):
        print("[INFO] W&B disabled in config")
        return

    # Get W&B credentials from environment or config
    api_key = os.environ.get("WANDB_API_KEY")
    entity = wandb_config.get("entity", "")
    project = wandb_config.get("project", "")

    if not api_key:
        print("[WARN] WANDB_API_KEY not set, W&B logging may fail")

    if not entity or not project:
        print("[WARN] W&B entity or project not configured")
        return

    # Set up W&B environment
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_ENTITY"] = entity
    os.environ["WANDB_PROJECT"] = project

    # Initialize W&B run
    config = {
        "exp_name": training.get("exp_name"),
        "dataset_name": training.get("dataset_name"),
        "learning_rate": training.get("learning_rate"),
        "batch_size_per_gpu": training.get("batch_size_per_gpu"),
        "batch_size_type": training.get("batch_size_type"),
        "max_samples": training.get("max_samples"),
        "epochs": training.get("epochs"),
        "num_warmup_updates": training.get("num_warmup_updates"),
        "save_per_updates": training.get("save_per_updates"),
        "keep_last_n_checkpoints": training.get("keep_last_n_checkpoints"),
        "finetune": training.get("finetune", False),
        "tokenizer": training.get("tokenizer"),
        "log_samples": training.get("log_samples", False),
        "logger": training.get("logger"),
    }

    try:
        run = wandb.init(
            entity=entity,
            project=project,
            name=wandb_config.get("name", "f5tts_training"),
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get("notes", ""),
            config=config,
            save_code=wandb_config.get("save_code", False),
            mode="online"
        )
        print(f"[WANDB] Initialized run: {run.name}")
        print(f"[WANDB] Dashboard: https://wandb.ai/{entity}/{project}/runs/{run.id}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize W&B: {e}")
        print("[INFO] Continuing training without W&B logging")


def build_finetune_command(params: dict, extra_args: list = None) -> list:
    """Build finetune_cli.py command."""
    training = params["training"]

    cmd = [
        "python", "src/f5_tts/train/finetune_cli.py",
        "--exp_name", training["exp_name"],
        "--dataset_name", training["dataset_name"],
        "--learning_rate", str(training["learning_rate"]),
        "--batch_size_per_gpu", str(training["batch_size_per_gpu"]),
        "--batch_size_type", training["batch_size_type"],
        "--max_samples", str(training["max_samples"]),
        "--epochs", str(training["epochs"]),
        "--num_warmup_updates", str(training["num_warmup_updates"]),
        "--save_per_updates", str(training["save_per_updates"]),
        "--keep_last_n_checkpoints", str(training["keep_last_n_checkpoints"]),
    ]

    if training.get("finetune", False):
        cmd.append("--finetune")

    if training.get("pretrain"):
        cmd.extend(["--pretrain", training["pretrain"]])

    cmd.extend(["--tokenizer", training["tokenizer"]])

    if training.get("tokenizer_path"):
        cmd.extend(["--tokenizer_path", training["tokenizer_path"]])

    if training.get("log_samples", False):
        cmd.append("--log_samples")

    if training.get("logger"):
        cmd.extend(["--logger", training["logger"]])

    if training.get("bnb_optimizer", False):
        cmd.append("--bnb_optimizer")

    # Add any extra arguments
    if extra_args:
        cmd.extend(extra_args)

    return cmd


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train F5-TTS with W&B logging")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args, unknown = parser.parse_known_args()

    print("=" * 70)
    print("F5-TTS Training with Weights & Biases")
    print("=" * 70)

    # Load configuration
    print(f"\n[INFO] Loading config from: {args.params}")
    params = load_params(args.params)

    # Setup W&B if enabled
    if not args.no_wandb:
        print("\n[INFO] Initializing Weights & Biases...")
        setup_wandb(params)

    # Build and run training command
    print("\n[INFO] Building training command...")
    cmd = build_finetune_command(params, extra_args=unknown)

    print(f"\n[INFO] Running: {' '.join(cmd)}\n")
    print("=" * 70)

    # Run training
    result = subprocess.run(cmd)

    # Finalize W&B run
    if not args.no_wandb and WANDB_AVAILABLE:
        try:
            wandb.finish()
            print("\n[WANDB] Run finished and synced")
        except Exception as e:
            print(f"\n[WARN] Error finishing W&B run: {e}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
