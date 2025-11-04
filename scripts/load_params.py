#!/usr/bin/env python3
"""
Load and validate params.yaml configuration file.
Used by Taskfile tasks to read configuration and build commands.
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path


def load_params(params_file: str = "params.yaml") -> dict:
    """Load params.yaml file."""
    params_path = Path(params_file)
    if not params_path.exists():
        print(f"Error: {params_file} not found")
        sys.exit(1)
    
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    
    return params


def validate_params(params: dict) -> bool:
    """Validate required parameters."""
    required_sections = ["training", "inference", "docker", "paths"]
    missing = [s for s in required_sections if s not in params]
    
    if missing:
        print(f"Error: Missing required sections: {missing}")
        return False
    
    # Validate training params
    training = params["training"]
    if "exp_name" not in training:
        print("Error: training.exp_name is required")
        return False
    if training["exp_name"] not in ["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base"]:
        print(f"Error: Invalid exp_name: {training['exp_name']}")
        return False
    
    if "dataset_name" not in training:
        print("Error: training.dataset_name is required")
        return False
    
    # Validate inference params
    inference = params["inference"]
    if "model" not in inference:
        print("Error: inference.model is required")
        return False
    
    return True


def check_config(params: dict):
    """Check and display current configuration."""
    print("=" * 70)
    print("Configuration Check")
    print("=" * 70)
    
    if not validate_params(params):
        sys.exit(1)
    
    print("\nTraining Configuration:")
    print("-" * 70)
    training = params["training"]
    for key, value in training.items():
        print(f"  {key:<30s} = {value}")
    
    print("\nInference Configuration:")
    print("-" * 70)
    inference = params["inference"]
    for key, value in inference.items():
        if value:  # Only show non-empty values
            print(f"  {key:<30s} = {value}")
    
    print("\nDocker Configuration:")
    print("-" * 70)
    docker = params["docker"]
    for key, value in docker.items():
        print(f"  {key:<30s} = {value}")
    
    print("\nPath Configuration:")
    print("-" * 70)
    paths = params["paths"]
    for key, value in paths.items():
        print(f"  {key:<30s} = {value}")
    
    # Validate paths exist
    print("\nPath Validation:")
    print("-" * 70)
    local_paths = {
        "Data": paths.get("data", "data"),
        "Checkpoints": paths.get("ckpts", "ckpts"),
        "Source": paths.get("src", "src"),
        "Tests": paths.get("tests", "tests"),
    }
    
    # Optional paths (warn but don't fail)
    optional_paths = {
        "Notebooks": paths.get("notebooks", "notebooks"),
    }
    
    all_ok = True
    for name, path in local_paths.items():
        if Path(path).exists():
            print(f"  [OK] {name:<20s} = {path}")
        else:
            print(f"  [FAIL] {name:<20s} = {path} (not found)")
            all_ok = False
    
    # Check optional paths
    for name, path in optional_paths.items():
        if Path(path).exists():
            print(f"  [OK] {name:<20s} = {path}")
        else:
            print(f"  [INFO] {name:<20s} = {path} (optional, not found)")
    
    print("=" * 70)
    if all_ok:
        print("[OK] All paths validated successfully!")
    else:
        print("[FAIL] Some paths are missing. Check paths above.")
    
    return all_ok


def build_training_command(params: dict, dry_run: bool = False, overrides: list = None) -> list:
    """Build training command from params."""
    training = params["training"]
    docker = params["docker"]
    paths = params["paths"]
    
    # Get current working directory (works on both Windows and Unix)
    import os
    pwd = os.getcwd().replace("\\", "/")  # Normalize Windows paths
    
    # Build command arguments
    cmd = [
        "docker", "run", "--rm", "--gpus", docker["gpus"],
        "--shm-size", docker["shm_size"],
        "-v", f"{pwd}/{paths['data']}:{docker['workdir']}/data:ro",
        "-v", f"{pwd}/{paths['ckpts']}:{docker['workdir']}/ckpts",
        "-v", f"{pwd}/{paths['src']}:{docker['workdir']}/src",
        "-w", docker["workdir"],
        docker["image"],
        "python", f"{docker['workdir']}/src/f5_tts/train/finetune_cli.py",
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
    
    # Apply overrides
    if overrides:
        cmd.extend(overrides)
    
    return cmd


def build_inference_command(params: dict, dry_run: bool = False, overrides: list = None) -> list:
    """Build inference command from params."""
    inference = params["inference"]
    docker = params["docker"]
    paths = params["paths"]
    
    # Get current working directory (works on both Windows and Unix)
    import os
    pwd = os.getcwd().replace("\\", "/")  # Normalize Windows paths
    
    # Build command arguments
    cmd = [
        "docker", "run", "--rm", "--gpus", docker["gpus"],
        "-v", f"{pwd}/{paths['data']}:{docker['workdir']}/data:ro",
        "-v", f"{pwd}/{paths['ckpts']}:{docker['workdir']}/ckpts:ro",
        "-v", f"{pwd}/{paths['src']}:{docker['workdir']}/src",
        "-v", f"{pwd}/{paths['tests']}:{docker['workdir']}/tests",
        "-w", docker["workdir"],
        docker["image"],
        "python", f"{docker['workdir']}/src/f5_tts/infer/infer_cli.py",
        "--model", inference["model"],
        "--ckpt_file", inference["ckpt_file"],
        "--vocab_file", inference["vocab_file"],
        "--ref_audio", inference["ref_audio"],
        "--ref_text", inference["ref_text"],
        "--gen_text", inference["gen_text"],
        "--output_dir", inference["output_dir"],
        "--output_file", inference["output_file"],
    ]
    
    if inference.get("gen_file"):
        cmd.extend(["--gen_file", inference["gen_file"]])
    
    if inference.get("remove_silence", False):
        cmd.append("--remove_silence")
    
    if inference.get("save_chunk", False):
        cmd.append("--save_chunk")
    
    # Apply overrides
    if overrides:
        cmd.extend(overrides)
    
    return cmd


def verify_setup(params: dict):
    """Verify setup (GPU, Docker, data)."""
    print("=" * 70)
    print("Setup Verification")
    print("=" * 70)
    
    import subprocess
    
    # Check Docker
    print("\n1. Checking Docker...")
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   [OK] Docker installed: {result.stdout.strip()}")
        else:
            print("   [FAIL] Docker not found")
            return False
    except FileNotFoundError:
        print("   [FAIL] Docker not found")
        return False
    
    # Check Docker image
    print("\n2. Checking Docker image...")
    docker = params["docker"]
    result = subprocess.run(["docker", "images", "-q", docker["image"]], capture_output=True, text=True)
    if result.stdout.strip():
        print(f"   [OK] Docker image exists: {docker['image']}")
    else:
        print(f"   [FAIL] Docker image not found: {docker['image']}")
        print(f"   Build it with: docker build -t {docker['image']} -f Dockerfile .")
        return False
    
    # Check GPU support
    print("\n3. Checking GPU support...")
    result = subprocess.run(["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.4.0-base-ubuntu22.04", "nvidia-smi"], 
                          capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        print("   [OK] GPU support available in Docker")
    else:
        print("   [WARN] GPU support check failed (may still work)")
    
    # Check data paths
    print("\n4. Checking data paths...")
    paths = params["paths"]
    data_path = Path(paths["data"])
    if data_path.exists():
        print(f"   [OK] Data directory exists: {data_path}")
    else:
        print(f"   [FAIL] Data directory not found: {data_path}")
        return False
    
    # Check dataset
    training = params["training"]
    dataset_path = data_path / training["dataset_name"]
    if dataset_path.exists():
        print(f"   [OK] Dataset directory exists: {dataset_path}")
    else:
        print(f"   [FAIL] Dataset directory not found: {dataset_path}")
        return False
    
    print("\n" + "=" * 70)
    print("[OK] Setup verification complete!")
    return True


def clean_checkpoints(params: dict, checkpoints_only: bool = False):
    """Clean checkpoints and/or outputs."""
    paths = params["paths"]
    
    if checkpoints_only:
        print("Cleaning checkpoints only...")
        ckpt_path = Path(paths["ckpts"])
        if ckpt_path.exists():
            # Remove checkpoint files but keep directory structure
            for ckpt_file in ckpt_path.rglob("*.pt"):
                ckpt_file.unlink()
                print(f"  Removed: {ckpt_file}")
            for ckpt_file in ckpt_path.rglob("*.safetensors"):
                ckpt_file.unlink()
                print(f"  Removed: {ckpt_file}")
    else:
        print("Cleaning checkpoints and outputs...")
        # Clean checkpoints
        ckpt_path = Path(paths["ckpts"])
        if ckpt_path.exists():
            for ckpt_file in ckpt_path.rglob("*.pt"):
                ckpt_file.unlink()
            for ckpt_file in ckpt_path.rglob("*.safetensors"):
                ckpt_file.unlink()
        
        # Clean outputs
        test_path = Path(paths["tests"])
        if test_path.exists():
            for output_file in test_path.glob("*.wav"):
                output_file.unlink()
                print(f"  Removed: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Load and validate params.yaml")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check configuration")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Build training command")
    train_parser.add_argument("--dry-run", action="store_true", help="Show command without executing")
    
    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Build inference command")
    infer_parser.add_argument("--dry-run", action="store_true", help="Show command without executing")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify setup")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean checkpoints/outputs")
    clean_parser.add_argument("--confirm", action="store_true", help="Confirm deletion")
    clean_parser.add_argument("--checkpoints-only", action="store_true", help="Clean only checkpoints")
    
    args, unknown = parser.parse_known_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Load params
    params = load_params()
    
    # Execute command
    if args.command == "check":
        success = check_config(params)
        sys.exit(0 if success else 1)
    
    elif args.command == "train":
        overrides = unknown if unknown else []
        cmd = build_training_command(params, dry_run=args.dry_run, overrides=overrides)
        if args.dry_run:
            print("Training command (dry-run):")
            print(" ".join(cmd))
        else:
            print("Running training command...")
            import subprocess
            sys.exit(subprocess.run(cmd).returncode)
    
    elif args.command == "infer":
        overrides = unknown if unknown else []
        cmd = build_inference_command(params, dry_run=args.dry_run, overrides=overrides)
        if args.dry_run:
            print("Inference command (dry-run):")
            print(" ".join(cmd))
        else:
            print("Running inference command...")
            import subprocess
            sys.exit(subprocess.run(cmd).returncode)
    
    elif args.command == "verify":
        success = verify_setup(params)
        sys.exit(0 if success else 1)
    
    elif args.command == "clean":
        if not args.confirm:
            print("Error: --confirm flag required for safety")
            sys.exit(1)
        clean_checkpoints(params, checkpoints_only=args.checkpoints_only)
        print("Cleanup complete!")


if __name__ == "__main__":
    main()

