#!/usr/bin/env python3
"""
Calculate training time estimate based on params.yaml configuration
"""

import json
import yaml
from pathlib import Path


def calculate_training_time():
    # Load params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    training = params["training"]
    
    # Load dataset info
    dataset_name = training["dataset_name"]
    duration_file = Path(f"data/{dataset_name}_char/duration.json")
    
    if not duration_file.exists():
        print(f"Error: {duration_file} not found")
        return
    
    with open(duration_file, "r") as f:
        data = json.load(f)
    
    durations = data["duration"]
    total_samples = len(durations)
    total_duration_hours = sum(durations) / 3600
    
    # Check manifest for full dataset size
    manifest_file = Path(f"data/{dataset_name}/manifests/train.jsonl")
    manifest_samples = None
    if manifest_file.exists():
        with open(manifest_file, "r") as f:
            manifest_samples = sum(1 for _ in f)
    
    # Training parameters
    epochs = training["epochs"]
    batch_size_per_gpu = training["batch_size_per_gpu"]
    batch_size_type = training["batch_size_type"]
    max_samples = training["max_samples"]
    grad_accumulation_steps = training.get("grad_accumulation_steps", 1)
    
    # Constants
    hop_length = 256
    sampling_rate = 24000
    gpus = 1  # Assuming single GPU (RTX 4090)
    
    # Calculate updates per epoch
    if batch_size_type == "frame":
        # Frame-based batching
        mini_batch_frames = batch_size_per_gpu * gpus * grad_accumulation_steps
        mini_batch_duration_hours = mini_batch_frames * hop_length / sampling_rate / 3600
        updates_per_epoch = total_duration_hours / mini_batch_duration_hours
    else:
        # Sample-based batching
        batch_size = batch_size_per_gpu * gpus * grad_accumulation_steps
        updates_per_epoch = total_samples / batch_size
    
    total_updates = updates_per_epoch * epochs
    
    # Estimate time per update (conservative estimate based on RTX 4090)
    # RTX 4090 can process ~0.5-1 second per update for F5TTS_v1_Base
    # Smaller batches = faster updates, larger batches = slower updates
    # With batch_size_per_gpu=400 frames, estimate ~0.5-0.8 seconds per update
    time_per_update_seconds = 0.6  # Optimistic estimate for RTX 4090
    
    total_training_time_seconds = total_updates * time_per_update_seconds
    total_training_time_hours = total_training_time_seconds / 3600
    total_training_time_minutes = total_training_time_seconds / 60
    
    # Print results
    print("=" * 70)
    print("Training Time Estimate")
    print("=" * 70)
    print(f"\nDataset Information:")
    print(f"  Total samples (in Arrow format): {total_samples:,}")
    if manifest_samples:
        print(f"  Total samples (in manifest): {manifest_samples:,}")
        if manifest_samples != total_samples:
            print(f"  [WARN] Note: Arrow dataset has {total_samples} samples, but manifest has {manifest_samples}")
            print(f"    Training will use: {total_samples} samples")
    print(f"  Total duration: {total_duration_hours:.2f} hours ({sum(durations) / 3600:.2f} hours)")
    print(f"  Average sample duration: {sum(durations) / len(durations):.2f} seconds")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size per GPU: {batch_size_per_gpu} ({batch_size_type})")
    print(f"  Max samples per batch: {max_samples}")
    print(f"  Gradient accumulation steps: {grad_accumulation_steps}")
    print(f"  GPUs: {gpus}")
    
    print(f"\nTraining Progress:")
    print(f"  Updates per epoch: {updates_per_epoch:.0f}")
    print(f"  Total updates: {total_updates:.0f}")
    
    if batch_size_type == "frame":
        print(f"  Mini-batch frames: {mini_batch_frames:,}")
        print(f"  Mini-batch duration: {mini_batch_duration_hours:.3f} hours")
    
    print(f"\nTime Estimate (RTX 4090):")
    print(f"  Time per update: ~{time_per_update_seconds:.2f} seconds")
    print(f"  Total training time: {total_training_time_hours:.2f} hours")
    print(f"  Total training time: {total_training_time_minutes:.0f} minutes")
    if total_training_time_hours < 1:
        print(f"  Total training time: {int(total_training_time_minutes)} minutes")
    else:
        print(f"  Total training time: {int(total_training_time_hours)}h {int((total_training_time_hours % 1) * 60)}m")
    
    # Checkpoint saves
    save_per_updates = training.get("save_per_updates", 200)
    num_checkpoints = int(total_updates / save_per_updates)
    print(f"\nCheckpoint Information:")
    print(f"  Checkpoints will be saved every {save_per_updates} updates")
    print(f"  Estimated number of checkpoints: ~{num_checkpoints}")
    
    # Warmup info
    num_warmup_updates = training.get("num_warmup_updates", 300)
    warmup_epochs = num_warmup_updates / updates_per_epoch
    print(f"\nWarmup Information:")
    print(f"  Warmup updates: {num_warmup_updates}")
    print(f"  Warmup epochs: ~{warmup_epochs:.2f}")
    
    print("\n" + "=" * 70)
    print("Note: This is an estimate. Actual time may vary based on:")
    print("  - GPU utilization")
    print("  - Data loading speed")
    print("  - Model checkpoint saving overhead")
    print("  - System load")
    print("  - WandB logging overhead")
    print("=" * 70)


if __name__ == "__main__":
    calculate_training_time()

