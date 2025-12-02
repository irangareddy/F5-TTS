#!/usr/bin/env python3
"""
Calculate training steps for voxe_data_v2 dataset before training starts.
Reads from manifest JSONL files and calculates steps based on params.yaml configuration.
"""

import json
import yaml
from pathlib import Path


def calculate_steps():
    # Load params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    training = params["training"]
    
    # Load dataset info from manifest
    manifest_file = Path("data/voxe_data_v2/manifests/train.jsonl")
    
    if not manifest_file.exists():
        print(f"Error: {manifest_file} not found")
        return
    
    print("Loading manifest file...")
    durations = []
    total_samples = 0
    
    with open(manifest_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    duration = data.get("duration_s", 0)
                    if duration > 0:
                        durations.append(duration)
                        total_samples += 1
                except json.JSONDecodeError:
                    continue
    
    if not durations:
        print("Error: No valid samples found in manifest")
        return
    
    total_duration_hours = sum(durations) / 3600
    total_duration_seconds = sum(durations)
    avg_duration = sum(durations) / len(durations)
    
    # Training parameters
    epochs = training["epochs"]
    batch_size_per_gpu = training["batch_size_per_gpu"]
    batch_size_type = training["batch_size_type"]
    max_samples = training["max_samples"]
    grad_accumulation_steps = training.get("grad_accumulation_steps", 1)
    num_warmup_updates = training.get("num_warmup_updates", 1500)
    early_stopping_patience = training.get("early_stopping_patience", 10)
    
    # Constants
    hop_length = 256
    sampling_rate = 24000
    gpus = 1  # Assuming single GPU
    
    # Calculate updates per epoch
    if batch_size_type == "frame":
        # Frame-based batching
        mini_batch_frames = batch_size_per_gpu * gpus * grad_accumulation_steps
        mini_batch_duration_seconds = mini_batch_frames * hop_length / sampling_rate
        mini_batch_duration_hours = mini_batch_duration_seconds / 3600
        updates_per_epoch = total_duration_hours / mini_batch_duration_hours
    else:
        # Sample-based batching
        batch_size = batch_size_per_gpu * gpus * grad_accumulation_steps
        updates_per_epoch = total_samples / batch_size
        mini_batch_frames = None
        mini_batch_duration_hours = None
    
    total_updates = updates_per_epoch * epochs
    total_steps = total_updates  # Steps = updates in this context
    
    # Calculate steps before training effectively starts
    # This includes warmup steps
    steps_before_training = num_warmup_updates
    
    # Print results
    print("=" * 70)
    print("Training Steps Calculation for voxe_data_v2")
    print("=" * 70)
    
    print(f"\nDataset Information:")
    print(f"  Manifest file: {manifest_file}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total duration: {total_duration_hours:.2f} hours ({total_duration_seconds:.2f} seconds)")
    print(f"  Average sample duration: {avg_duration:.2f} seconds")
    print(f"  Min duration: {min(durations):.2f} seconds")
    print(f"  Max duration: {max(durations):.2f} seconds")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Early stopping patience: {early_stopping_patience}")
    print(f"  Batch size per GPU: {batch_size_per_gpu} ({batch_size_type})")
    print(f"  Max samples per batch: {max_samples}")
    print(f"  Gradient accumulation steps: {grad_accumulation_steps}")
    print(f"  GPUs: {gpus}")
    print(f"  Warmup updates: {num_warmup_updates}")
    
    print(f"\nTraining Progress Calculation:")
    if batch_size_type == "frame":
        print(f"  Mini-batch frames: {mini_batch_frames:,}")
        print(f"  Mini-batch duration: {mini_batch_duration_seconds:.2f} seconds ({mini_batch_duration_hours:.4f} hours)")
    print(f"  Updates per epoch: {updates_per_epoch:.0f}")
    print(f"  Total updates (steps) for {epochs} epochs: {total_steps:.0f}")
    
    print(f"\nSteps Before Training Starts:")
    print(f"  Warmup steps: {num_warmup_updates}")
    print(f"  Steps before effective training: ~{steps_before_training}")
    print(f"  Warmup epochs: ~{num_warmup_updates / updates_per_epoch:.2f}")
    
    print(f"\nTraining Schedule:")
    print(f"  Total steps: {int(total_steps):,}")
    print(f"  Steps per epoch: {int(updates_per_epoch):,}")
    print(f"  Estimated steps before early stopping (if triggered):")
    print(f"    Minimum: {int(updates_per_epoch * early_stopping_patience):,} steps")
    print(f"    Maximum: {int(total_steps):,} steps")
    
    # Checkpoint saves
    save_per_updates = training.get("save_per_updates", 2000)
    num_checkpoints = int(total_updates / save_per_updates)
    print(f"\nCheckpoint Information:")
    print(f"  Checkpoints saved every: {save_per_updates} updates")
    print(f"  Estimated number of checkpoints: ~{num_checkpoints}")
    
    # Time estimate
    time_per_update_seconds = 0.6  # Conservative estimate for RTX 4090
    total_training_time_seconds = total_updates * time_per_update_seconds
    total_training_time_hours = total_training_time_seconds / 3600
    
    print(f"\nTime Estimate (RTX 4090):")
    print(f"  Time per update: ~{time_per_update_seconds:.2f} seconds")
    print(f"  Total training time: {total_training_time_hours:.2f} hours")
    print(f"  Total training time: {int(total_training_time_hours)}h {int((total_training_time_hours % 1) * 60)}m")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  You will have {int(total_steps):,} total steps for {epochs} epochs")
    print(f"  {int(updates_per_epoch):,} steps per epoch")
    print(f"  {num_warmup_updates} warmup steps before effective training begins")
    print("=" * 70)


if __name__ == "__main__":
    calculate_steps()

