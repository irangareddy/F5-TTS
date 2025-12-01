#!/usr/bin/env python3
"""
Training script that uses params.yaml configuration and supports early stopping with validation.
"""

import os
import yaml
from importlib.resources import files

from f5_tts.model import CFM, DiT, Trainer
from f5_tts.model.dataset import load_dataset, ManifestDataset
from f5_tts.model.utils import get_tokenizer


def load_config(config_path="params.yaml"):
    """Load configuration from params.yaml"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Change to project root
    os.chdir(str(files("f5_tts").joinpath("../..")))

    # Load configuration
    config = load_config("params.yaml")

    # Extract configuration sections
    training_cfg = config["training"]
    inference_cfg = config["inference"]
    wandb_cfg = config.get("wandb", {})
    docker_cfg = config.get("docker", {})
    paths_cfg = config.get("paths", {})

    # Model and dataset settings
    exp_name = training_cfg["exp_name"]
    dataset_name = training_cfg["dataset_name"]
    tokenizer_type = training_cfg.get("tokenizer", "char")
    tokenizer_path = training_cfg.get("tokenizer_path", None)

    # Training hyperparameters
    epochs = training_cfg["epochs"]
    learning_rate = training_cfg["learning_rate"]
    batch_size_per_gpu = training_cfg["batch_size_per_gpu"]
    batch_size_type = training_cfg.get("batch_size_type", "frame")
    max_samples = training_cfg.get("max_samples", 16)
    grad_accumulation_steps = training_cfg.get("grad_accumulation_steps", 1)
    max_grad_norm = training_cfg.get("max_grad_norm", 1.0)

    # Checkpoint saving
    save_per_updates = training_cfg.get("save_per_updates", 100)
    keep_last_n_checkpoints = training_cfg.get("keep_last_n_checkpoints", 2)
    last_per_updates = training_cfg.get("last_per_updates", 5000)

    # Warmup
    num_warmup_updates = training_cfg.get("num_warmup_updates", 100)

    # Validation and early stopping
    validation_split = training_cfg.get("validation_split", None)
    validation_interval = training_cfg.get("validation_interval", 1)
    early_stopping_patience = training_cfg.get("early_stopping_patience", 3)
    early_stopping_threshold = training_cfg.get("early_stopping_threshold", 0.001)

    # Logging
    log_samples = training_cfg.get("log_samples", True)
    logger_type = training_cfg.get("logger", "wandb")

    # Optimizer settings
    bnb_optimizer = training_cfg.get("bnb_optimizer", False)

    # Mel spec settings
    mel_spec_type = inference_cfg.get("model_cfg", {}).get("mel_spec", {}).get("mel_spec_type", "vocos")

    mel_spec_kwargs = dict(
        target_sample_rate=24000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type=mel_spec_type,
    )

    # Get tokenizer
    if tokenizer_type != "custom":
        tokenizer_name = dataset_name
    else:
        tokenizer_name = tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_name, tokenizer_type)

    # Create model
    print(f"Creating model: {exp_name}")
    if exp_name == "F5TTS_v1_Base":
        from f5_tts.model.cfm import CFM
        model = CFM(
            transformer=DiT(
                dim=1024,
                depth=22,
                heads=16,
                ff_mult=2,
                text_dim=512,
                text_num_embeds=vocab_size,
                mel_dim=100,
                conv_layers=4,
            ),
            mel_spec_kwargs=mel_spec_kwargs,
            vocab_char_map=vocab_char_map,
        )
    else:
        raise ValueError(f"Unknown model: {exp_name}")

    # Checkpoint path
    checkpoint_path = f"ckpts/{exp_name}_{mel_spec_type}_{tokenizer_type}_{dataset_name}"
    os.makedirs(checkpoint_path, exist_ok=True)

    # WandB configuration
    wandb_project = wandb_cfg.get("project", "f5-tts")
    wandb_entity = wandb_cfg.get("entity", None)
    wandb_run_name = wandb_cfg.get("name", f"{exp_name}_{dataset_name}")

    # Create trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        epochs=epochs,
        learning_rate=learning_rate,
        num_warmup_updates=num_warmup_updates,
        save_per_updates=save_per_updates,
        keep_last_n_checkpoints=keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=batch_size_per_gpu,
        batch_size_type=batch_size_type,
        max_samples=max_samples,
        grad_accumulation_steps=grad_accumulation_steps,
        max_grad_norm=max_grad_norm,
        logger=logger_type,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        last_per_updates=last_per_updates,
        log_samples=log_samples,
        bnb_optimizer=bnb_optimizer,
        mel_spec_type=mel_spec_type,
        validation_interval=validation_interval,
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
    )

    # Load training dataset
    print(f"Loading training dataset: {dataset_name}")
    train_dataset = load_dataset(dataset_name, tokenizer_type, mel_spec_kwargs=mel_spec_kwargs)

    # Load validation dataset if specified
    val_dataset = None
    if validation_split and os.path.exists(validation_split):
        print(f"Loading validation dataset from: {validation_split}")
        val_dataset = ManifestDataset(
            validation_split,
            **mel_spec_kwargs,
        )
        print(f"Validation dataset size: {len(val_dataset)}")
    else:
        print("No validation dataset configured. Skipping validation and early stopping.")

    # Start training
    print("\nStarting training...")
    print(f"Epochs: {epochs}")
    print(f"Batch size per GPU: {batch_size_per_gpu} ({batch_size_type})")
    print(f"Learning rate: {learning_rate}")
    print(f"Checkpoint path: {checkpoint_path}")

    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_workers=16,
        resumable_with_seed=666,
    )

    print("\nTraining completed!")
    print(f"Checkpoints saved to: {checkpoint_path}")
    if os.path.exists(f"{checkpoint_path}/model_best.pt"):
        print(f"Best model: {checkpoint_path}/model_best.pt")
    if os.path.exists(f"{checkpoint_path}/model_last.pt"):
        print(f"Last model: {checkpoint_path}/model_last.pt")


if __name__ == "__main__":
    main()
