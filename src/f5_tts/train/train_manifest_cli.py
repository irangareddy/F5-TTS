#!/usr/bin/env python3
"""
Training CLI using manifest-based dataset loading (full dataset support).
Loads training data from JSONL manifest files instead of Arrow format.
Supports validation and early stopping with manifest-based validation set.
"""

import argparse
import logging
import os
import shutil
import warnings
from importlib.resources import files

from cached_path import cached_path

from f5_tts.model import CFM, DiT, Trainer, UNetT
from f5_tts.model.dataset import ManifestDataset
from f5_tts.model.utils import get_tokenizer

# Suppress CUDA/PyTorch warnings and info messages (quiet mode)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


# Dataset Settings
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"


def parse_args():
    parser = argparse.ArgumentParser(description="Train CFM Model using Manifest Dataset")

    parser.add_argument(
        "--exp_name",
        type=str,
        default="F5TTS_v1_Base",
        choices=["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base"],
        help="Experiment name",
    )
    parser.add_argument("--dataset_name", type=str, default="voxe", help="Name of the dataset")
    parser.add_argument(
        "--manifest_train",
        type=str,
        default="data/voxe/manifests/train.jsonl",
        help="Path to training manifest file",
    )
    parser.add_argument(
        "--manifest_val",
        type=str,
        default="data/voxe/manifests/val.jsonl",
        help="Path to validation manifest file",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size_per_gpu", type=int, default=190, help="Batch size per GPU (frames)")
    parser.add_argument(
        "--batch_size_type", type=str, default="frame", choices=["frame", "sample"], help="Batch size type"
    )
    parser.add_argument("--max_samples", type=int, default=32, help="Max sequences per batch")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--num_warmup_updates", type=int, default=500, help="Warmup updates")
    parser.add_argument("--save_per_updates", type=int, default=500, help="Save checkpoint every N updates")
    parser.add_argument(
        "--keep_last_n_checkpoints",
        type=int,
        default=2,
        help="Keep last N checkpoints",
    )
    parser.add_argument("--last_per_updates", type=int, default=2000, help="Save last model every N updates")
    parser.add_argument("--finetune", action="store_true", help="Use fine-tuning")
    parser.add_argument("--pretrain", type=str, default=None, help="Path to pretrained checkpoint")
    parser.add_argument("--tokenizer", type=str, default="char", choices=["pinyin", "char", "custom"])
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to custom tokenizer")
    parser.add_argument("--log_samples", action="store_true", help="Log sample audio during training")
    parser.add_argument("--logger", type=str, default="wandb", choices=["wandb", "tensorboard", None])
    parser.add_argument("--bnb_optimizer", action="store_true", help="Use 8-bit optimizer")
    parser.add_argument(
        "--validation_interval", type=int, default=1, help="Validate every N epochs"
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=3, help="Early stopping patience"
    )
    parser.add_argument(
        "--early_stopping_threshold", type=float, default=0.001, help="Early stopping threshold"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = str(files("f5_tts").joinpath(f"../../ckpts/{args.exp_name}_{args.dataset_name}_manifest"))

    # Model parameters based on experiment name
    if args.exp_name == "F5TTS_v1_Base":
        model_cls = DiT
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
        )
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
            else:
                ckpt_path = args.pretrain

    elif args.exp_name == "F5TTS_Base":
        model_cls = DiT
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            text_mask_padding=False,
            conv_layers=4,
            pe_attn_head=1,
        )
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"))
            else:
                ckpt_path = args.pretrain

    elif args.exp_name == "E2TTS_Base":
        model_cls = UNetT
        model_cfg = dict(
            dim=1024,
            depth=24,
            heads=16,
            ff_mult=4,
            text_mask_padding=False,
            pe_attn_head=1,
        )
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.pt"))
            else:
                ckpt_path = args.pretrain

    if args.finetune:
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)

        file_checkpoint = os.path.basename(ckpt_path)
        if not file_checkpoint.startswith("pretrained_"):
            file_checkpoint = "pretrained_" + file_checkpoint
        file_checkpoint = os.path.join(checkpoint_path, file_checkpoint)
        if not os.path.isfile(file_checkpoint):
            shutil.copy2(ckpt_path, file_checkpoint)
            print("Copied pretrained checkpoint for fine-tuning")

    # Tokenizer
    tokenizer = args.tokenizer
    if tokenizer == "custom":
        if not args.tokenizer_path:
            raise ValueError("Custom tokenizer selected but no tokenizer_path provided")
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = args.dataset_name

    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    print(f"\nvocab: {vocab_size}")
    print(f"vocoder: {mel_spec_type}")
    print(f"Training manifest: {args.manifest_train}")
    print(f"Validation manifest: {args.manifest_val}")

    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    # Create model
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    # Build comprehensive model config dict for WandB
    model_cfg_dict = {
        # Model architecture
        "model_name": args.exp_name,
        "backbone": "DiT" if args.exp_name in ["F5TTS_v1_Base", "F5TTS_Base"] else "UNetT",
        **model_cfg,

        # Training configuration
        "dataset_name": args.dataset_name,
        "dataset_type": "manifest",
        "learning_rate": args.learning_rate,
        "batch_size_per_gpu": args.batch_size_per_gpu,
        "batch_size_type": args.batch_size_type,
        "max_samples": args.max_samples,
        "grad_accumulation_steps": args.grad_accumulation_steps,
        "max_grad_norm": args.max_grad_norm,
        "epochs": args.epochs,
        "num_warmup_updates": args.num_warmup_updates,
        "save_per_updates": args.save_per_updates,
        "keep_last_n_checkpoints": args.keep_last_n_checkpoints,
        "last_per_updates": args.last_per_updates,

        # Validation/Early stopping
        "validation_interval": args.validation_interval,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,

        # Additional settings
        "tokenizer": args.tokenizer,
        "bnb_optimizer": args.bnb_optimizer,
        "finetune": args.finetune,
        "log_samples": args.log_samples,
        "mel_spec_type": mel_spec_type,
    }

    # Create trainer
    trainer = Trainer(
        model,
        args.epochs,
        args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=args.batch_size_per_gpu,
        batch_size_type=args.batch_size_type,
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        logger=args.logger,
        wandb_project=args.dataset_name,
        wandb_run_name=args.exp_name,
        log_samples=args.log_samples,
        last_per_updates=args.last_per_updates,
        bnb_optimizer=args.bnb_optimizer,
        model_cfg_dict=model_cfg_dict,
        validation_interval=args.validation_interval,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
    )

    # Load training dataset from manifest
    print("\nLoading training dataset from manifest...")
    train_dataset = ManifestDataset(
        args.manifest_train,
        **mel_spec_kwargs,
    )
    print(f"Training samples: {len(train_dataset)}")

    # Load validation dataset from manifest if available
    val_dataset = None
    if os.path.exists(args.manifest_val):
        print(f"Loading validation dataset from manifest...")
        val_dataset = ManifestDataset(
            args.manifest_val,
            **mel_spec_kwargs,
        )
        print(f"Validation samples: {len(val_dataset)}")
    else:
        print(f"Validation manifest not found: {args.manifest_val}")

    print(f"\nStarting training...")
    print(f"Epochs: {args.epochs}")
    print(f"Checkpoint path: {checkpoint_path}")

    # Train with validation and early stopping
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_workers=16,
        resumable_with_seed=666,
    )

    print("\nTraining completed!")
    print(f"Best model: {checkpoint_path}/model_best.pt")
    print(f"Last model: {checkpoint_path}/model_last.pt")


if __name__ == "__main__":
    main()
