# F5-TTS: Fine-Tuning Setup

Quick setup guide for fine-tuning F5-TTS models.

## Clone & Setup

```bash
# Clone repository
git clone <your-repo-url>
cd F5-TTS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install PyTorch (adjust CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install package
pip install -e .
```

## Data Check

```bash
# Verify data structure
# Training: data/<dataset_name>/manifests/train.jsonl
# Validation: data/<dataset_name>/manifests/val.jsonl

# Calculate training steps (for voxe_data_v2)
python scripts/calculate_steps_voxe_data_v2.py

# Verify vocab exists (for char tokenizer)
# Script auto-falls back to data/voxe_char/vocab.txt if needed
```

## Fine-Tune Start

### Single Config (params.yaml)

```bash
# Edit params.yaml:
#   - dataset_name: your_dataset
#   - validation_split: data/your_dataset/manifests/val.jsonl
#   - epochs, batch_size, etc.

python train_with_validation.py
```

### Experimental Design (params2.yaml)

```bash
# Copy and edit for experiments
cp params.yaml params2.yaml
# Edit params2.yaml, then modify train_with_validation.py to use it
```

### Key Settings

```yaml
training:
  dataset_name: voxe_data_v2
  validation_split: data/voxe_data_v2/manifests/val.jsonl
  epochs: 50
  batch_size_per_gpu: 170
  num_workers: 8
  early_stopping_patience: 10
```

### Output

- Checkpoints: `ckpts/F5TTS_v1_Base_<dataset>_manifest/`
- Best: `model_best.pt`, Last: `model_last.pt`
- Logs: `wandb/` (if enabled)

## Files to Track

Essential files for training:
- `train_with_validation.py` - Training script
- `scripts/calculate_steps_voxe_data_v2.py` - Step calculator
- `params.yaml` - Main config (or `params2.yaml` for experiments)
- `src/f5_tts/model/dataset.py` - Dataset loader with path mapping

## Troubleshooting

- **Vocab error**: Auto-falls back to `data/voxe_char/vocab.txt`
- **GPU not used**: Trainer uses Accelerator (auto-detects)
- **Path errors**: Use `data/` prefix in manifests, not `outputs/full/`

---

For full documentation, see original [README](README.md) or [training guide](src/f5_tts/train).
