# VOXE Fine-Tuning Guide: F5-TTS on Custom Dataset

This guide provides a quick-start workflow for fine-tuning F5-TTS on the VOXE dataset. Follow these steps to get results fast, then scale up based on your findings.

---

## Overview

**What is fine-tuning?**
- Start with a pre-trained F5-TTS model and adapt it to your dataset
- Much faster and cheaper than training from scratch
- Requires far fewer samples (we're using 20 to test)

**Your VOXE Dataset Structure:**
```
data/voxe/
├── manifests/          # Dataset metadata (JSONL format)
│   ├── train.jsonl     # Training samples
│   ├── val.jsonl       # Validation samples
│   ├── test.jsonl      # Test samples
│   └── all.jsonl       # Combined manifest
├── mels/               # Pre-computed mel spectrograms
│   └── esd/0011/angry/0011_000351.mel.npy  (example)
└── wav24k/             # Raw audio files at 24kHz
    └── esd/0011/Angry/0011_000351.wav  (example)
```

**Dataset Details:**
- **Source:** ESD (Emotional Speech Dataset)
- **Format:** English speech with emotions (Angry, Neutral, etc.)
- **Sample Duration:** 1-3 seconds
- **Speaker:** Multiple speakers (e.g., 0011)
- **Audio Format:** WAV files, 24kHz sample rate

---

## Quick Setup

### 1. Activate Environment
```bash
# If using conda (recommended)
conda activate f5-tts

# If not yet created, create first:
# conda create -n f5-tts python=3.11
# conda activate f5-tts
```

### 2. Install Dependencies (if not already done)
```bash
# Install f5-tts in editable mode for development
pip install -e .

# Or if already installed, just verify:
python -c "from f5_tts.api import F5TTS; print('F5-TTS installed correctly')"
```

### 3. Verify VOXE Data is Accessible
```bash
# Check that your data is in the right place
dir data\voxe\manifests
# Should show: all.jsonl  test.jsonl  train.jsonl  val.jsonl

dir data\voxe\wav24k
# Should show: esd (or similar dataset directories)

dir data\voxe\mels
# Should show: esd (or similar dataset directories)
```

---

## Data Structure & Preparation

### Current Data Format

Your VOXE dataset is already in the correct format. Each sample in the manifest includes:

```json
{
  "wav_path": "outputs/full/wav24k/esd/0011/Angry/0011_000351.wav",
  "mel_path": "outputs/full/mels/esd/0011/angry/0011_000351.mel.npy",
  "text": "the nine the eggs, i keep.",
  "n_mels": 100,
  "n_frames": 153,
  "duration_s": 1.6213333333333333,
  "speaker_id": "0011",
  "emotion": "angry",
  "dataset": "ESD",
  "lang": "en",
  "split": "train"
}
```

### Key Fields Explained
| Field | Purpose | Example |
|-------|---------|---------|
| `wav_path` | Path to raw audio file | `outputs/full/wav24k/esd/0011/Angry/0011_000351.wav` |
| `mel_path` | Path to pre-computed mel spectrogram | `outputs/full/mels/esd/0011/angry/0011_000351.mel.npy` |
| `text` | Transcription of speech | `"the nine the eggs, i keep."` |
| `duration_s` | Audio duration in seconds | `1.62` |
| `speaker_id` | Speaker identifier | `"0011"` |
| `emotion` | Emotional category | `"angry"` |
| `split` | Dataset split (train/val/test) | `"train"` |

### Adding New Data

If you want to add more data later, follow this format:
1. Add audio files to `data/voxe/wav24k/{dataset}/{speaker}/{emotion}/`
2. Pre-compute mels: `python src/f5_tts/train/datasets/prepare_csv_wavs.py` (optional, speeds up training)
3. Add entries to the appropriate `.jsonl` manifest file

---

## Ready-to-Run: 20-Sample / 2-Epoch Experiment

This is your starting configuration for fast experimentation.

### Quick Start Command

```bash
python src/f5_tts/train/finetune_cli.py ^
  --exp_name F5TTS_v1_Base ^
  --dataset_name voxe ^
  --learning_rate 1e-5 ^
  --batch_size_per_gpu 800 ^
  --batch_size_type frame ^
  --max_samples 8 ^
  --epochs 2 ^
  --num_warmup_updates 100 ^
  --save_per_updates 50 ^
  --keep_last_n_checkpoints 3 ^
  --finetune ^
  --tokenizer char
```

**Note:** `--exp_name` selects the model architecture (F5TTS_v1_Base, F5TTS_Base, or E2TTS_Base), not a custom experiment name. The checkpoint output directory is determined by `--dataset_name` (will be saved to `ckpts/voxe/`).

**For Linux/Mac:** Replace `^` with `\` and adjust path syntax.

### What This Does
- **exp_name:** Selects model architecture (F5TTS_v1_Base, F5TTS_Base, or E2TTS_Base)
- **dataset_name:** Points to `data/voxe/` directory and determines checkpoint output directory (`ckpts/voxe/`)
- **epochs:** Runs for only 2 passes through the data (very fast)
- **batch_size_per_gpu:** 800 frames per GPU batch (memory-efficient)
- **save_per_updates:** Saves full checkpoint every 50 updates (frequent saves)
- **finetune:** Uses pre-trained weights as starting point

### Expected Output
```
[2025-11-03 10:30:15] Training started
[2025-11-03 10:30:20] Epoch 1/2, Batch 1/3
[2025-11-03 10:30:45] Loss: 3.45 | LR: 1e-5
[2025-11-03 10:31:10] Epoch 1/2, Batch 2/3
[2025-11-03 10:31:35] Loss: 2.98 | LR: 1e-5
[2025-11-03 10:32:00] Epoch 1/2, Batch 3/3
[2025-11-03 10:32:25] Loss: 2.52 | LR: 1e-5
[2025-11-03 10:32:30] Epoch 1/2 complete. Saving checkpoint...
[2025-11-03 10:32:50] Checkpoint saved to: ckpts/voxe/ckpt_50_updates.pt
...
[2025-11-03 10:35:00] Training complete!
```

### Configuration Explained

| Parameter | Value | Why This Works |
|-----------|-------|----------------|
| `learning_rate` | `1e-5` | Very conservative - preserves pre-trained knowledge |
| `batch_size_per_gpu` | `800` | Small batches for small dataset, fits in GPU memory |
| `max_samples` | `8` | Max 8 sequences per batch (prevents overfitting) |
| `epochs` | `2` | Quick validation - not meant for accuracy yet |
| `num_warmup_updates` | `100` | Warmup over ~100 steps (total ~200 steps in 2 epochs) |
| `save_per_updates` | `50` | Save after every 50 updates (watch training progress) |

---

## Running Fine-Tuning

### Step 1: Check Prerequisites
```bash
# Verify F5-TTS installation
python -c "from f5_tts.train.finetune_cli import main; print('Ready to finetune')"

# Check VOXE data exists
python -c "import json; list(map(json.loads, open('data/voxe/manifests/train.jsonl')))[0:1]"
```

### Step 2: Run Training
Copy the command from the "Ready-to-Run" section above and execute it.

```bash
# Windows
python src/f5_tts/train/finetune_cli.py ^
  --exp_name F5TTS_VOXE_small_test ^
  --dataset_name voxe ^
  --learning_rate 1e-5 ^
  --batch_size_per_gpu 800 ^
  --batch_size_type frame ^
  --max_samples 8 ^
  --epochs 2 ^
  --num_warmup_updates 100 ^
  --save_per_updates 50 ^
  --keep_last_n_checkpoints 3 ^
  --finetune ^
  --tokenizer char
```

### Step 3: Monitor Real-Time Output
The script will print progress to console:
- **Loss values:** Should decrease over time (e.g., 3.45 → 2.98 → 2.52)
- **Learning rate:** Fixed at 1e-5 after warmup
- **Checkpoint saves:** Notification when weights are saved
- **Estimated time:** ~1-2 minutes for 20 samples with 2 epochs

### Step 4: Locate Checkpoints
After training completes, your checkpoints are saved in:
```
ckpts/voxe/
├── ckpt_50_updates.pt              # Checkpoint after 50 updates
├── ckpt_100_updates.pt             # Checkpoint after 100 updates
├── ckpt_150_updates.pt             # Checkpoint after 150 updates
├── config.json                     # Training configuration
└── training_log.txt                # Full training log
```

---

## Monitoring & Debugging

### 1. Understanding Loss Curves

**Good sign (loss decreasing):**
```
Epoch 1: Loss 3.45 → 3.20 → 2.98 → 2.75
Epoch 2: Loss 2.50 → 2.30 → 2.10 → 1.95
```

**Bad sign (loss increasing or flat):**
```
Epoch 1: Loss 3.45 → 3.50 → 3.55 → 3.60
```
→ **Solution:** Reduce `learning_rate` (try 5e-6) or check data quality

### 2. Common Errors & Solutions

#### Error: "RuntimeError: CUDA out of memory"
**Cause:** Batch size too large for your GPU
**Fix:** Reduce `--batch_size_per_gpu` value
```bash
# Instead of 800, try:
--batch_size_per_gpu 400
```

#### Error: "FileNotFoundError: data/voxe/manifests/train.jsonl"
**Cause:** Dataset path incorrect
**Fix:** Verify VOXE data location
```bash
# Make sure you're in the F5-TTS root directory
cd C:\Users\ranga\Desktop\F5-TTS
dir data\voxe\manifests
```

#### Error: "KeyError: 'wav_path'" or "Invalid audio file"
**Cause:** Manifest paths don't match actual file locations
**Fix:** Check path mapping
```bash
# Verify audio files exist
dir "data\voxe\wav24k\esd\0011\Angry"
# Should list: 0011_000351.wav, 0011_000352.wav, etc.
```

#### Error: "No such file or directory: ckpts"
**Cause:** Checkpoints directory doesn't exist
**Fix:** Create it first
```bash
mkdir ckpts
```

### 3. GPU Memory Debugging
```bash
# Check GPU status before training
nvidia-smi

# Expected output during training:
# GPU Memory Usage: ~2-4 GB (for small batch size)
# During 2-epoch run: Should complete in < 5 minutes
```

### 4. Checking Training Progress Mid-Training

While training is running, open another terminal:
```bash
# Watch checkpoint sizes grow
dir /S ckpts\voxe

# Tail training output (if logging to file)
# On Windows PowerShell:
Get-Content training.log -Tail 20 -Wait
```

---

## Evaluation & Testing

### Step 1: Use Fine-Tuned Model for Inference

After training, test your model on new text:

```python
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import tts_infer_stream

# Load your fine-tuned model
model = F5TTS(
    ckpt_path="ckpts/voxe/ckpt_150_updates.pt",
    use_ema=False  # Use regular weights (EMA weights are pre-trained dominated)
)

# Generate speech
ref_audio_path = "data/voxe/wav24k/esd/0011/Angry/0011_000351.wav"  # Reference voice
text_to_speak = "Hello, this is a test of the fine-tuned model."

gen, sr = tts_infer_stream(
    text_to_speak,
    ref_audio_path=ref_audio_path,
    model_obj=model
)

# Save result
import soundfile as sf
sf.write("test_output.wav", gen, sr)
print(f"Generated audio saved to test_output.wav (sample rate: {sr} Hz)")
```

### Step 2: Qualitative Evaluation Checklist

After generating test audio, ask yourself:
- [ ] Does the output sound natural?
- [ ] Is the speaker's voice recognizable?
- [ ] Are emotions/characteristics from the reference audio preserved?
- [ ] Is the speech clear and intelligible?
- [ ] Are there any artifacts or noise?

**Your Findings → Next Steps:**
- ✓ Good quality → Scale up to 100-200 samples, 5-10 epochs
- ✗ Poor quality → Check data quality, try different reference audio, adjust learning rate
- ? Unclear → Compare with base model inference (without fine-tuning)

### Step 3: Compare with Base Model

```python
# Load base pre-trained model (no fine-tuning)
base_model = F5TTS(ckpt_path="F5TTS_v1_Base")

# Generate with same reference and text
gen_base, sr_base = tts_infer_stream(
    "Hello, this is a test of the fine-tuned model.",
    ref_audio_path="data/voxe/wav24k/esd/0011/Angry/0011_000351.wav",
    model_obj=base_model
)

# Save for comparison
sf.write("base_model_output.wav", gen_base, sr_base)

# Compare: test_output.wav vs base_model_output.wav
# Use a listening tool like VLC Media Player or Audacity
```

### Step 4: Validation Metrics

During training, check your `ckpts/voxe/training_log.txt`:

```
Epoch 1/2
  Step 1: Loss = 3.45, Learning Rate = 1.0e-05
  Step 2: Loss = 3.20, Learning Rate = 1.0e-05
  Step 3: Loss = 2.98, Learning Rate = 1.0e-05
Epoch 1 complete. Loss: 2.88 (average)

Epoch 2/2
  Step 1: Loss = 2.50, Learning Rate = 1.0e-05
  Step 2: Loss = 2.30, Learning Rate = 1.0e-05
  Step 3: Loss = 2.10, Learning Rate = 1.0e-05
Epoch 2 complete. Loss: 2.30 (average)
```

**What to look for:**
- Loss should decrease by epoch 2
- No spikes or erratic behavior
- Consistent updates every ~50 steps

---

## Scaling Up: From 20 to Full Dataset

Once you validate results with 20 samples, gradually scale:

### Phase 1: Medium Dataset (50-100 samples)
```bash
python src/f5_tts/train/finetune_cli.py ^
  --exp_name F5TTS_v1_Base ^
  --dataset_name voxe ^
  --learning_rate 5e-6 ^
  --batch_size_per_gpu 1600 ^
  --batch_size_type frame ^
  --max_samples 16 ^
  --epochs 5 ^
  --num_warmup_updates 500 ^
  --save_per_updates 200 ^
  --keep_last_n_checkpoints 5 ^
  --finetune ^
  --tokenizer char
```

**Changes:**
- **learning_rate:** Reduced to 5e-6 (more conservative)
- **epochs:** 5 (more training passes)
- **num_warmup_updates:** 500 (longer warmup)
- **save_per_updates:** 200 (checkpoint every 200 steps)

### Phase 2: Large Dataset (200+ samples)
```bash
python src/f5_tts/train/finetune_cli.py ^
  --exp_name F5TTS_v1_Base ^
  --dataset_name voxe ^
  --learning_rate 7.5e-6 ^
  --batch_size_per_gpu 3200 ^
  --batch_size_type frame ^
  --max_samples 32 ^
  --epochs 10 ^
  --num_warmup_updates 2000 ^
  --save_per_updates 500 ^
  --keep_last_n_checkpoints 5 ^
  --finetune ^
  --tokenizer char
```

### Scaling Recommendations

| Dataset Size | Epochs | Batch Size | Learning Rate | Expected Time |
|--------------|--------|------------|---------------|----------------|
| 20 samples | 2 | 800 | 1e-5 | ~1-2 min |
| 50 samples | 5 | 1600 | 5e-6 | ~5 min |
| 100 samples | 5 | 2400 | 5e-6 | ~10 min |
| 200+ samples | 10 | 3200 | 7.5e-6 | ~30 min |
| 1000+ samples | 15 | 6400 | 1e-5 | ~2 hours |

**Guidelines:**
- Increase `batch_size_per_gpu` as dataset grows (more data = larger batches)
- Reduce `learning_rate` for larger datasets (stability)
- Increase `epochs` for better convergence (typical: 5-15)
- Adjust `num_warmup_updates` proportionally (rule of thumb: ~5% of total steps)

---

## Advanced Configuration Reference

### Parameter Meanings

| Parameter | Range | Effect |
|-----------|-------|--------|
| `learning_rate` | 1e-6 to 1e-4 | Lower = stable but slow, Higher = fast but unstable |
| `batch_size_per_gpu` | 400-10000 | Lower = memory efficient, Higher = faster but more memory |
| `max_samples` | 8-64 | Max sequences per batch (prevents overfitting with small data) |
| `epochs` | 1-20 | More epochs = better learning but risk of overfitting |
| `num_warmup_updates` | 100-5000 | Gradually increases LR at start (stabilizes training) |
| `save_per_updates` | 50-1000 | How often to save full checkpoints |
| `keep_last_n_checkpoints` | 1-10 | Keep N most recent; older ones deleted (saves disk) |

### Model Selection

```bash
# Use this line for different base models:

# Option 1: Latest F5-TTS v1 (RECOMMENDED)
--model F5TTS_v1_Base

# Option 2: Earlier F5-TTS version
--model F5TTS_Base

# Option 3: Smaller model (faster, lower quality)
--model F5TTS_Small

# Option 4: E2-TTS (alternative architecture)
--model E2TTS_Base
```

### Tokenizer Options

```bash
# Character-level tokenization (good for mixed languages)
--tokenizer char

# Pinyin tokenization (good for Chinese)
--tokenizer pinyin

# Custom tokenizer (if you prepared a vocab)
--tokenizer custom --tokenizer_path data/voxe_pinyin/vocab.txt
```

---

## Troubleshooting Checklist

### Before Starting Training
- [ ] F5-TTS is installed: `pip install -e .`
- [ ] VOXE data exists: `dir data/voxe/manifests`
- [ ] Manifest files are readable: `type data/voxe/manifests/train.jsonl | more`
- [ ] Audio files match paths in manifest

### During Training
- [ ] GPU is detected: `nvidia-smi` shows your GPU
- [ ] Loss is decreasing: Check console output
- [ ] No out-of-memory errors: If OOM, reduce `batch_size_per_gpu`
- [ ] Checkpoints are saving: Check `ckpts/voxe/` folder size growing

### After Training
- [ ] Checkpoint file exists: `dir ckpts/voxe`
- [ ] Can load checkpoint: Test with Python code above
- [ ] Generated audio is audible: Play generated .wav file

### If Still Stuck
1. Check the training log for specific error messages
2. Try the simplest configuration first (20 samples, 1 epoch)
3. Verify your environment: `python -c "import torch; print(torch.cuda.is_available())"`
4. Check disk space: `dir C:\` (need ~5GB free)

---

## Next Steps After Initial Results

### If Results are Good:
1. **Increase dataset size** to 100-200 samples
2. **Fine-tune more** with 5-10 epochs
3. **Adjust learning rate** down slightly (e.g., 5e-6)
4. **Test on different speakers/emotions** from VOXE
5. **Compare quality** with base model systematically

### If Results are Poor:
1. **Check data quality**: Are audio files clean? Are transcriptions accurate?
2. **Try different reference audio**: Some voices work better as references
3. **Reduce learning rate**: Try 5e-6 instead of 1e-5
4. **Verify manifest integrity**: Run validation script on JSONL files
5. **Test with more epochs**: Small datasets need more training to show effects

### For Production Use:
1. Fine-tune on your complete dataset (aim for 1000+ samples)
2. Run on 10-20 epochs with learning rate 5e-6 to 1e-5
3. Evaluate on held-out test set
4. Use the best checkpoint based on test loss

---

## Key Takeaways

✓ **Quick Start:** Run the command in the "Ready-to-Run" section
✓ **Monitor:** Watch loss decrease in console output
✓ **Evaluate:** Test checkpoint on new text with your reference audio
✓ **Scale:** Increase dataset size and epochs based on results
✓ **Iterate:** Adjust learning rate and batch size as needed

Your VOXE data is already in the correct format. You're ready to fine-tune!

---

## References & Further Reading

### Official Documentation
- Fine-tuning guide: `src/f5_tts/train/README.md`
- API reference: `src/f5_tts/api.py`
- Configuration examples: `src/f5_tts/configs/`

### Dataset Utilities
- Data preparation: `src/f5_tts/train/datasets/prepare_csv_wavs.py`
- Dataset loading: `src/f5_tts/model/dataset.py`

### Model Inference
- Inference utilities: `src/f5_tts/infer/utils_infer.py`
- Example configs: `src/f5_tts/infer/examples/`

### Useful Commands
```bash
# View available model checkpoints
dir ckpts

# Check latest checkpoint
dir ckpts\voxe /O:D /S

# Clean up old checkpoints (manual)
del ckpts\voxe\ckpt_*_updates.pt
```

---

**Last Updated:** November 3, 2025
**F5-TTS Version:** v1 Base
**For questions:** Check GitHub issues or review official F5-TTS documentation
