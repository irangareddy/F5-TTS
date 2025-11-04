# Step 01: Environment Setup & Verification (NVIDIA RTX 4090)

This document outlines Step 01 for setting up your F5-TTS fine-tuning environment on an NVIDIA RTX 4090 machine running Windows.

---

## Prerequisites Checklist

Before starting, verify you have:
- [ ] NVIDIA RTX 4090 GPU installed and recognized
- [ ] CUDA-capable drivers installed (recommended: CUDA 12.4+)
- [ ] Docker Desktop for Windows installed (recommended) OR Conda/Miniconda installed
- [ ] At least 10GB free disk space (for Docker image)
- [ ] Internet connection for downloading packages

---

## Setup Method: Choose Docker (Recommended) or Conda

**We recommend using Docker** as it provides a pre-configured environment with all dependencies. If you prefer local installation, use the Conda method below.

---

## Option A: Docker-Based Setup (Recommended)

### Step 01.A.1: Verify Docker and GPU Support

```powershell
# Check Docker is installed
docker --version

# Verify GPU support in Docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

**Expected output:** Should show your RTX 4090 GPU information.

### Step 01.A.2: Build F5-TTS Docker Image

```powershell
# Navigate to F5-TTS directory
cd C:\Users\ranga\Desktop\F5-TTS

# Build Docker image with tag
docker build -t f5-tts:latest -f Dockerfile .

# Verify image was created
docker images | findstr f5-tts
```

**Build time:** ~10-15 minutes (first time only)

**Alternative:** Pull pre-built image:
```powershell
docker pull ghcr.io/swivid/f5-tts:main
docker tag ghcr.io/swivid/f5-tts:main f5-tts:latest
```

### Step 01.A.3: Verify Docker Image Setup

```powershell
# Test PyTorch and CUDA
docker run --rm --gpus all f5-tts:latest python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Test F5-TTS installation
docker run --rm --gpus all f5-tts:latest python -c "from f5_tts.api import F5TTS; print('F5-TTS installed correctly')"
```

**Expected output:**
```
PyTorch: 2.4.0
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
F5-TTS installed correctly
```

### Step 01.A.4: Verify VOXE Dataset Accessibility

```powershell
# Check manifest files
docker run --rm --gpus all -v "${PWD}\data:/workspace/F5-TTS/data:ro" f5-tts:latest ls /workspace/F5-TTS/data/voxe/manifests/

# Count training samples
docker run --rm --gpus all -v "${PWD}\data:/workspace/F5-TTS/data:ro" f5-tts:latest python -c "with open('/workspace/F5-TTS/data/voxe/manifests/train.jsonl') as f: print('Train samples:', sum(1 for _ in f))"
```

**Expected output:** Should show manifest files and count training samples (e.g., 15152).

### Step 01.A.5: Ready to Train!

You can now run training using the provided script:

```powershell
# Run the training script
.\run_training_docker.ps1
```

**Advantages of Docker:**
- ✅ No Python version conflicts
- ✅ All dependencies pre-installed
- ✅ PyTorch with CUDA pre-configured
- ✅ FFmpeg included
- ✅ Isolated environment
- ✅ Reproducible across machines

**Jump to:** [Step 01.6: Verify VOXE Dataset Structure](#step-016-verify-voxe-dataset-structure) (skip conda steps)

---

## Option B: Conda-Based Setup (Local Installation)

## Step 01.1: Verify GPU Setup

### 1.1.1 Check NVIDIA Driver & CUDA
```powershell
# Check NVIDIA driver version
nvidia-smi

# Expected output should show:
# - Driver Version: 550.x or higher
# - CUDA Version: 12.4 or compatible
# - GPU: RTX 4090 with available memory
```

**If `nvidia-smi` fails:**
- Install/update NVIDIA drivers from [NVIDIA website](https://www.nvidia.com/Download/index.aspx)
- Restart your computer after driver installation

### 1.1.2 Verify CUDA Installation (Optional Check)
```powershell
# Check if CUDA toolkit is installed (optional)
nvcc --version

# Note: PyTorch comes with its own CUDA libraries, so system CUDA is optional
```

---

## Step 01.2: Create Conda Environment

### 1.2.1 Create Python 3.11 Environment
```powershell
# Navigate to F5-TTS directory (if not already there)
cd C:\Users\ranga\Desktop\F5-TTS

# Create new conda environment with Python 3.11
conda create -n f5-tts python=3.11 -y

# Activate the environment
conda activate f5-tts
```

**If conda command not found:**
- Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
- Or use Anaconda from https://www.anaconda.com/download
- Restart PowerShell after installation

---

## Step 01.3: Install PyTorch with CUDA Support

### 1.3.1 Install PyTorch 2.4.0 with CUDA 12.4
```powershell
# Make sure f5-tts environment is activated
conda activate f5-tts

# Install PyTorch with CUDA 12.4 support (recommended for RTX 4090)
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

**Alternative PyTorch versions:**
- For CUDA 11.8: `pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118`
- For CUDA 12.1: `pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121`

### 1.3.2 Verify PyTorch CUDA Installation
```powershell
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected output:**
```
PyTorch version: 2.4.0+cu124
CUDA available: True
CUDA version: 12.4
GPU device: NVIDIA GeForce RTX 4090
```

**If CUDA is False:**
- Check NVIDIA driver installation
- Verify CUDA version compatibility
- Try reinstalling PyTorch with matching CUDA version

---

## Step 01.4: Install F5-TTS Package

### 1.4.1 Install F5-TTS in Editable Mode
```powershell
# Ensure you're in the F5-TTS root directory
cd C:\Users\ranga\Desktop\F5-TTS

# Make sure f5-tts environment is activated
conda activate f5-tts

# Install F5-TTS and all dependencies
pip install -e .
```

**This will install:**
- F5-TTS package and dependencies from `pyproject.toml`
- Required packages: accelerate, transformers, librosa, soundfile, vocos, etc.
- Training utilities and CLI commands

**Installation time:** ~5-10 minutes depending on internet speed

### 1.4.2 Verify F5-TTS Installation
```powershell
# Test basic import
python -c "from f5_tts.api import F5TTS; print('F5-TTS installed correctly')"

# Test CLI commands are available
f5-tts_infer-cli --help
f5-tts_finetune-cli --help
```

**Expected output:**
```
F5-TTS installed correctly
Usage: f5-tts_infer-cli [OPTIONS]
...
```

---

## Step 01.5: Install Additional System Dependencies

### 1.5.1 FFmpeg (Required for Audio Processing)
```powershell
# Check if FFmpeg is installed
ffmpeg -version

# If not installed, download from https://ffmpeg.org/download.html
# Or use chocolatey (if installed):
# choco install ffmpeg
```

**FFmpeg Installation Options:**
1. **Download binary:** https://www.gyan.dev/ffmpeg/builds/
   - Extract to a folder (e.g., `C:\ffmpeg`)
   - Add `C:\ffmpeg\bin` to Windows PATH environment variable
2. **Chocolatey:** `choco install ffmpeg` (requires admin)
3. **Scoop:** `scoop install ffmpeg`

**Verify FFmpeg after installation:**
```powershell
ffmpeg -version
# Should show version info without errors
```

---

## Step 01.6: Verify VOXE Dataset Structure

### 1.6.1 Check Manifest Files Exist
```powershell
# Navigate to F5-TTS directory
cd C:\Users\ranga\Desktop\F5-TTS

# Check manifest files
dir data\voxe\manifests
```

**Expected output:**
```
all.jsonl
test.jsonl
train.jsonl
val.jsonl
```

### 1.6.2 Verify Manifest File Format
```powershell
# Check first entry in train.jsonl
python -c "import json; data = [json.loads(line) for line in open('data/voxe/manifests/train.jsonl')]; print(json.dumps(data[0], indent=2))"
```

**Expected output should show:**
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

### 1.6.3 Check Audio Files Exist
```powershell
# Check WAV files directory
dir data\voxe\wav24k\esd\0011\Angry

# Check mel spectrograms directory
dir data\voxe\mels\esd\0011\angry
```

**Expected:** Should see `.wav` files in `wav24k` and `.mel.npy` files in `mels`

### 1.6.4 Count Training Samples
```powershell
# Count lines in train.jsonl (each line = 1 sample)
python -c "with open('data/voxe/manifests/train.jsonl', 'r') as f: print(f'Training samples: {sum(1 for _ in f)}')"
```

**Note:** You'll be starting with a small subset (8 samples) for testing, but it's good to know your total dataset size.

---

## Step 01.7: Create Checkpoints Directory

### 1.7.1 Ensure Checkpoints Directory Exists
```powershell
# Create ckpts directory if it doesn't exist
if (-not (Test-Path "ckpts")) {
    New-Item -ItemType Directory -Path "ckpts"
    Write-Host "Created ckpts directory"
} else {
    Write-Host "ckpts directory already exists"
}
```

---

## Step 01.8: Final Verification Script

Run this comprehensive check to verify everything is ready:

```powershell
# Complete verification script
Write-Host "=== F5-TTS Setup Verification ===" -ForegroundColor Cyan

# 1. Check conda environment
Write-Host "`n1. Checking conda environment..." -ForegroundColor Yellow
if ($env:CONDA_DEFAULT_ENV -eq "f5-tts") {
    Write-Host "   ✓ f5-tts environment is active" -ForegroundColor Green
} else {
    Write-Host "   ✗ f5-tts environment is NOT active. Run: conda activate f5-tts" -ForegroundColor Red
}

# 2. Check PyTorch
Write-Host "`n2. Checking PyTorch..." -ForegroundColor Yellow
python -c "import torch; print(f'   ✓ PyTorch {torch.__version__}'); print(f'   ✓ CUDA available: {torch.cuda.is_available()}'); print(f'   ✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 3. Check F5-TTS
Write-Host "`n3. Checking F5-TTS..." -ForegroundColor Yellow
python -c "from f5_tts.api import F5TTS; print('   ✓ F5-TTS installed correctly')"

# 4. Check FFmpeg
Write-Host "`n4. Checking FFmpeg..." -ForegroundColor Yellow
try {
    $ffmpegVersion = ffmpeg -version 2>&1 | Select-Object -First 1
    Write-Host "   ✓ FFmpeg is installed: $ffmpegVersion" -ForegroundColor Green
} catch {
    Write-Host "   ✗ FFmpeg not found. Install from https://ffmpeg.org/download.html" -ForegroundColor Red
}

# 5. Check VOXE data
Write-Host "`n5. Checking VOXE dataset..." -ForegroundColor Yellow
if (Test-Path "data\voxe\manifests\train.jsonl") {
    $trainCount = (Get-Content "data\voxe\manifests\train.jsonl" | Measure-Object -Line).Lines
    Write-Host "   ✓ train.jsonl exists ($trainCount samples)" -ForegroundColor Green
} else {
    Write-Host "   ✗ train.jsonl not found" -ForegroundColor Red
}

# 6. Check checkpoints directory
Write-Host "`n6. Checking checkpoints directory..." -ForegroundColor Yellow
if (Test-Path "ckpts") {
    Write-Host "   ✓ ckpts directory exists" -ForegroundColor Green
} else {
    Write-Host "   ⚠ ckpts directory doesn't exist (will be created during training)" -ForegroundColor Yellow
}

Write-Host "`n=== Verification Complete ===" -ForegroundColor Cyan
```

**Save this as `verify_setup.ps1` and run:**
```powershell
.\verify_setup.ps1
```

---

## Step 01.9: Troubleshooting Common Issues

### Issue: Conda command not found
**Solution:** 
- Install Miniconda/Anaconda
- Restart PowerShell
- Or use full path: `C:\Users\<username>\miniconda3\Scripts\conda.exe`

### Issue: CUDA not available in PyTorch
**Solutions:**
1. Verify NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch with correct CUDA version
3. Check CUDA compatibility: RTX 4090 requires CUDA 11.8+ (preferably 12.4)

### Issue: FFmpeg not found
**Solution:**
- Download FFmpeg for Windows
- Add to PATH or use full path in scripts
- Or install via Chocolatey: `choco install ffmpeg`

### Issue: Import errors after pip install
**Solutions:**
1. Ensure you're in the F5-TTS root directory
2. Use `pip install -e .` (editable mode)
3. Check Python version: `python --version` (should be 3.11)
4. Reinstall: `pip uninstall f5-tts -y && pip install -e .`

### Issue: Path errors in Windows
**Solutions:**
- Use backslashes `\` for Windows paths
- Or use forward slashes `/` (Python handles both)
- Use PowerShell's `Test-Path` for verification

---

## Next Steps After Step 01

Once Step 01 is complete and verified:

1. **Proceed to Step 02:** Run your first 8-sample / 2-epoch test training
   - **Docker users:** Run `.\run_training_docker.ps1`
   - **Conda users:** Command ready in `VOXE_FINETUNE.md` section "Ready-to-Run"
   - Expected time: ~1-2 minutes

2. **Monitor GPU usage:** Use `nvidia-smi` in another terminal during training

3. **Check training logs:** Monitor console output for loss values

**For Docker users:** The `run_training_docker.ps1` script handles all volume mounting and runs the training command inside the container. Checkpoints will be saved to `ckpts/` directory on your host machine.

---

## Quick Reference Commands

```powershell
# Activate environment
conda activate f5-tts

# Verify GPU
nvidia-smi

# Verify PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Verify F5-TTS
python -c "from f5_tts.api import F5TTS; print('OK')"

# Check data
dir data\voxe\manifests
```

---

## Summary

**Step 01 Checklist:**
- [ ] NVIDIA GPU detected (`nvidia-smi` works)
- [ ] Conda environment `f5-tts` created and activated
- [ ] PyTorch 2.4.0 with CUDA 12.4 installed
- [ ] PyTorch detects CUDA and RTX 4090
- [ ] F5-TTS installed in editable mode (`pip install -e .`)
- [ ] F5-TTS imports successfully
- [ ] FFmpeg installed and accessible
- [ ] VOXE manifest files exist (`train.jsonl`, `val.jsonl`, `test.jsonl`)
- [ ] Audio files accessible (`data/voxe/wav24k/` and `data/voxe/mels/`)
- [ ] Checkpoints directory exists or ready to be created

**Estimated time:** 15-30 minutes (depending on download speeds)

**Once all checks pass, proceed to Step 02: First Training Run**

