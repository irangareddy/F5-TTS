# F5-TTS Setup Verification Script for NVIDIA RTX 4090
# Run this script after completing Step 01 setup

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "F5-TTS Setup Verification" -ForegroundColor Cyan
Write-Host "NVIDIA RTX 4090 - Windows" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$allChecksPassed = $true

# 1. Check conda environment
Write-Host "[1/6] Checking conda environment..." -ForegroundColor Yellow
if ($env:CONDA_DEFAULT_ENV -eq "f5-tts") {
    Write-Host "    ✓ f5-tts environment is active" -ForegroundColor Green
} else {
    Write-Host "    ✗ f5-tts environment is NOT active" -ForegroundColor Red
    Write-Host "      Run: conda activate f5-tts" -ForegroundColor Yellow
    $allChecksPassed = $false
}

# 2. Check Python version
Write-Host "`n[2/6] Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python 3\.(1[0-9]|[2-9])") {
        Write-Host "    ✓ $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "    ⚠ $pythonVersion (recommended: Python 3.11)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "    ✗ Python not found" -ForegroundColor Red
    $allChecksPassed = $false
}

# 3. Check PyTorch and CUDA
Write-Host "`n[3/6] Checking PyTorch and CUDA..." -ForegroundColor Yellow
try {
    $pytorchCheck = python -c "import torch; print(f'{torch.__version__}'); print(f'{torch.cuda.is_available()}'); print(f'{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1
    $lines = $pytorchCheck -split "`n"
    if ($lines.Count -ge 3) {
        Write-Host "    ✓ PyTorch version: $($lines[0])" -ForegroundColor Green
        if ($lines[1] -eq "True") {
            Write-Host "    ✓ CUDA available: True" -ForegroundColor Green
            Write-Host "    ✓ GPU device: $($lines[2])" -ForegroundColor Green
        } else {
            Write-Host "    ✗ CUDA not available" -ForegroundColor Red
            $allChecksPassed = $false
        }
    }
} catch {
    Write-Host "    ✗ PyTorch not installed or error: $_" -ForegroundColor Red
    $allChecksPassed = $false
}

# 4. Check F5-TTS installation
Write-Host "`n[4/6] Checking F5-TTS installation..." -ForegroundColor Yellow
try {
    python -c "from f5_tts.api import F5TTS; print('OK')" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    ✓ F5-TTS installed correctly" -ForegroundColor Green
    } else {
        Write-Host "    ✗ F5-TTS import failed" -ForegroundColor Red
        $allChecksPassed = $false
    }
} catch {
    Write-Host "    ✗ F5-TTS not installed. Run: pip install -e ." -ForegroundColor Red
    $allChecksPassed = $false
}

# 5. Check FFmpeg
Write-Host "`n[5/6] Checking FFmpeg..." -ForegroundColor Yellow
try {
    $ffmpegVersion = ffmpeg -version 2>&1 | Select-Object -First 1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    ✓ FFmpeg is installed" -ForegroundColor Green
    } else {
        Write-Host "    ⚠ FFmpeg not found (may cause issues with audio processing)" -ForegroundColor Yellow
        Write-Host "      Install from: https://ffmpeg.org/download.html" -ForegroundColor Yellow
    }
} catch {
    Write-Host "    ⚠ FFmpeg not found (may cause issues with audio processing)" -ForegroundColor Yellow
}

# 6. Check VOXE dataset
Write-Host "`n[6/6] Checking VOXE dataset..." -ForegroundColor Yellow
$voxeChecks = @{
    "Manifests directory" = "data\voxe\manifests"
    "Train manifest" = "data\voxe\manifests\train.jsonl"
    "Val manifest" = "data\voxe\manifests\val.jsonl"
    "Test manifest" = "data\voxe\manifests\test.jsonl"
    "WAV files directory" = "data\voxe\wav24k"
    "Mel files directory" = "data\voxe\mels"
}

foreach ($check in $voxeChecks.GetEnumerator()) {
    if (Test-Path $check.Value) {
        if ($check.Key -match "manifest") {
            $lineCount = (Get-Content $check.Value -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
            Write-Host "    ✓ $($check.Key): $lineCount samples" -ForegroundColor Green
        } else {
            Write-Host "    ✓ $($check.Key) exists" -ForegroundColor Green
        }
    } else {
        Write-Host "    ✗ $($check.Key) not found: $($check.Value)" -ForegroundColor Red
        $allChecksPassed = $false
    }
}

# 7. Check checkpoints directory
Write-Host "`n[7/6] Checking checkpoints directory..." -ForegroundColor Yellow
if (Test-Path "ckpts") {
    Write-Host "    ✓ ckpts directory exists" -ForegroundColor Green
} else {
    Write-Host "    ⚠ ckpts directory doesn't exist (will be created during training)" -ForegroundColor Yellow
}

# 8. Check NVIDIA GPU directly
Write-Host "`n[8/9] Checking NVIDIA GPU..." -ForegroundColor Yellow
try {
    $nvidiaSmi = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    ✓ GPU Info:" -ForegroundColor Green
        $nvidiaSmi | ForEach-Object { Write-Host "      $_" -ForegroundColor Cyan }
    } else {
        Write-Host "    ✗ nvidia-smi failed. Check NVIDIA drivers." -ForegroundColor Red
        $allChecksPassed = $false
    }
} catch {
    Write-Host "    ✗ Cannot run nvidia-smi. Check NVIDIA drivers." -ForegroundColor Red
    $allChecksPassed = $false
}

# 9. Check Docker and GPU support
Write-Host "`n[9/9] Checking Docker setup..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    ✓ $dockerVersion" -ForegroundColor Green
        
        # Check F5-TTS Docker image
        $imageExists = docker images f5-tts:latest --format "{{.Repository}}:{{.Tag}}" 2>&1
        if ($imageExists -match "f5-tts") {
            Write-Host "    ✓ F5-TTS Docker image exists" -ForegroundColor Green
        } else {
            Write-Host "    ⚠ F5-TTS Docker image not found. Run: docker build -t f5-tts:latest ." -ForegroundColor Yellow
        }
        
        # Check GPU support in Docker
        try {
            docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "    ✓ Docker GPU support working" -ForegroundColor Green
            } else {
                Write-Host "    ⚠ Docker GPU support may not be configured" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "    ⚠ Could not verify Docker GPU support" -ForegroundColor Yellow
        }
    } else {
        Write-Host "    ✗ Docker not found. Install Docker Desktop for Windows." -ForegroundColor Red
        $allChecksPassed = $false
    }
} catch {
    Write-Host "    ✗ Docker not found. Install Docker Desktop for Windows." -ForegroundColor Red
    $allChecksPassed = $false
}

# Final summary
Write-Host "`n========================================" -ForegroundColor Cyan
if ($allChecksPassed) {
    Write-Host "✓ All critical checks passed!" -ForegroundColor Green
    Write-Host "You're ready to proceed to Step 02: Training`n" -ForegroundColor Green
} else {
    Write-Host "✗ Some checks failed. Please fix the issues above before proceeding.`n" -ForegroundColor Red
    Write-Host "Refer to STEP01_SETUP.md for detailed troubleshooting.`n" -ForegroundColor Yellow
}
Write-Host "========================================`n" -ForegroundColor Cyan

