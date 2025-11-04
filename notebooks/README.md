# F5-TTS Jupyter Lab Notebooks

This directory contains interactive Jupyter notebooks for F5-TTS fine-tuning workflows.

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
docker-compose up
```

Then open http://localhost:8888 in your browser.

### Option 2: Using PowerShell Script

```powershell
.\start_jupyter.ps1
```

Then open http://localhost:8888 in your browser.

### Option 3: Using Bash Script

```bash
chmod +x start_jupyter.sh
./start_jupyter.sh
```

Then open http://localhost:8888 in your browser.

### Option 4: Manual Docker Command

```bash
docker run -d \
  --name f5-tts-jupyter \
  --gpus all \
  --shm-size=2g \
  -p 8888:8888 \
  -v "${PWD}/data:/workspace/F5-TTS/data" \
  -v "${PWD}/ckpts:/workspace/F5-TTS/ckpts" \
  -v "${PWD}/src:/workspace/F5-TTS/src" \
  -v "${PWD}/notebooks:/workspace/F5-TTS/notebooks" \
  -v "${PWD}/tests:/workspace/F5-TTS/tests" \
  -w /workspace/F5-TTS \
  f5-tts:latest \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```

## Available Notebooks

### `voxe_finetune.ipynb`
Complete workflow for fine-tuning F5-TTS on the VOXE dataset:
- Environment setup and verification
- Dataset verification
- Training configuration
- Fine-tuning execution
- Model testing
- Inference examples

## Stopping Jupyter Lab

```bash
# Stop the container
docker stop f5-tts-jupyter

# Remove the container
docker rm f5-tts-jupyter

# Or if using docker-compose
docker-compose down
```

## Notes

- Jupyter Lab runs with no password/token for easy access (localhost only)
- All data, checkpoints, and notebooks are mounted as volumes
- GPU access is enabled for training within notebooks
- The default workspace is `/workspace/F5-TTS` inside the container

