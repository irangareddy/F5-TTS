# Taskfile Usage Guide

This project uses task automation with `params.yaml` for centralized configuration management. You can use either the PowerShell wrapper (`task.ps1`) or install Task runner.

## Quick Start (PowerShell - No Installation Required)

Use the PowerShell wrapper script directly:

```powershell
# Check configuration
.\task.ps1 check

# Run training
.\task.ps1 train

# Show training command without executing
.\task.ps1 train:dry-run
```

## Installation (Optional - For Task Runner)

If you prefer using the `task` command directly, install Task:

### Windows (PowerShell):

**Option 1: Using WinGet (Recommended)**
```powershell
winget install Task.Task
```

**Option 2: Using Chocolatey**
```powershell
choco install go-task
```

**Option 3: Using Scoop**
```powershell
scoop install task
```

**Option 4: Using npm**
```powershell
npm install -g @go-task/cli
```

**Option 5: Using Install Script**
```powershell
# Install to current directory ./bin
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d

# Or install to user directory (add to PATH manually)
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin
```

**Option 6: Download Binary**
Download from https://github.com/go-task/task/releases and add to PATH

**Verify installation:**
```powershell
task --version
```

### Linux (Ubuntu/Debian):

**Option 1: Using Official Repository**
```bash
# Set up repository
curl -1sLf 'https://dl.cloudsmith.io/public/task/task/setup.deb.sh' | sudo -E bash

# Install Task
sudo apt install task
```

**Option 2: Using Homebrew**
```bash
brew install go-task/tap/go-task
```

**Option 3: Using Install Script**
```bash
# Install to ~/.local/bin (add to PATH manually)
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin

# Or install to /usr/local/bin (system-wide)
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin
```

**Verify installation:**
```bash
task --version
```

### macOS:

**Option 1: Using Homebrew (Recommended)**
```bash
brew install go-task/tap/go-task
```

**Option 2: Using Snap**
```bash
sudo snap install task --classic
```

**Option 3: Using npm**
```bash
npm install -g @go-task/cli
```

**Option 4: Using Install Script**
```bash
# Install to ~/.local/bin (add to PATH manually)
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin
```

**Verify installation:**
```bash
task --version
```

### Other Linux Distributions:

**Fedora/CentOS:**
```bash
curl -1sLf 'https://dl.cloudsmith.io/public/task/task/setup.rpm.sh' | sudo -E bash
sudo dnf install task
```

**Arch Linux:**
```bash
pacman -S go-task
```

**Install Script (Works on all platforms):**
```bash
# Install to current directory ./bin
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d

# Install to specific directory
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /path/to/bin
```

**Note:** After installation, make sure the Task binary is in your PATH. You may need to restart your terminal or add the installation directory to your PATH environment variable.

## Configuration

All configuration is stored in `params.yaml`. Edit this file to change training, inference, or Docker settings.

### Key Sections

- **training**: Training hyperparameters (learning rate, batch size, epochs, etc.)
- **inference**: Inference settings (model, checkpoint, reference audio, etc.)
- **docker**: Docker image and container settings
- **paths**: Local and container paths

## Available Tasks

### Check Configuration

Validate and display current configuration:

**Using PowerShell wrapper:**
```powershell
.\task.ps1 check
```

**Using Task runner (if installed):**
```bash
task check
```

This will:
- Validate all required parameters
- Display current training/inference/docker settings
- Verify that all paths exist

### Training

Run training with configuration from `params.yaml`:

**Using PowerShell wrapper:**
```powershell
.\task.ps1 train
```

**Using Task runner:**
```bash
task train
```

Show training command without executing:

**PowerShell:**
```powershell
.\task.ps1 train:dry-run
```

**Task runner:**
```bash
task train:dry-run
```

Override parameters (passed to the training script):

**PowerShell:**
```powershell
.\task.ps1 train -- --epochs 10 --learning_rate 2e-5
```

**Task runner:**
```bash
task train -- --epochs 10 --learning_rate 2e-5
```

### Inference

Run inference with configuration from `params.yaml`:

**Using PowerShell wrapper:**
```powershell
.\task.ps1 infer
```

**Using Task runner:**
```bash
task infer
```

Show inference command without executing:

**PowerShell:**
```powershell
.\task.ps1 infer:dry-run
```

**Task runner:**
```bash
task infer:dry-run
```

Override parameters:

**PowerShell:**
```powershell
.\task.ps1 infer -- --gen_text "Your custom text here"
```

**Task runner:**
```bash
task infer -- --gen_text "Your custom text here"
```

### Jupyter Lab

Start Jupyter Lab using Docker Compose:

**Using PowerShell wrapper:**
```powershell
.\task.ps1 jupyter
```

**Using Task runner:**
```bash
task jupyter
```

Access Jupyter Lab at: http://localhost:8888

Stop Jupyter Lab:

**PowerShell:**
```powershell
.\task.ps1 jupyter:stop
```

**Task runner:**
```bash
task jupyter:stop
```

View logs:

**PowerShell:**
```powershell
.\task.ps1 jupyter:logs
```

**Task runner:**
```bash
task jupyter:logs
```

### Verify Setup

Verify that everything is set up correctly:

**Using PowerShell wrapper:**
```powershell
.\task.ps1 verify
```

**Using Task runner:**
```bash
task verify
```

This checks:
- Docker installation
- Docker image exists
- GPU support in Docker
- Data paths exist
- Dataset directory exists

### Clean

Clean checkpoints and outputs:

**Using PowerShell wrapper:**
```powershell
.\task.ps1 clean --confirm
```

**Using Task runner:**
```bash
task clean --confirm
```

Clean only checkpoints (keep outputs):

**PowerShell:**
```powershell
.\task.ps1 clean --confirm --checkpoints-only
```

**Task runner:**
```bash
task clean:checkpoints --confirm
```

**Note:** The `--confirm` flag is required for safety.

## Examples

### Example 1: Quick Training Run

**Using PowerShell wrapper:**
```powershell
# Check configuration
.\task.ps1 check

# Run training
.\task.ps1 train

# Check for checkpoints
ls ckpts/voxe/
```

**Using Task runner:**
```bash
# Check configuration
task check

# Run training
task train

# Check for checkpoints
ls ckpts/voxe/
```

### Example 2: Custom Training Parameters

Edit `params.yaml`:
```yaml
training:
  epochs: 5
  learning_rate: 2e-5
  batch_size_per_gpu: 1600
```

Then run:

**PowerShell:**
```powershell
.\task.ps1 train
```

**Task runner:**
```bash
task train
```

### Example 3: Test Inference

**PowerShell:**
```powershell
# First, ensure model is trained
.\task.ps1 check

# Run inference
.\task.ps1 infer

# Check output
ls tests/
```

**Task runner:**
```bash
# First, ensure model is trained
task check

# Run inference
task infer

# Check output
ls tests/
```

### Example 4: Workflow with Jupyter

**PowerShell:**
```powershell
# Start Jupyter Lab
.\task.ps1 jupyter

# Work in notebook, then stop
.\task.ps1 jupyter:stop

# Run training from command line
.\task.ps1 train
```

**Task runner:**
```bash
# Start Jupyter Lab
task jupyter

# Work in notebook, then stop
task jupyter:stop

# Run training from command line
task train
```

## Customizing Configuration

### Changing Dataset

Edit `params.yaml`:
```yaml
training:
  dataset_name: your_dataset_name
```

### Changing Model Architecture

Edit `params.yaml`:
```yaml
training:
  exp_name: F5TTS_Base  # or E2TTS_Base
```

### Changing Checkpoint Path

Edit `params.yaml`:
```yaml
inference:
  ckpt_file: ckpts/voxe/model_last.pt
```

## Troubleshooting

### Task command not found

**Option 1:** Use the PowerShell wrapper script instead:
```powershell
.\task.ps1 check
```

**Option 2:** Install Task following the installation instructions above.

### Docker image not found

Build the Docker image:
```bash
docker build -t f5-tts:latest -f Dockerfile .
```

### Path issues on Windows

The `load_params.py` script handles Windows paths automatically. If you encounter issues, ensure paths in `params.yaml` use forward slashes (e.g., `data/voxe` not `data\voxe`).

### Permission errors

On Linux/Mac, you may need to make `scripts/load_params.py` executable:
```bash
chmod +x scripts/load_params.py
```

## Integration with Notebooks

The Taskfile system works alongside Jupyter notebooks. You can:

1. Use `task jupyter` to start Jupyter Lab
2. Use `task check` to verify configuration before training
3. Use `task train` to run training from command line
4. Use `task infer` to test models after training

This provides flexibility to work interactively in notebooks or run automated training from scripts.

