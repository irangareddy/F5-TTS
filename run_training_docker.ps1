# Run F5-TTS fine-tuning in Docker
# This script runs the fine-tuning command inside the Docker container
# Note: --exp_name selects the model architecture, --dataset_name determines output directory

docker run --rm --gpus all `
  --shm-size=2g `
  -v "${PWD}/data:/workspace/F5-TTS/data:ro" `
  -v "${PWD}/ckpts:/workspace/F5-TTS/ckpts" `
  -v "${PWD}/src:/workspace/F5-TTS/src" `
  -w /workspace/F5-TTS `
  f5-tts:latest `
  python src/f5_tts/train/finetune_cli.py `
    --exp_name F5TTS_v1_Base `
    --dataset_name voxe `
    --learning_rate 1e-5 `
    --batch_size_per_gpu 800 `
    --batch_size_type frame `
    --max_samples 8 `
    --epochs 2 `
    --num_warmup_updates 100 `
    --save_per_updates 50 `
    --keep_last_n_checkpoints 3 `
    --finetune `
    --tokenizer char

