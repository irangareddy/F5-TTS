#!/usr/bin/env powershell
# Test the fine-tuned VOXE model using inference CLI
# This script runs inference to generate audio from the fine-tuned model

Write-Host "Testing fine-tuned VOXE model..." -ForegroundColor Green

docker run --rm --gpus all `
  -v "${PWD}/data:/workspace/F5-TTS/data:ro" `
  -v "${PWD}/ckpts:/workspace/F5-TTS/ckpts:ro" `
  -v "${PWD}/src:/workspace/F5-TTS/src" `
  -v "${PWD}/test_voxe_model.toml:/workspace/F5-TTS/test_voxe_model.toml" `
  -w /workspace/F5-TTS `
  f5-tts:latest `
  python src/f5_tts/infer/infer_cli.py `
    -c test_voxe_model.toml

Write-Host "`nTest complete! Check the output in tests/voxe_finetuned_test.wav" -ForegroundColor Green

