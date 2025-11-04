# Testing the Fine-Tuned VOXE Model

## Quick Start

To test your fine-tuned model, use the provided test script:

```powershell
.\test_voxe_model.ps1
```

Or run the inference command directly:

```powershell
docker run --rm --gpus all `
  -v "${PWD}/data:/workspace/F5-TTS/data:ro" `
  -v "${PWD}/ckpts:/workspace/F5-TTS/ckpts:ro" `
  -v "${PWD}/src:/workspace/F5-TTS/src" `
  -v "${PWD}/test_voxe_model.toml:/workspace/F5-TTS/test_voxe_model.toml" `
  -v "${PWD}/tests:/workspace/F5-TTS/tests" `
  -w /workspace/F5-TTS `
  f5-tts:latest `
  python src/f5_tts/infer/infer_cli.py -c test_voxe_model.toml
```

## Configuration

The test configuration is in `test_voxe_model.toml`:

- **Model**: F5TTS_v1_Base
- **Checkpoint**: `ckpts/voxe/model_last.pt` (your fine-tuned model)
- **Vocab**: `data/voxe_char/vocab.txt` (Emilia vocab with 2545 tokens)
- **Reference Audio**: `data/voxe/wav24k/esd/0011/Angry/0011_000351.wav`
- **Output**: `tests/voxe_finetuned_test.wav`

## Customizing the Test

### Change the Reference Audio

Edit `test_voxe_model.toml` and update:
```toml
ref_audio = "data/voxe/wav24k/esd/0011/Angry/0011_000351.wav"
ref_text = "the nine the eggs, i keep."
```

### Change the Generated Text

Edit `test_voxe_model.toml` and update:
```toml
gen_text = "Your custom text here."
```

### Use Different Emotions/Speakers

You can test with different emotions by changing the reference audio path:
- Angry: `data/voxe/wav24k/esd/0011/Angry/0011_000351.wav`
- Happy: `data/voxe/wav24k/esd/0011/Happy/0011_000351.wav`
- Sad: `data/voxe/wav24k/esd/0011/Sad/0011_000351.wav`
- Neutral: `data/voxe/wav24k/esd/0011/Neutral/0011_000351.wav`
- Surprise: `data/voxe/wav24k/esd/0011/Surprise/0011_000351.wav`

## Direct CLI Usage

You can also use the CLI directly with command-line arguments:

```powershell
docker run --rm --gpus all `
  -v "${PWD}/data:/workspace/F5-TTS/data:ro" `
  -v "${PWD}/ckpts:/workspace/F5-TTS/ckpts:ro" `
  -v "${PWD}/src:/workspace/F5-TTS/src" `
  -v "${PWD}/tests:/workspace/F5-TTS/tests" `
  -w /workspace/F5-TTS `
  f5-tts:latest `
  python src/f5_tts/infer/infer_cli.py `
    --model F5TTS_v1_Base `
    --ckpt_file ckpts/voxe/model_last.pt `
    --vocab_file data/voxe_char/vocab.txt `
    --ref_audio "data/voxe/wav24k/esd/0011/Angry/0011_000351.wav" `
    --ref_text "the nine the eggs, i keep." `
    --gen_text "This is a test of the fine-tuned F5-TTS model." `
    --output_dir tests `
    --output_file my_test.wav
```

## Expected Output

After running inference, you should see:
- Generated audio file: `tests/voxe_finetuned_test.wav`
- The audio should match the voice characteristics from the reference audio
- The text should be synthesized with similar emotion/style as the reference

## Troubleshooting

1. **If checkpoint loading fails**: Make sure `model_last.pt` exists in `ckpts/voxe/`
2. **If vocab issues**: Ensure `data/voxe_char/vocab.txt` exists and matches the training vocab
3. **If audio is silent**: Check that the reference audio file exists and is readable
4. **If CUDA errors**: Ensure GPU is available: `docker run --rm --gpus all nvidia-smi`

