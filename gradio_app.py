#!/usr/bin/env python3
"""
Gradio web interface for F5-TTS Text-to-Speech inference.
"""

import os
import sys
import torch
import torchaudio
import gradio as gr
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from f5_tts.infer.infer_cli import F5TTS, get_tokenizer, load_vocoder, mel_to_audio
from f5_tts.model.utils import get_mel_spectrogram


class TTSInference:
    def __init__(self,
                 ckpt_path: str = "ckpts/voxe/model_last.pt",
                 vocab_file: str = "data/voxe_char/vocab.txt",
                 ref_audio: str = "data/voxe/wav24k/esd/0011/Neutral/0011_000001.wav",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize TTS inference engine.

        Args:
            ckpt_path: Path to fine-tuned model checkpoint
            vocab_file: Path to vocabulary file
            ref_audio: Path to reference audio for speaker embedding
            device: Device to run inference on (cuda or cpu)
        """
        self.device = device
        self.ckpt_path = ckpt_path
        self.vocab_file = vocab_file
        self.ref_audio = ref_audio

        print(f"[TTS] Using device: {device}")
        print(f"[TTS] Loading model from: {ckpt_path}")

        # Load model
        self.model = F5TTS()
        self.model = self.model.to(device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        print(f"[TTS] Model loaded successfully")

        # Load tokenizer
        self.vocab_char_map, self.vocab_size = get_tokenizer(vocab_file, "char")
        print(f"[TTS] Tokenizer loaded (vocab size: {self.vocab_size})")

        # Load vocoder
        self.vocoder = load_vocoder(vocoder_name="vocos")
        print(f"[TTS] Vocoder loaded")

        # Load reference audio
        self.ref_audio_data, sr = torchaudio.load(ref_audio)
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            self.ref_audio_data = resampler(self.ref_audio_data)
        self.ref_audio_data = self.ref_audio_data.to(device)
        print(f"[TTS] Reference audio loaded")

    def text_to_speech(self, text: str, speed: float = 1.0) -> tuple:
        """
        Generate speech from text.

        Args:
            text: Input text to synthesize
            speed: Speech speed multiplier (0.5-2.0)

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            print(f"\n[Inference] Generating speech for: '{text}'")

            # Tokenize text
            tokens = [self.vocab_char_map.get(c, 0) for c in text]
            tokens = torch.tensor([tokens], dtype=torch.long).to(self.device)

            # Get reference mel-spectrogram
            ref_mel = get_mel_spectrogram(self.ref_audio_data)

            # Generate mel-spectrogram
            with torch.no_grad():
                mel_spec = self.model.inference(
                    tokens=tokens,
                    ref_mel=ref_mel,
                    speed=speed
                )

            # Convert mel-spectrogram to audio
            audio = mel_to_audio(mel_spec, vocoder=self.vocoder)

            print(f"[Inference] Generated audio: {audio.shape}")

            return (24000, audio.cpu().numpy())

        except Exception as e:
            print(f"[Error] Failed to generate speech: {str(e)}")
            raise


def create_interface(tts_engine: TTSInference) -> gr.Interface:
    """Create Gradio interface."""

    def generate_speech(text, speed):
        """Wrapper for Gradio."""
        if not text or not text.strip():
            return None, "Please enter some text!"

        try:
            sr, audio = tts_engine.text_to_speech(text, speed=speed)
            return (sr, audio), f"Generated {len(audio)} samples at {sr}Hz"
        except Exception as e:
            return None, f"Error: {str(e)}"

    # Create interface
    iface = gr.Interface(
        fn=generate_speech,
        inputs=[
            gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter text here...",
                lines=3,
                value="This is a test of the fine-tuned F5-TTS model on the VOXE dataset."
            ),
            gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Speech Speed"
            )
        ],
        outputs=[
            gr.Audio(label="Generated Speech"),
            gr.Textbox(label="Status")
        ],
        title="F5-TTS: Fine-tuned Text-to-Speech",
        description="Generate natural speech from text using the F5-TTS model fine-tuned on VOXE dataset.",
        examples=[
            ["Hello, this is a test of the fine-tuned model.", 1.0],
            ["The quick brown fox jumps over the lazy dog.", 1.0],
            ["How are you doing today?", 0.8],
            ["I am very excited about this new model!", 1.2],
        ],
        theme=gr.themes.Soft()
    )

    return iface


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="F5-TTS Gradio Interface")
    parser.add_argument("--ckpt", default="ckpts/voxe/model_last.pt", help="Path to checkpoint")
    parser.add_argument("--vocab", default="data/voxe_char/vocab.txt", help="Path to vocab file")
    parser.add_argument("--ref-audio", default="data/voxe/wav24k/esd/0011/Neutral/0011_000001.wav", help="Path to reference audio")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Check paths
    if not Path(args.ckpt).exists():
        print(f"Error: Checkpoint not found: {args.ckpt}")
        sys.exit(1)
    if not Path(args.vocab).exists():
        print(f"Error: Vocab file not found: {args.vocab}")
        sys.exit(1)
    if not Path(args.ref_audio).exists():
        print(f"Error: Reference audio not found: {args.ref_audio}")
        sys.exit(1)

    # Initialize TTS engine
    print("Initializing TTS engine...")
    tts_engine = TTSInference(
        ckpt_path=args.ckpt,
        vocab_file=args.vocab,
        ref_audio=args.ref_audio,
        device=args.device
    )

    # Create and launch interface
    print(f"\nLaunching Gradio interface on {args.host}:{args.port}")
    iface = create_interface(tts_engine)
    iface.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        show_api=False
    )


if __name__ == "__main__":
    main()
