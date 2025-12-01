#!/usr/bin/env python3
"""
Generate 5 Emotion Key Samples using F5TTS infer_cli
Runs inference via Docker (default) or directly for emotion-specific evaluation samples.

Usage:
    python generate_emotion_samples.py [--ckpt_file path/to/checkpoint.pt] [--output_dir tests/emotion_samples]
    
    # Use Docker (default, recommended)
    python generate_emotion_samples.py
    
    # Run directly on host (requires local dependencies)
    python generate_emotion_samples.py --no-docker
"""

import os
import sys
import json
import subprocess
import argparse
import shutil
import time
import csv
import re
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from tqdm import tqdm

# Import load_params from scripts
sys.path.insert(0, str(Path(__file__).parent))
try:
    from scripts.load_params import load_params
except ImportError:
    load_params = None

# Fix encoding on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available in the system PATH."""
    return shutil.which("ffmpeg") is not None


def check_pydub_ffmpeg() -> bool:
    """Check if pydub can access ffmpeg."""
    try:
        from pydub.utils import which
        return which("ffmpeg") is not None
    except Exception:
        return False


def find_ffmpeg_installation():
    """Try to find ffmpeg installation in common locations."""
    if sys.platform == "win32":
        # Common Windows installation locations
        common_paths = [
            # Winget installation location
            os.path.join(os.environ.get("LOCALAPPDATA", ""), 
                        "Microsoft", "WinGet", "Packages"),
            # Program Files
            "C:\\Program Files\\ffmpeg\\bin",
            "C:\\Program Files (x86)\\ffmpeg\\bin",
            # Common user installation
            os.path.join(os.environ.get("USERPROFILE", ""), "ffmpeg", "bin"),
        ]
        
        # Search in WinGet packages
        winget_packages = os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Microsoft", "WinGet", "Packages"
        )
        if os.path.exists(winget_packages):
            try:
                for root, dirs, files in os.walk(winget_packages):
                    if "ffmpeg.exe" in files:
                        bin_dir = os.path.dirname(os.path.join(root, "ffmpeg.exe"))
                        if bin_dir not in common_paths:
                            common_paths.insert(0, bin_dir)
                        break
            except Exception:
                pass
        
        # Check each path
        for path in common_paths:
            if path and os.path.exists(path):
                ffmpeg_exe = os.path.join(path, "ffmpeg.exe")
                if os.path.exists(ffmpeg_exe):
                    return path
    else:
        # Unix-like systems
        common_paths = [
            "/usr/bin",
            "/usr/local/bin",
            "/opt/ffmpeg/bin",
        ]
        for path in common_paths:
            if os.path.exists(os.path.join(path, "ffmpeg")):
                return path
    
    return None


def check_docker() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def verify_ffmpeg_installation():
    """Verify ffmpeg is installed and accessible, exit with helpful message if not."""
    if not check_ffmpeg():
        # Try to find ffmpeg installation
        ffmpeg_bin = find_ffmpeg_installation()
        
        if ffmpeg_bin:
            # Add to PATH for this session
            print(f"\n📦 Found ffmpeg installation at: {ffmpeg_bin}")
            print("   Adding to PATH for this session...")
            os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")
            
            # Verify it works now
            if check_ffmpeg():
                print("   ✅ ffmpeg is now accessible!\n")
                return
            else:
                print("   ⚠️  Warning: Could not add ffmpeg to PATH\n")
        
        # If still not found, show error message
        print("\n" + "="*70)
        print("❌ ERROR: ffmpeg is not installed or not in PATH")
        print("="*70)
        print("\nThe pydub library requires ffmpeg to process audio files.")
        print("\nInstallation options:")
        print("\nOption 1: Install using Chocolatey (recommended for Windows)")
        print("  choco install ffmpeg")
        print("\nOption 2: Install using winget")
        print("  winget install ffmpeg")
        print("\nOption 3: Manual installation")
        print("  1. Download from: https://ffmpeg.org/download.html")
        print("  2. Extract to a folder (e.g., C:\\ffmpeg)")
        print("  3. Add C:\\ffmpeg\\bin to your system PATH")
        print("  4. Restart your terminal/PowerShell")
        print("\nAfter installation, verify with:")
        print("  ffmpeg -version")
        print("\n" + "="*70 + "\n")
        sys.exit(1)
    
    # Also check if pydub can access it
    if not check_pydub_ffmpeg():
        print("\n" + "="*70)
        print("⚠️  WARNING: ffmpeg found in PATH but pydub cannot access it")
        print("="*70)
        print("\nThis may cause issues during audio processing.")
        print("Try restarting your terminal/PowerShell after installation.")
        print("\n" + "="*70 + "\n")
    
    # Try to verify ffmpeg works
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            print("\n⚠️  WARNING: ffmpeg is installed but may not be working correctly.\n")
    except Exception:
        pass  # Non-critical check

# Emotion-Specific Key Samples
EMOTION_SAMPLES = {
    "neutral": "The meeting starts at 10 am in Room 204.",
    "angry": "Seriously? You ignored the plan and broke everything.",
    "happy": "Yes! We nailed it—this feels amazing!",
    "sad": "I'm trying my best, but today feels heavy.",
    "surprise": "Whoa—wait, what? I didn't see that coming!",
}

# Reference voices
REFERENCE_VOICES = {
    "female": "0019",
    "male": "0014",
}

# Emotion-to-dataset mapping
EMOTION_MAP = {
    "neutral": "Neutral",
    "angry": "Angry",
    "happy": "Happy",
    "sad": "Sad",
    "surprise": "Surprise",
}


def find_reference_audio(
    emotion: str, speaker_id: str, dataset_root: str = "data/voxe/wav24k"
) -> Tuple[str, str]:
    """Find reference audio for emotion and speaker."""
    emotion_key = EMOTION_MAP.get(emotion, emotion.capitalize())
    audio_dir = Path(dataset_root) / "esd" / speaker_id / emotion_key

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    wav_files = list(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")

    audio_file = wav_files[0]

    # Get transcription from manifest
    manifest_path = Path("data/voxe/manifests/all.jsonl")
    ref_text = "the nine the eggs, i keep."

    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    if str(audio_file).replace("\\", "/") in entry.get("wav_path", ""):
                        ref_text = entry.get("text", ref_text)
                        break
        except Exception as e:
            print(f"  Warning: Could not read manifest: {e}")

    return str(audio_file), ref_text


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Sanitize text for use in filename.
    
    Args:
        text: Text to sanitize
        max_length: Maximum length of filename (excluding extension)
    
    Returns:
        Sanitized filename-safe string
    """
    # Take first 3-4 words
    words = text.split()[:4]
    # Join with underscores, lowercase
    filename = "_".join(words).lower()
    # Remove special characters except underscores
    filename = re.sub(r'[^a-z0-9_]', '', filename)
    # Limit length
    if len(filename) > max_length:
        filename = filename[:max_length]
    return filename


def load_docker_config() -> Optional[dict]:
    """Load Docker configuration from params.yaml."""
    if load_params is None:
        return None
    
    try:
        params = load_params("params.yaml")
        return params
    except Exception:
        return None


def build_docker_inference_command(
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    output_path: str,
    ckpt_file: str,
    model: str,
    params: dict,
) -> list:
    """Build Docker command for inference."""
    docker = params["docker"]
    paths = params["paths"]
    
    # Get current working directory and normalize Windows paths
    pwd = os.getcwd().replace("\\", "/")
    
    # Convert absolute paths to relative paths from project root, then to container paths
    # ref_audio: convert to relative path, then to container path
    ref_audio_path = Path(ref_audio)
    if ref_audio_path.is_absolute():
        # Try to make it relative to project root
        try:
            ref_audio_rel = ref_audio_path.relative_to(Path.cwd())
        except ValueError:
            # If not under project root, use as-is (will need to be mounted)
            ref_audio_rel = ref_audio_path
    else:
        ref_audio_rel = ref_audio_path
    
    # Map to container path based on which volume it's under
    ref_audio_str = str(ref_audio_rel).replace("\\", "/")
    if ref_audio_str.startswith(paths["data"]):
        ref_audio_container = ref_audio_str.replace(paths["data"], f"{docker['workdir']}/data", 1)
    else:
        ref_audio_container = f"{docker['workdir']}/{ref_audio_str}"
    
    # ckpt_file: similar conversion
    ckpt_file_path = Path(ckpt_file)
    if ckpt_file_path.is_absolute():
        try:
            ckpt_file_rel = ckpt_file_path.relative_to(Path.cwd())
        except ValueError:
            ckpt_file_rel = ckpt_file_path
    else:
        ckpt_file_rel = ckpt_file_path
    
    ckpt_file_str = str(ckpt_file_rel).replace("\\", "/")
    if ckpt_file_str.startswith(paths["ckpts"]):
        ckpt_file_container = ckpt_file_str.replace(paths["ckpts"], f"{docker['workdir']}/ckpts", 1)
    else:
        ckpt_file_container = f"{docker['workdir']}/{ckpt_file_str}"
    
    # output_path: similar conversion
    output_path_obj = Path(output_path)
    if output_path_obj.is_absolute():
        try:
            output_path_rel = output_path_obj.relative_to(Path.cwd())
        except ValueError:
            output_path_rel = output_path_obj
    else:
        output_path_rel = output_path_obj
    
    output_path_str = str(output_path_rel).replace("\\", "/")
    output_dir_container = str(output_path_rel.parent).replace("\\", "/")
    if output_dir_container.startswith(paths["tests"]):
        output_dir_container = output_dir_container.replace(paths["tests"], f"{docker['workdir']}/tests", 1)
    elif output_dir_container.startswith(paths["ckpts"]):
        output_dir_container = output_dir_container.replace(paths["ckpts"], f"{docker['workdir']}/ckpts", 1)
    else:
        output_dir_container = f"{docker['workdir']}/{output_dir_container}"
    
    output_file_name = output_path_rel.name
    
    # Build Docker command
    cmd = [
        "docker", "run", "--rm", "--gpus", docker["gpus"],
        "-v", f"{pwd}/{paths['data']}:{docker['workdir']}/data:ro",
        "-v", f"{pwd}/{paths['ckpts']}:{docker['workdir']}/ckpts:ro",
        "-v", f"{pwd}/{paths['src']}:{docker['workdir']}/src",
        "-v", f"{pwd}/{paths['tests']}:{docker['workdir']}/tests",
        "-w", docker["workdir"],
        docker["image"],
        "python", f"{docker['workdir']}/src/f5_tts/infer/infer_cli.py",
        "-m", model,
        "-p", ckpt_file_container,
        "-r", ref_audio_container,
        "-s", ref_text,
        "-t", gen_text,
        "-o", output_dir_container,
        "-w", output_file_name,
        "--nfe_step", "32",
        "--cfg_strength", "1.2",
        "--speed", "1.0",
    ]
    
    return cmd


def run_inference(
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    output_path: str,
    ckpt_file: str,
    model: str = "F5TTS_v1_Base",
    use_docker: bool = True,
    params: Optional[dict] = None,
) -> Tuple[bool, str]:
    """Run F5TTS inference via infer_cli (Docker or direct).
    
    Args:
        ref_audio: Path to reference audio file
        ref_text: Reference text
        gen_text: Text to generate
        output_path: Output file path
        ckpt_file: Path to checkpoint file
        model: Model name
        use_docker: Whether to use Docker (default: True)
        params: Docker params dict (loaded if None and use_docker=True)
    
    Returns:
        tuple: (success: bool, error_message: str)
    """
    # Build command
    if use_docker:
        if params is None:
            params = load_docker_config()
            if params is None:
                return False, "Docker mode enabled but params.yaml not found or invalid"
        
        if not check_docker():
            return False, "Docker is not available. Install Docker or use --no-docker flag"
        
        cmd = build_docker_inference_command(
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            output_path=output_path,
            ckpt_file=ckpt_file,
            model=model,
            params=params,
        )
    else:
        cmd = [
            "python",
            "src/f5_tts/infer/infer_cli.py",
            "-m", model,
            "-p", ckpt_file,
            "-r", ref_audio,
            "-s", ref_text,
            "-t", gen_text,
            "-o", str(Path(output_path).parent),
            "-w", Path(output_path).name,
            "--nfe_step", "32",
            "--cfg_strength", "1.2",
            "--speed", "1.0",
        ]

    try:
        # Don't print here - progress bar handles it
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            # Get full error messages
            stderr_lines = result.stderr.split('\n') if result.stderr else []
            stdout_lines = result.stdout.split('\n') if result.stdout else []
            
            # Filter out warnings and find actual errors
            error_lines = []
            warning_lines = []
            
            for line in stderr_lines + stdout_lines:
                line_lower = line.lower()
                # Skip warnings
                if any(w in line_lower for w in ['warning', 'deprecated', 'pkg_resources', 'jieba']):
                    warning_lines.append(line)
                    continue
                # Look for actual errors
                if any(e in line_lower for e in ['error', 'exception', 'traceback', 'failed', 'cannot', 'not found']):
                    error_lines.append(line)
            
            # If no specific errors found, show last few lines of stderr
            if not error_lines:
                error_lines = stderr_lines[-10:] if stderr_lines else stdout_lines[-10:]
            
            # Format error message
            if error_lines:
                error_msg = '\n'.join(error_lines[:20])  # Show up to 20 lines
                # Remove very long lines
                error_msg = '\n'.join(line[:200] + '...' if len(line) > 200 else line for line in error_msg.split('\n'))
            else:
                error_msg = result.stderr[:500] if result.stderr else result.stdout[:500]
            
            # Check for ffmpeg-related errors
            if "ffmpeg" in error_msg.lower() or "couldn't find ffmpeg" in error_msg.lower():
                return False, "ffmpeg is required but not found"
            
            return False, error_msg or "Unknown error occurred"

        if Path(output_path).exists():
            return True, ""
        else:
            # Check if there's any error output even if returncode is 0
            if result.stderr and any('error' in line.lower() for line in result.stderr.split('\n')):
                error_msg = result.stderr[:500] if result.stderr else "Output file not created"
                return False, error_msg
            return False, "Output file not created"

    except subprocess.TimeoutExpired:
        return False, "Timeout (>300s)"
    except Exception as e:
        error_str = str(e).lower()
        if "ffmpeg" in error_str:
            return False, "ffmpeg is required but not found"
        return False, f"Exception: {str(e)}"


def generate_emotion_samples(
    ckpt_file: str,
    output_dir: str = "tests/emotion_samples",
    voices: list = None,
    use_docker: bool = True,
) -> Dict[str, int]:
    """
    Generate 5 emotion key samples.

    Args:
        ckpt_file: Path to model checkpoint
        output_dir: Output directory
        voices: List of voices to generate (default: both)

    Returns:
        Statistics dict
    """
    if voices is None:
        voices = list(REFERENCE_VOICES.keys())

    if not Path(ckpt_file).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    ckpt_tag = Path(ckpt_file).stem.replace("model_", "")
    base_output = Path(output_dir) / ckpt_tag

    stats = {"total": 0, "success": 0, "failed": 0}
    total_samples = len(EMOTION_SAMPLES) * len(voices)
    start_time = time.time()
    
    # Create error log file
    error_log_file = base_output / "error_log.txt"
    error_log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Metadata collection for CSV and JSON
    samples_metadata: List[Dict] = []

    # Load Docker params if using Docker
    docker_params = None
    if use_docker:
        docker_params = load_docker_config()
        if docker_params is None:
            print(f"\n⚠️  WARNING: Docker mode enabled but params.yaml not found or invalid")
            print(f"   Falling back to direct execution mode...")
            use_docker = False

    print(f"\n{'='*70}")
    print(f"F5TTS Emotion Key Samples Generation")
    print(f"{'='*70}")
    print(f"Mode:     {'Docker' if use_docker else 'Direct'}")
    print(f"Checkpoint: {ckpt_file}")
    print(f"Output Dir: {base_output}")
    print(f"Emotions: {len(EMOTION_SAMPLES)}")
    print(f"Voices: {', '.join(voices)}")
    print(f"Total Samples: {len(EMOTION_SAMPLES)} × {len(voices)} = {total_samples}")
    print(f"{'='*70}\n")

    # Create progress bar with custom format showing percentage and timer
    progress_bar = tqdm(
        total=total_samples,
        desc="Generating",
        unit="sample",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ncols=100,
        colour="green",
        dynamic_ncols=True
    )

    try:
        for emotion_idx, (emotion, prompt) in enumerate(EMOTION_SAMPLES.items(), 1):
            emotion_status = f"[{emotion_idx}/{len(EMOTION_SAMPLES)}] {emotion.upper()}"
            
            for voice_type in voices:
                if voice_type not in REFERENCE_VOICES:
                    progress_bar.write(f"  ❌ Unknown voice type: {voice_type}")
                    continue

                speaker_id = REFERENCE_VOICES[voice_type]
                current_task = f"{emotion_status} | {voice_type.capitalize()} (Speaker {speaker_id})"
                
                # Update progress bar description with current task
                progress_bar.set_description(f"{current_task}")

                try:
                    ref_audio, ref_text = find_reference_audio(emotion, speaker_id)
                except FileNotFoundError as e:
                    progress_bar.write(f"  ❌ {e}")
                    stats["failed"] += 1
                    stats["total"] += 1
                    progress_bar.update(1)
                    continue

                output_emotion_dir = base_output / emotion / speaker_id
                output_emotion_dir.mkdir(parents=True, exist_ok=True)

                # Generate descriptive filename
                text_snippet = sanitize_filename(prompt)
                filename = f"{speaker_id}_{emotion}_{text_snippet}.wav"
                output_file = output_emotion_dir / filename

                stats["total"] += 1

                # Run inference
                success, error_msg = run_inference(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    gen_text=prompt,
                    output_path=str(output_file),
                    ckpt_file=ckpt_file,
                    use_docker=use_docker,
                    params=docker_params,
                )
                
                # Calculate relative file path from base_output
                file_path_rel = f"{emotion}/{speaker_id}/{filename}"
                
                # Collect metadata
                sample_meta = {
                    "emotion": emotion,
                    "speaker_id": speaker_id,
                    "speaker_type": voice_type.capitalize(),
                    "file_path": file_path_rel,
                    "file_name": filename,
                    "text": prompt,
                    "ref_audio": str(Path(ref_audio).as_posix()),
                    "ref_text": ref_text,
                    "emotion_index": emotion_idx,
                    "speaker_index": list(voices).index(voice_type) + 1 if voice_type in voices else 0,
                    "success": success,
                }
                
                if success:
                    stats["success"] += 1
                    progress_bar.write(f"  ✅ {emotion.upper()} | {voice_type.capitalize()}: {filename}")
                else:
                    stats["failed"] += 1
                    progress_bar.write(f"  ❌ {emotion.upper()} | {voice_type.capitalize()}: Failed")
                    if error_msg:
                        # Show first line of error message, truncate if too long
                        error_first_line = error_msg.split('\n')[0]
                        if len(error_first_line) > 150:
                            error_first_line = error_first_line[:150] + "..."
                        progress_bar.write(f"      Error: {error_first_line}")
                        
                        # Log full error to file
                        with open(error_log_file, "a", encoding="utf-8") as f:
                            f.write(f"\n{'='*70}\n")
                            f.write(f"Failed: {emotion.upper()} | {voice_type.capitalize()}\n")
                            f.write(f"Output: {output_file}\n")
                            f.write(f"Error:\n{error_msg}\n")
                            f.write(f"{'='*70}\n")
                
                samples_metadata.append(sample_meta)
                
                # Update progress bar
                progress_bar.update(1)
                
                # Update postfix with stats (shown after the progress bar)
                elapsed = time.time() - start_time
                success_rate = (100 * stats['success'] / stats['total']) if stats['total'] > 0 else 0
                progress_bar.set_postfix({
                    "✅": stats['success'],
                    "❌": stats['failed'],
                    "Rate": f"{success_rate:.1f}%"
                })
    finally:
        progress_bar.close()

    # Generate CSV and JSON files
    csv_file = base_output / "text.csv"
    json_file = base_output / "samples.json"
    
    # Write CSV file
    if samples_metadata:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "emotion", "speaker_id", "speaker_type", "file_path", 
                "text", "ref_audio", "ref_text"
            ])
            writer.writeheader()
            for meta in samples_metadata:
                writer.writerow({
                    "emotion": meta["emotion"],
                    "speaker_id": meta["speaker_id"],
                    "speaker_type": meta["speaker_type"],
                    "file_path": meta["file_path"],
                    "text": meta["text"],
                    "ref_audio": meta["ref_audio"],
                    "ref_text": meta["ref_text"],
                })
    
    # Write JSON file (only successful samples)
    successful_samples = [meta for meta in samples_metadata if meta.get("success", False)]
    if successful_samples:
        # Remove success field for JSON output
        json_data = []
        for meta in successful_samples:
            json_entry = {k: v for k, v in meta.items() if k != "success"}
            json_data.append(json_entry)
        
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

    # Summary
    total_elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Generation Complete!")
    print(f"{'='*70}")
    print(f"Total:    {stats['total']}")
    print(f"✅ Success: {stats['success']}")
    print(f"❌ Failed:  {stats['failed']}")
    if stats['total'] > 0:
        print(f"Rate:     {100 * stats['success'] / stats['total']:.1f}%")
    print(f"Time:     {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Output:   {base_output}")
    if stats['failed'] > 0:
        print(f"\n⚠️  Full error details saved to: {error_log_file}")
    if samples_metadata:
        print(f"\n📄 Documentation files:")
        print(f"   CSV:  {csv_file}")
        print(f"   JSON: {json_file}")
    print(f"{'='*70}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate 5 emotion key samples for F5TTS evaluation"
    )
    parser.add_argument(
        "--ckpt_file",
        type=str,
        default="ckpts/F5TTS_v1_Base_voxe_manifest/model_best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tests/emotion_samples",
        help="Output directory",
    )
    parser.add_argument(
        "--voices",
        type=str,
        nargs="+",
        default=None,
        help="Voice types: female, male, or both",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Disable Docker and run directly (requires local dependencies). Default: use Docker",
    )

    args = parser.parse_args()
    
    # Use Docker by default, unless --no-docker is specified
    use_docker = not args.no_docker
    
    # Only verify ffmpeg if not using Docker
    if not use_docker:
        verify_ffmpeg_installation()

    try:
        stats = generate_emotion_samples(
            ckpt_file=args.ckpt_file,
            output_dir=args.output_dir,
            voices=args.voices,
            use_docker=use_docker,
        )
        sys.exit(0 if stats["failed"] == 0 else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
