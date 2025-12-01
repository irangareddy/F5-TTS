# Evaluation Framework for Emotional F5TTS

This guide provides structured methods to evaluate your fine-tuned F5TTS model on emotional speech synthesis across intelligibility, emotion fidelity, and speaker consistency.

---

## Part 1: Subjective Listening Evaluation (20-30 min)

### Overview
Listen to generated samples and rate them on three key dimensions:
1. **Intelligibility**: Can you understand the words?
2. **Emotion**: Does the delivery match the target emotion?
3. **Speaker Consistency**: Does the voice remain recognizable across samples?

### Evaluation Rubric

#### A. Intelligibility
**Rating Scale: 1-5**

| Score | Definition | Examples |
|-------|-----------|----------|
| **5** | Perfect clarity | All words clear, natural articulation |
| **4** | Mostly clear | Minor mumbling, but fully understandable |
| **3** | Acceptable | Some words unclear but context preserves meaning |
| **2** | Difficult | Frequent unclear words, requires effort to understand |
| **1** | Unintelligible | Cannot understand most content |

**How to evaluate:**
- Play each sample once at normal volume
- Don't read the transcript before listening
- Note any skipped/mispronounced words
- Average across all samples per emotion

---

#### B. Emotion Matching
**Rating Scale: 1-5** per emotion

##### 🔵 NEUTRAL
- **Target**: Steady pitch, moderate pace, balanced energy, conversational
- **Indicators of Success**:
  - Consistent mid-range pitch (no wide jumps)
  - Natural speech rate (~130-150 words/min)
  - Minimal emotional intensity
  - Balanced loudness
- **Common Failures**:
  - Too breathy/soft
  - Monotone robotic sound
  - Inconsistent pacing

| Score | Neutral Quality |
|-------|-----------------|
| **5** | Clear, natural conversation tone |
| **4** | Mostly neutral, slight emotional lean |
| **3** | Neutral with occasional emotional artifacts |
| **2** | Emotional color evident but not strong |
| **1** | Strong unintended emotion |

---

##### 🔴 ANGRY
- **Target**: High intensity, strained/harsh tone, clipped speech, shorter pauses
- **Indicators of Success**:
  - Elevated pitch (especially on stressed syllables)
  - Rapid delivery, shorter pauses
  - Harsh/strained voice quality
  - Sharp consonants, increased volume
- **Common Failures**:
  - Too soft or gentle
  - Normal pace (not clipped)
  - Smooth prosody (not strained)

| Score | Angry Quality |
|-------|---------------|
| **5** | Intense, strained, aggressive delivery |
| **4** | Clear anger, minor roughness issues |
| **3** | Angry undertone but could be stronger |
| **2** | Slight edge but mostly neutral |
| **1** | No anger present |

---

##### 💛 HAPPY
- **Target**: Bright pitch, high variability, energetic, frequent pitch jumps
- **Indicators of Success**:
  - Brighter overall pitch (higher fundamental frequency)
  - Large pitch jumps and contour variation
  - Quick delivery with upbeat rhythm
  - Warm, resonant voice quality
  - Natural smiling effect
- **Common Failures**:
  - Flat pitch (no variation)
  - Sluggish delivery
  - Dull voice quality
  - Overly synthetic smile

| Score | Happy Quality |
|-------|---------------|
| **5** | Bright, energetic, warm, natural smile |
| **4** | Clearly happy, good pitch variation |
| **3** | Happy tone present, somewhat variable |
| **2** | Slight cheerfulness, mostly neutral |
| **1** | No happiness detected |

---

##### 💙 SAD
- **Target**: Lower pitch, slower pace, softer volume, breathy quality
- **Indicators of Success**:
  - Lower fundamental frequency throughout
  - Slower speech rate (~80-100 words/min)
  - Softer volume, breathy quality
  - Longer pauses between phrases
  - Sagging pitch contour (falling endings)
- **Common Failures**:
  - Too high pitched for sadness
  - Normal or fast pace
  - Too loud/resonant
  - Clipped delivery

| Score | Sad Quality |
|-------|-------------|
| **5** | Low pitch, slow, soft, breathy, drooping intonation |
| **4** | Clearly sad delivery, minor issues |
| **3** | Sad undertone but somewhat subdued |
| **2** | Slight downturn, mostly neutral |
| **1** | No sadness present |

---

##### ⭐ SURPRISE
- **Target**: Widened pitch jumps (especially at phrase start), initial pitch spike, upward inflection
- **Indicators of Success**:
  - Large, sudden pitch jumps (especially at start of utterance)
  - Quick initial burst of energy
  - Raised pitch on exclamatory content
  - Natural startled effect
  - Upward inflection on questions
- **Common Failures**:
  - Smooth, gradual pitch (no sudden spikes)
  - No distinctive energy burst
  - Same as happy (too much variation)
  - Overly exaggerated

| Score | Surprise Quality |
|-------|-----------------|
| **5** | Sudden pitch spikes, clear startled effect |
| **4** | Clear surprise, good pitch jumps |
| **3** | Surprise tone present, somewhat variable |
| **2** | Slight elevation, mostly neutral |
| **1** | No surprise detected |

---

#### C. Speaker Consistency
**Rating Scale: 1-5**

Evaluate whether the **same speaker** maintains consistent vocal characteristics across different prompts and emotions.

**Key Features to Track:**
- **Timbre**: Voice quality/color remains recognizable
- **Resonance**: Same nasal/oral resonance balance
- **Vocal Range**: Fundamental frequency baseline is consistent
- **Accent/Articulation**: Pronunciation patterns stay the same

| Score | Speaker Consistency |
|-------|-------------------|
| **5** | Same speaker across all samples, obvious identity |
| **4** | Clearly same speaker, minor variations |
| **3** | Likely same speaker, some drift in characteristics |
| **2** | Noticeable variation between samples |
| **1** | Different speaker (too much drift) |

**Evaluation Method:**
1. Pick one emotion (e.g., neutral)
2. Listen to all 5 prompts for Speaker 0011 (female)
3. Rate consistency of voice identity
4. Repeat for Speaker 0013 (male)
5. Repeat for another emotion if needed

---

### Evaluation Spreadsheet Template

Create a CSV with this structure:

```
Emotion,Speaker,Prompt,Intelligibility,Emotion_Match,Notes
neutral,0011,01_best_day,5,5,"Clear delivery, natural conversation tone"
neutral,0011,02_couldnt_believe,4,4,"Good clarity, slight tension"
...
angry,0013,01_best_day,5,5,"Intense, strained voice"
...
```

---

## Part 2: Automatic Speech Recognition (ASR) Evaluation

### Whisper-based WER (Word Error Rate)

#### Setup
```bash
pip install openai-whisper
```

#### Evaluation Script

Create `eval_asr.py`:

```python
import whisper
import json
from pathlib import Path
import numpy as np

MODEL_SIZE = "base"  # base, small, medium, large
DEMO_DIR = Path("tests/demos/ckptbest")

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate between reference and hypothesis."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Simple edit distance (WER)
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, ref_words, hyp_words)
    matching_words = sum(block.size for block in matcher.get_matching_blocks())

    wer = 1.0 - (matching_words / max(len(ref_words), 1))
    return wer

def evaluate_demos():
    """Evaluate all demo samples with Whisper."""
    model = whisper.load_model(MODEL_SIZE)

    results = []
    prompts_to_check = [
        "This is the best day of my life.",
        "I can't believe you did that to me.",
        "I don't know what to say anymore.",
        "Did you hear what just happened?",
        "Everything is going to be fine.",
    ]

    for audio_file in sorted(DEMO_DIR.rglob("*.wav")):
        # Transcribe with Whisper
        result = model.transcribe(str(audio_file))
        transcribed = result["text"].strip()

        # Find matching prompt
        emotion = audio_file.parent.parent.name
        speaker = audio_file.parent.name
        prompt_idx = int(audio_file.stem.split("_")[0]) - 1

        if prompt_idx < len(prompts_to_check):
            reference = prompts_to_check[prompt_idx]
            wer = calculate_wer(reference, transcribed)

            results.append({
                "emotion": emotion,
                "speaker": speaker,
                "prompt": reference,
                "transcribed": transcribed,
                "wer": wer,
            })

            status = "✅" if wer < 0.15 else "⚠️" if wer < 0.25 else "❌"
            print(f"{status} {emotion:10} | {speaker} | WER: {wer:.2%}")
            if wer > 0.15:
                print(f"   Expected: {reference}")
                print(f"   Got:      {transcribed}")

    # Summary statistics
    wers = [r["wer"] for r in results]
    print(f"\n{'='*60}")
    print(f"ASR Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total Samples:    {len(results)}")
    print(f"Mean WER:         {np.mean(wers):.2%}")
    print(f"Median WER:       {np.median(wers):.2%}")
    print(f"Max WER:          {np.max(wers):.2%}")
    print(f"Samples <15% WER: {sum(1 for w in wers if w < 0.15)}/{len(results)} ({100*sum(1 for w in wers if w < 0.15)/len(results):.1f}%)")

    # Save results
    with open("eval_asr_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return np.mean(wers)

if __name__ == "__main__":
    mean_wer = evaluate_demos()
    print(f"\nResult: Mean WER = {mean_wer:.2%}")
```

**Run evaluation:**
```bash
python eval_asr.py
```

#### Interpretation

| Mean WER | Quality Assessment |
|----------|-------------------|
| < 5% | Excellent - Near-perfect intelligibility |
| 5-10% | Very Good - Minor pronunciation issues |
| 10-15% | Good - Acceptable intelligibility |
| 15-25% | Fair - Some difficulty understanding |
| > 25% | Poor - Significant comprehension issues |

**Expected Result**: For a well-trained emotional TTS, expect **< 10% WER** (ideally 5-8%).

---

## Part 3: Training Metrics Analysis

### Convergence Documentation

From `RUN_REVIEW.md`, you have:

```
Epoch 6: loss = 1.32
Epoch 7: loss = 0.845 (36% improvement)
Epoch 8: loss = 0.554 (34% improvement)
Epoch 9: Early stop (no improvement)
```

### What This Tells You

✅ **Healthy Convergence**:
- Loss decreased monotonically across epochs
- Large improvements early (epochs 6-7)
- Smaller but consistent improvements later (epochs 7-8)
- Early stopping prevented overfitting

⚠️ **Potential Issues to Monitor**:
- If validation loss plateaus but training loss decreases → overfitting
- If validation loss starts increasing → overfitting already occurred
- If validation loss becomes noisy → batch size or learning rate too high

### Visualization (Optional)

Create a simple plot with:
- X-axis: Epoch (6, 7, 8, 9)
- Y-axis: Validation Loss
- Line showing: 1.32 → 0.845 → 0.554 → STOP

Include in final evaluation report.

---

## Part 4: Emotion Evaluation at Scale (Optional)

### Evaluate on Full Test Set

If you want to evaluate beyond the 50 demo samples:

```bash
# Use your test manifest
ls data/voxe/manifests/test.jsonl
```

**Test set contains**: 1,894 samples across all emotions/speakers

**Quick evaluation approach**:
1. Select 5 random samples per emotion from test.jsonl (25 total)
2. Use same reference voice (speaker 0011 or 0013) if available
3. Rate intelligibility + emotion match
4. Record failure cases for analysis

---

## Part 5: Quick Quality Checklist

Before concluding evaluation, verify:

- [ ] All 50 demo samples generated successfully
- [ ] Intelligibility: Mean score ≥ 4.0/5.0
- [ ] Emotion matching: Each emotion has ≥ 3.5/5.0 average
- [ ] Speaker consistency: ≥ 4.0/5.0 for both speakers
- [ ] No obvious artifacts (clicks, static, robotic sound)
- [ ] Whisper WER: Mean < 10%
- [ ] Convergence: Loss decreased without overfitting

**If any metric falls below targets**:
- Re-examine failing samples in detail
- Check if specific emotion/speaker combinations have issues
- Document failures in eval_results.md for future improvement

---

## Summary: Evaluation Workflow

1. **Generate Demos**: `python generate_demos.py`
2. **Listen Subjectively**: Use rubric above, record scores in CSV
3. **Run ASR**: `python eval_asr.py` (generates WER scores)
4. **Analyze Convergence**: Reference `RUN_REVIEW.md` + create simple convergence plot
5. **Document Results**: Write `eval_results.md` with summary, findings, recommendations

**Estimated Total Time**: 1-2 hours
- 30 min: Demo generation
- 20 min: Subjective listening
- 10 min: ASR evaluation
- 20 min: Analysis + documentation

---

## Example Evaluation Output

After completing evaluation, your `eval_results.md` should contain:

```markdown
# Evaluation Results

## Summary
- 50/50 demo samples generated successfully (100%)
- Mean intelligibility: 4.3/5.0 ✅
- Emotion matching: neutral=4.2, angry=4.5, happy=4.1, sad=4.0, surprise=3.8 ✅
- Speaker consistency: 4.4/5.0 ✅
- Whisper ASR WER: 7.2% ✅

## Key Findings
1. Model successfully captures emotion prosody
2. Happy emotion sounds most natural (best pitch variation)
3. Angry emotion shows good strained voice quality
4. Surprise needs more pitch jump prominence
5. Speaker identity well-preserved across emotions

## Recommendations
- Slight improvement needed for surprise emotion
- Consider training additional emotion (fear/disgust) if needed
- Ready for production use for neutral/angry/happy/sad
```

Good luck with your evaluation! 🎯
