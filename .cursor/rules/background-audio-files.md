---
description: Guidelines for adding background noise and burst audio files
globs:
  - data/voice/background_noise_audio_pcm_mono_verified/**
  - src/tau2/voice/utils/convert_audio_files.py
---

# Adding Background Noise Audio Files

Guidelines for adding new audio files to `data/voice/background_noise_audio_pcm_mono_verified/`.

## Format Requirements

### Hard Requirements (Validated in Code)

| Property | Requirement | What Happens If Violated |
|----------|-------------|--------------------------|
| File format | WAV (`.wav`) | Load fails |
| Channels | **Mono (1 channel)** | `ValueError` raised |

Source: `noise_generator.py:121-124` validates mono requirement.

### Recommended Format (Auto-Converted at Runtime)

| Property | Recommended | Notes |
|----------|-------------|-------|
| Encoding | PCM 16-bit signed (`PCM_S16LE`) | Other encodings auto-converted |
| Sample rate | **16000 Hz (16kHz)** | Other rates auto-resampled |

**Why 16kHz?** Upsampling (from lower rates) and downsampling (from higher rates) can introduce artifacts. Files below 16kHz will degrade audio quality. Use the conversion tool to pre-convert files to 16kHz to avoid runtime resampling.

## Directory Structure

Full path: `data/voice/background_noise_audio_pcm_mono_verified/`

| Type | Directory | Description |
|------|-----------|-------------|
| **Continuous** | `continuous/` | Looping ambient sounds (busy street, TV, people talking) |
| **Burst** | `bursts/` | Short, distinct sounds (car horn, dog bark, siren, phone ring) |

### Recommended Durations (Not Validated in Code)

| Type | Recommended Duration | Reason |
|------|---------------------|--------|
| Continuous | At least 30 seconds | File is constantly looped; shorter files create noticeable repetition |
| Burst | Max 5 seconds | Burst sounds should be brief interruptions, not extended events |

## Conversion Workflow

Use the existing `convert_audio_files.py` utility to pre-convert files:

```bash
python -m tau2.voice.utils.convert_audio_files \
    /path/to/input/dir \
    /path/to/output/dir \
    --sample-rate 16000
```

This converts to PCM 16-bit mono at 16kHz, avoiding runtime conversion overhead and resampling artifacts.

## Supported Input Formats

The conversion tool can accept:

- PCM WAV (8/16/24/32-bit)
- μ-law (G.711) WAV
- A-law WAV
- Stereo (will be converted to mono)

## Content Guidelines

- Files are automatically normalized to prevent clipping
- The system applies SNR-based mixing, so raw volume matters less than audio quality
- For continuous files: seamless loops are ideal but not required (file is long enough that loop points are infrequent)

