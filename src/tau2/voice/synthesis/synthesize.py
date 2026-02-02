"""Core voice synthesis (TTS) functions."""

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from tau2.data_model.audio import AudioData
from tau2.data_model.voice import ElevenLabsTTSConfig, SynthesisConfig, SynthesisResult
from tau2.utils.retry import tts_retry
from tau2.voice.synthesis.audio_effects.noise_generator import (
    BackgroundNoiseGenerator,
    apply_background_noise,
    create_background_noise_generator,
)
from tau2.voice.utils.audio_io import save_wav_file
from tau2.voice.utils.elevenlabs_utils import tts_elevenlabs

load_dotenv()

ProviderConfig = ElevenLabsTTSConfig


def synthesize_from_text(
    text: str,
    config: Optional[SynthesisConfig] = None,
    output_path: Optional[Path] = None,
    background_noise_generator: Optional[BackgroundNoiseGenerator] = None,
) -> SynthesisResult:
    """Synthesize text to speech and optionally save to file."""
    try:
        if config is None:
            config = SynthesisConfig()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        audio_data = synthesize_voice(
            text=text,
            provider=config.provider,
            provider_config=config.provider_config,
        )

        # Create background noise generator if not provided
        if background_noise_generator is None:
            logger.warning("No background noise generator provided, creating a new one")
            background_noise_generator = create_background_noise_generator(
                config.source_effects_config,
                audio_data.format.sample_rate,
            )

        # Apply background noise
        audio_data = apply_background_noise(
            audio=audio_data,
            noise_generator=background_noise_generator,
        )

        if output_path:
            audio_data.audio_path = output_path
            save_wav_file(audio_data, str(output_path))

        return SynthesisResult(
            text_input=text,
            audio_data=audio_data,
            error=None,
        )

    except Exception as e:
        return SynthesisResult(error=f"Synthesis failed: {str(e)}")


@tts_retry
def synthesize_voice(
    text: str,
    provider: str,
    provider_config: ProviderConfig,
) -> AudioData:
    """Synthesize voice from text using the specified configuration."""
    if provider == "elevenlabs":
        audio_data = tts_elevenlabs(text=text, config=provider_config)
    else:
        raise ValueError(f"Unsupported synthesis provider: {provider}")

    if not audio_data.format.is_pcm16:
        raise ValueError(
            f"TTS must output PCM_S16LE, got {audio_data.format.encoding}. "
            "Configure the provider to use PCM output format."
        )

    return audio_data
