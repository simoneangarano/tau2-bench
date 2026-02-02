# Copyright Sierra
"""Voice synthesis module."""

from tau2.data_model.voice_personas import ALL_PERSONA_NAMES

from .audio_effects import BackgroundNoiseGenerator
from .synthesize import synthesize_from_text

__all__ = [
    "synthesize_from_text",
    "ALL_PERSONA_NAMES",
    "BackgroundNoiseGenerator",
]
