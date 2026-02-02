# Copyright Sierra
"""Audio effects data models (config and result classes)."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from tau2.voice_config import (
    BURST_NOISE_EVENTS_PER_MINUTE,
    BURST_SNR_RANGE_DB,
    ENABLE_BACKGROUND_NOISE,
    ENABLE_BURST_NOISE,
    ENABLE_DYNAMIC_MUFFLING,
    ENABLE_FRAME_DROPS,
    ENABLE_OUT_OF_TURN_SPEECH,
    ENABLE_VOCAL_TICS_DURING_SPEECH,
    FRAME_DROP_BURST_DURATION_MS,
    FRAME_DROP_COUNT,
    FRAME_DROP_DURATION_MS,
    FRAME_DROP_RATE,
    MIN_WORDS_FOR_VOCAL_TICS,
    MUFFLE_CUTOFF_FREQ,
    MUFFLE_PROBABILITY,
    MUFFLE_SEGMENT_COUNT,
    MUFFLE_SEGMENT_DURATION_MS,
    MUFFLE_TRANSITION_MS,
    NOISE_SNR_DB,
    NOISE_SNR_DRIFT_DB,
    NOISE_VARIATION_SPEED,
    NON_DIRECTED_PHRASES,
    OUT_OF_TURN_SPEECH_EVENTS_PER_MINUTE,
    VOCAL_TICS,
)

UserSpeechInsertType = Literal["vocal_tic", "non_directed_phrase"]


class UserSpeechInsert(BaseModel):
    """A user speech insert (vocal tic or non-directed phrase)."""

    text: str = Field(description="Text to synthesize")
    type: UserSpeechInsertType = Field(description="Type of speech insert")

    @property
    def is_muffled(self) -> bool:
        return self.type == "non_directed_phrase"


class ChannelEffectsConfig(BaseModel):
    """Channel/transmission effects (frame drops via Gilbert-Elliott model)."""

    enable_frame_drops: bool = Field(
        default=ENABLE_FRAME_DROPS,
        description="Enable frame drop simulation",
    )
    frame_drop_rate: float = Field(
        default=FRAME_DROP_RATE,
        ge=0.0,
        lt=0.2,
        description="Target average loss rate (0.0 to 0.2). E.g., 0.02 for 2% loss.",
    )
    frame_drop_burst_duration_ms: float = Field(
        default=FRAME_DROP_BURST_DURATION_MS,
        gt=0.0,
        description="Average burst duration in ms. Longer = more consecutive drops.",
    )
    frame_drop_count: int = Field(
        default=FRAME_DROP_COUNT,
        ge=1,
        description="Number of drops per trigger (batch mode only)",
    )
    frame_drop_duration_ms: int = Field(
        default=FRAME_DROP_DURATION_MS,
        ge=1,
        description="Duration of each individual frame drop in ms",
    )


class ChannelEffectsResult(BaseModel):
    """Result of transmission/network channel effects."""

    frame_drops_enabled: bool = Field(default=False)
    frame_drop_ms: int = Field(default=0)


class SourceEffectsConfig(BaseModel):
    """Source/acoustic environment effects (background noise, burst noise)."""

    # Background noise SNR settings
    enable_background_noise: bool = Field(default=ENABLE_BACKGROUND_NOISE)
    noise_snr_db: float = Field(default=NOISE_SNR_DB)
    noise_snr_drift_db: float = Field(default=NOISE_SNR_DRIFT_DB)
    noise_variation_speed: float = Field(default=NOISE_VARIATION_SPEED)

    # Burst noise SNR settings
    enable_burst_noise: bool = Field(default=ENABLE_BURST_NOISE)
    burst_noise_events_per_minute: float = Field(
        default=BURST_NOISE_EVENTS_PER_MINUTE, ge=0.0
    )
    burst_snr_range_db: tuple[float, float] = Field(default=BURST_SNR_RANGE_DB)


class SourceEffectsResult(BaseModel):
    """Result of acoustic environment/source effects."""

    burst_noise_file: Optional[str] = Field(default=None)
    speech_insert: Optional[UserSpeechInsert] = Field(default=None)


class SpeechEffectsConfig(BaseModel):
    """Speech effects applied to the speaker's voice."""

    enable_dynamic_muffling: bool = Field(default=ENABLE_DYNAMIC_MUFFLING)
    muffle_probability: float = Field(default=MUFFLE_PROBABILITY, ge=0.0, le=1.0)
    muffle_segment_count: int = Field(default=MUFFLE_SEGMENT_COUNT, ge=1)
    muffle_segment_duration_ms: int = Field(default=MUFFLE_SEGMENT_DURATION_MS, ge=1)
    muffle_cutoff_freq: float = Field(default=MUFFLE_CUTOFF_FREQ, ge=100.0)
    muffle_transition_ms: int = Field(default=MUFFLE_TRANSITION_MS, ge=0)

    enable_vocal_tics: bool = Field(default=ENABLE_VOCAL_TICS_DURING_SPEECH)
    vocal_tics: list[UserSpeechInsert] = Field(
        default_factory=lambda: [
            UserSpeechInsert(text=t, type="vocal_tic") for t in VOCAL_TICS
        ]
    )
    min_words_for_vocal_tics: int = Field(default=MIN_WORDS_FOR_VOCAL_TICS, ge=1)

    enable_non_directed_phrases: bool = Field(default=ENABLE_OUT_OF_TURN_SPEECH)
    non_directed_phrases: list[UserSpeechInsert] = Field(
        default_factory=lambda: [
            UserSpeechInsert(text=t, type="non_directed_phrase")
            for t in NON_DIRECTED_PHRASES
        ]
    )

    speech_insert_events_per_minute: float = Field(
        default=OUT_OF_TURN_SPEECH_EVENTS_PER_MINUTE, ge=0.0
    )

    def get_out_of_turn_speech_inserts(self) -> list[UserSpeechInsert]:
        """Get combined list of out-of-turn speech inserts."""
        items: list[UserSpeechInsert] = []
        if self.enable_vocal_tics:
            items.extend(self.vocal_tics)
        if self.enable_non_directed_phrases:
            items.extend(self.non_directed_phrases)
        return items


class SpeechEffectsResult(BaseModel):
    """Result of speech effects applied to the speaker's voice."""

    dynamic_muffling_enabled: bool = Field(default=False)
    speech_insert: Optional[UserSpeechInsert] = Field(default=None)
