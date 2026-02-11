"""
Experimental user simulators for tau-voice full-duplex text streaming.

TextStreamingUserSimulator was the original text-based full-duplex user simulator,
superseded by VoiceStreamingUserSimulator for production voice evals.
Preserved here for experimental/research purposes.
"""

from experiments.tau_voice.users.text_streaming_user_simulator import (
    TextStreamingUserSimulator,
    UserTextChunkingMixin,
)

__all__ = [
    "TextStreamingUserSimulator",
    "UserTextChunkingMixin",
]
