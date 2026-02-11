"""
Unit tests for VoiceStreamingLLMAgent.

This test suite verifies that the voice streaming agent works correctly with:
- Audio chunk handling (receives audio, sends text)
- State management with VoiceState
- Turn-taking logic
- Speech detection from audio
- Time counters
"""

import base64

import pytest

from experiments.tau_voice.agents.llm_streaming_agent import VoiceStreamingLLMAgent
from tau2.data_model.audio import AudioEncoding, AudioFormat
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.voice import SynthesisConfig, TranscriptionConfig, VoiceSettings


@pytest.fixture
def voice_settings():
    """Create voice settings for testing (transcription enabled, synthesis disabled)."""
    return VoiceSettings(
        transcription_config=TranscriptionConfig(),
        synthesis_config=None,
    )


@pytest.fixture
def streaming_agent(get_environment, voice_settings) -> VoiceStreamingLLMAgent:
    """Create a voice streaming agent with default behavior."""
    return VoiceStreamingLLMAgent(
        llm="gpt-4o-mini",
        tools=get_environment().get_tools(),
        domain_policy=get_environment().get_policy(),
        voice_settings=voice_settings,
        chunk_by="words",
        chunk_size=5,
        wait_to_respond_threshold_other=2,
        wait_to_respond_threshold_self=4,
    )


@pytest.fixture
def audio_format():
    """Create a standard audio format for testing."""
    return AudioFormat(
        encoding=AudioEncoding.PCM_S16LE,
        sample_rate=16000,
    )


@pytest.fixture
def audio_user_message(audio_format):
    """Create a sample audio user message."""
    # Create fake audio data: 100 samples at 16-bit (2 bytes per sample)
    audio_bytes = b"\x00\x01" * 100
    return UserMessage(
        role="user",
        content="Hello, I need help",
        is_audio=True,
        audio_content=base64.b64encode(audio_bytes).decode("utf-8"),
        audio_format=audio_format,
        contains_speech=True,
    )


@pytest.fixture
def text_user_message():
    """Create a simple text user message (for agents that receive text)."""
    return UserMessage(
        role="user",
        content="Hello, I need help with my reservation",
        contains_speech=True,
    )


# --- Basic Interface Tests ---


def test_voice_streaming_agent_has_get_next_chunk(
    streaming_agent: VoiceStreamingLLMAgent,
):
    """Test that voice streaming agent has get_next_chunk method."""
    assert hasattr(streaming_agent, "get_next_chunk")
    assert callable(streaming_agent.get_next_chunk)


def test_voice_streaming_agent_is_full_duplex_only(
    streaming_agent: VoiceStreamingLLMAgent,
):
    """Test that voice streaming agent is full-duplex only (no generate_next_message)."""
    assert not hasattr(streaming_agent, "generate_next_message")
    assert hasattr(streaming_agent, "get_next_chunk")


def test_voice_streaming_agent_has_voice_methods(
    streaming_agent: VoiceStreamingLLMAgent,
):
    """Test that voice streaming agent has voice-specific methods."""
    # Should have voice methods from VoiceMixin
    assert hasattr(streaming_agent, "transcribe_voice")
    assert hasattr(streaming_agent, "speech_detection")
    assert hasattr(streaming_agent, "voice_settings")


# --- State Initialization Tests ---


def test_voice_streaming_agent_state_initialization(
    streaming_agent: VoiceStreamingLLMAgent,
):
    """Test that state initializes correctly with streaming and voice fields."""
    state = streaming_agent.get_init_state()

    # Streaming state fields
    assert hasattr(state, "input_turn_taking_buffer")
    assert hasattr(state, "output_streaming_queue")
    assert state.input_turn_taking_buffer == []
    assert state.output_streaming_queue == []

    # Time counters
    assert hasattr(state, "time_since_last_talk")
    assert hasattr(state, "time_since_last_other_talk")
    assert state.time_since_last_talk == 0
    assert state.time_since_last_other_talk == 0


def test_voice_streaming_agent_state_independence(
    streaming_agent: VoiceStreamingLLMAgent,
):
    """Test that states are properly isolated."""
    state1 = streaming_agent.get_init_state()
    state2 = streaming_agent.get_init_state()

    # Verify states are independent
    assert id(state1) != id(state2)
    assert id(state1.input_turn_taking_buffer) != id(state2.input_turn_taking_buffer)

    # Modify one state
    state1.input_turn_taking_buffer.append(
        UserMessage(role="user", content="State 1 chunk")
    )

    # Verify other state is unaffected
    assert len(state1.input_turn_taking_buffer) == 1
    assert len(state2.input_turn_taking_buffer) == 0


def test_voice_streaming_agent_state_serialization(
    streaming_agent: VoiceStreamingLLMAgent,
):
    """Test that state can be serialized."""
    state = streaming_agent.get_init_state()

    # Test that model_dump works and includes streaming fields
    state_dict = state.model_dump()
    assert "input_turn_taking_buffer" in state_dict
    assert "output_streaming_queue" in state_dict
    assert "time_since_last_talk" in state_dict
    assert "time_since_last_other_talk" in state_dict


# --- get_next_chunk Behavior Tests ---


def test_voice_streaming_agent_get_next_chunk_returns_message(
    streaming_agent: VoiceStreamingLLMAgent, audio_user_message: UserMessage
):
    """Test that get_next_chunk always returns a message (never None)."""
    state = streaming_agent.get_init_state()

    chunk, state = streaming_agent.get_next_chunk(state, audio_user_message)

    assert chunk is not None
    assert isinstance(chunk, AssistantMessage)
    assert hasattr(chunk, "contains_speech")
    assert chunk.contains_speech is not None


def test_voice_streaming_agent_chunk_accumulation(
    streaming_agent: VoiceStreamingLLMAgent, audio_format
):
    """Test that incoming chunks accumulate in state."""
    state = streaming_agent.get_init_state()

    # Send first audio chunk
    audio_bytes = b"\x00\x01" * 50
    chunk_0 = UserMessage(
        role="user",
        content="Hello ",
        is_audio=True,
        audio_content=base64.b64encode(audio_bytes).decode("utf-8"),
        audio_format=audio_format,
        chunk_id=0,
        is_final_chunk=False,
        contains_speech=True,
    )
    response_0, state = streaming_agent.get_next_chunk(state, chunk_0)

    assert response_0 is not None
    assert len(state.input_turn_taking_buffer) >= 1

    # Send second audio chunk
    chunk_1 = UserMessage(
        role="user",
        content="world",
        is_audio=True,
        audio_content=base64.b64encode(audio_bytes).decode("utf-8"),
        audio_format=audio_format,
        chunk_id=1,
        is_final_chunk=False,
        contains_speech=True,
    )
    response_1, state = streaming_agent.get_next_chunk(state, chunk_1)

    assert response_1 is not None
    assert len(state.input_turn_taking_buffer) >= 2


def test_voice_streaming_agent_contains_speech_on_all_responses(
    streaming_agent: VoiceStreamingLLMAgent,
):
    """Test that all response chunks have contains_speech set."""
    state = streaming_agent.get_init_state()

    # Test with various input types
    test_inputs = [
        UserMessage(role="user", content="Hello", contains_speech=True),
        UserMessage(role="user", content=None, contains_speech=False),
        UserMessage(role="user", content="Help me", contains_speech=True),
    ]

    for incoming in test_inputs:
        response, state = streaming_agent.get_next_chunk(state, incoming)

        assert response is not None
        assert hasattr(response, "contains_speech")
        assert response.contains_speech is not None
        assert isinstance(response.contains_speech, bool)


# --- Time Counter Tests ---


def test_voice_streaming_agent_time_counters(
    streaming_agent: VoiceStreamingLLMAgent,
):
    """Test that time counters update correctly."""
    state = streaming_agent.get_init_state()

    # Initial values
    assert state.time_since_last_talk == 0
    assert state.time_since_last_other_talk == 0

    # Send speech chunk
    speech = UserMessage(role="user", content="Hello", contains_speech=True)
    _, state = streaming_agent.get_next_chunk(state, speech)

    # After receiving speech, time_since_last_other_talk should reset
    assert state.time_since_last_other_talk == 0

    # Send silence chunks
    silence = UserMessage(role="user", content=None, contains_speech=False)
    for i in range(3):
        _, state = streaming_agent.get_next_chunk(state, silence)
        assert state.time_since_last_other_talk == i + 1


# --- Speech Detection Tests ---


def test_voice_streaming_agent_speech_detection_audio(
    streaming_agent: VoiceStreamingLLMAgent, audio_format
):
    """Test speech detection with audio chunks."""
    # Audio chunk with speech
    audio_bytes = b"\x00\x01" * 50
    speech_chunk = UserMessage(
        role="user",
        content="Hello",
        is_audio=True,
        audio_content=base64.b64encode(audio_bytes).decode("utf-8"),
        audio_format=audio_format,
        contains_speech=True,
    )
    assert streaming_agent.speech_detection(speech_chunk) is True

    # Audio chunk without speech (silence)
    silence_chunk = UserMessage(
        role="user",
        content=None,
        is_audio=True,
        audio_content=base64.b64encode(audio_bytes).decode("utf-8"),
        audio_format=audio_format,
        contains_speech=False,
    )
    assert streaming_agent.speech_detection(silence_chunk) is False


def test_voice_streaming_agent_speech_detection_text(
    streaming_agent: VoiceStreamingLLMAgent,
):
    """Test speech detection with text chunks (fallback)."""
    # Text chunk with speech
    speech_chunk = UserMessage(role="user", content="Hello", contains_speech=True)
    assert streaming_agent.speech_detection(speech_chunk) is True

    # Text chunk without speech
    silence_chunk = UserMessage(role="user", content=None, contains_speech=False)
    assert streaming_agent.speech_detection(silence_chunk) is False


def test_voice_streaming_agent_speech_detection_wrong_role(
    streaming_agent: VoiceStreamingLLMAgent,
):
    """Test that speech detection returns False for wrong role (assistant)."""
    # Agent should only detect speech from user messages
    agent_chunk = AssistantMessage(
        role="assistant", content="Hi there", contains_speech=True
    )
    assert streaming_agent.speech_detection(agent_chunk) is False


# --- Tool Call Tests ---


def test_voice_streaming_agent_tool_calls_not_chunked(
    streaming_agent: VoiceStreamingLLMAgent,
):
    """Test that tool calls are sent as single chunks, not split."""
    state = streaming_agent.get_init_state()

    # Send a message that might trigger a tool call
    user_msg = UserMessage(
        role="user", content="Create a task called 'Test Task'", contains_speech=True
    )
    chunk, state = streaming_agent.get_next_chunk(state, user_msg)

    # If it's a tool call, it should be complete
    if chunk and chunk.tool_calls:
        assert chunk.is_final_chunk is True or not hasattr(chunk, "is_final_chunk")
        assert chunk.tool_calls is not None
        assert len(chunk.tool_calls) > 0


# --- Voice Settings Tests ---


def test_voice_streaming_agent_requires_transcription(get_environment):
    """Test that voice streaming agent requires transcription to be enabled."""
    # Settings with transcription disabled should fail
    invalid_settings = VoiceSettings(
        transcription_config=None,
        synthesis_config=SynthesisConfig(),
    )

    with pytest.raises(ValueError, match="transcription"):
        VoiceStreamingLLMAgent(
            llm="gpt-4o-mini",
            tools=get_environment().get_tools(),
            domain_policy=get_environment().get_policy(),
            voice_settings=invalid_settings,
            chunk_by="words",
            chunk_size=5,
        )


def test_voice_streaming_agent_thresholds(get_environment, voice_settings):
    """Test that wait thresholds are properly set."""
    agent = VoiceStreamingLLMAgent(
        llm="gpt-4o-mini",
        tools=get_environment().get_tools(),
        domain_policy=get_environment().get_policy(),
        voice_settings=voice_settings,
        chunk_by="words",
        chunk_size=5,
        wait_to_respond_threshold_other=3,
        wait_to_respond_threshold_self=5,
    )

    assert agent.wait_to_respond_threshold_other == 3
    assert agent.wait_to_respond_threshold_self == 5


# --- Chunking Configuration Tests ---


def test_voice_streaming_agent_chunking_config(
    streaming_agent: VoiceStreamingLLMAgent,
):
    """Test that chunking configuration is properly set."""
    assert hasattr(streaming_agent, "chunk_by")
    assert hasattr(streaming_agent, "chunk_size")
    assert streaming_agent.chunk_by in ["chars", "words", "sentences"]
    assert streaming_agent.chunk_size > 0


def test_voice_streaming_agent_outputs_text_chunks(
    streaming_agent: VoiceStreamingLLMAgent,
):
    """Test that voice streaming agent outputs text chunks (not audio).

    VoiceStreamingLLMAgent receives audio but outputs text (to be synthesized externally).
    """
    state = streaming_agent.get_init_state()

    user_msg = UserMessage(role="user", content="Hello", contains_speech=True)
    chunk, state = streaming_agent.get_next_chunk(state, user_msg)

    # Output should be text, not audio
    assert chunk is not None
    assert chunk.is_audio is False or chunk.is_audio is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
