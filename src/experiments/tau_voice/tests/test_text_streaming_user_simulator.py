import pytest

from experiments.tau_voice.users.text_streaming_user_simulator import (
    TextStreamingUserSimulator,
)
from tau2.data_model.message import AssistantMessage, UserMessage


@pytest.fixture
def user_instructions() -> str:
    return (
        "You are Mia Li. You want to fly from New York to Seattle on May 20 (one way)."
    )


@pytest.fixture
def streaming_user(user_instructions: str) -> TextStreamingUserSimulator:
    """Create a streaming user with default (immediate response) behavior."""
    return TextStreamingUserSimulator(
        llm="gpt-4o-mini",
        instructions=user_instructions,
        chunk_by="words",
        chunk_size=5,
    )


@pytest.fixture
def waiting_user(user_instructions: str) -> TextStreamingUserSimulator:
    """Create a streaming user that waits for complete messages."""

    class WaitingUser(TextStreamingUserSimulator):
        def _should_respond_to_chunk(self, incoming_chunk, state):
            # Only respond to final chunks
            return getattr(incoming_chunk, "is_final_chunk", True)

    return WaitingUser(
        llm="gpt-4o-mini",
        instructions=user_instructions,
        chunk_by="words",
        chunk_size=5,
    )


@pytest.fixture
def first_agent_message() -> AssistantMessage:
    return AssistantMessage(
        content="Hello, how can I help you today?", role="assistant"
    )


def test_streaming_user_has_both_methods(streaming_user: TextStreamingUserSimulator):
    """Test that streaming user has get_next_chunk (full-duplex only)."""
    # Streaming users are pure full-duplex, they don't have generate_next_message
    assert not hasattr(streaming_user, "generate_next_message")
    assert hasattr(streaming_user, "get_next_chunk")
    assert callable(streaming_user.get_next_chunk)


def test_streaming_user_generate_next_message(
    streaming_user: TextStreamingUserSimulator, first_agent_message: AssistantMessage
):
    """Test that streaming user is full-duplex only (no generate_next_message)."""
    # Streaming users are pure full-duplex, they don't have generate_next_message
    assert not hasattr(streaming_user, "generate_next_message")
    # They use get_next_chunk for communication
    assert hasattr(streaming_user, "get_next_chunk")


def test_streaming_user_get_next_chunk_always_returns_message(
    streaming_user: TextStreamingUserSimulator, first_agent_message: AssistantMessage
):
    """Test that get_next_chunk always returns a message (never None)."""
    user_state = streaming_user.get_init_state()

    # Send an agent message chunk
    first_agent_message.contains_speech = True
    chunk, user_state = streaming_user.get_next_chunk(user_state, first_agent_message)

    # Should always get a chunk (never None)
    assert chunk is not None
    assert isinstance(chunk, UserMessage)
    assert hasattr(chunk, "contains_speech")
    assert chunk.contains_speech is not None


def test_streaming_user_chunk_accumulation(streaming_user: TextStreamingUserSimulator):
    """Test that chunks accumulate in state and contains_speech is properly tracked."""
    user_state = streaming_user.get_init_state()

    # Send chunk 0 (not final, contains speech)
    chunk_0 = AssistantMessage(
        role="assistant",
        content="Hello ",
        chunk_id=0,
        is_final_chunk=False,
        contains_speech=True,
    )
    response_0, user_state = streaming_user.get_next_chunk(user_state, chunk_0)

    # Always returns a chunk (may have contains_speech=False if waiting)
    assert response_0 is not None
    assert isinstance(response_0, UserMessage)
    # Chunk should be in input_turn_taking_buffer
    assert len(user_state.input_turn_taking_buffer) >= 1

    # Send chunk 1 (not final, contains speech)
    chunk_1 = AssistantMessage(
        role="assistant",
        content="there! ",
        chunk_id=1,
        is_final_chunk=False,
        contains_speech=True,
    )
    response_1, user_state = streaming_user.get_next_chunk(user_state, chunk_1)

    # Always returns a chunk
    assert response_1 is not None
    # Chunks should be accumulated
    assert len(user_state.input_turn_taking_buffer) >= 2

    # Verify all chunks have contains_speech set
    for chunk in user_state.input_turn_taking_buffer:
        assert hasattr(chunk, "contains_speech")
        assert chunk.contains_speech is not None


def test_streaming_user_state_independence(streaming_user: TextStreamingUserSimulator):
    """Test that user input_turn_taking_buffer are properly isolated per state."""
    # Create two independent states
    state1 = streaming_user.get_init_state()
    state2 = streaming_user.get_init_state()

    # Verify states start independent
    assert id(state1) != id(state2)
    assert id(state1.input_turn_taking_buffer) != id(state2.input_turn_taking_buffer)

    # Manually add different pending chunks to each state
    state1.input_turn_taking_buffer.append(
        AssistantMessage(role="assistant", content="Chunk for state 1")
    )
    state2.input_turn_taking_buffer.append(
        AssistantMessage(role="assistant", content="Chunk for state 2")
    )

    # Verify independence
    assert len(state1.input_turn_taking_buffer) == 1
    assert len(state2.input_turn_taking_buffer) == 1
    assert state1.input_turn_taking_buffer[0].content == "Chunk for state 1"
    assert state2.input_turn_taking_buffer[0].content == "Chunk for state 2"


def test_streaming_user_chunk_metadata(waiting_user: TextStreamingUserSimulator):
    """Test that chunks have proper metadata (chunk_id, is_final_chunk)."""
    user_state = waiting_user.get_init_state()

    # Trigger a response
    agent_msg = AssistantMessage(
        role="assistant", content="Tell me about your trip", is_final_chunk=True
    )

    chunks = []
    chunk, user_state = waiting_user.get_next_chunk(user_state, agent_msg)

    # Collect all chunks
    while chunk is not None:
        chunks.append(chunk)
        if getattr(chunk, "is_final_chunk", True):
            break
        chunk, user_state = waiting_user.get_next_chunk(user_state, None)

    # Should have at least one chunk
    assert len(chunks) > 0

    # Last chunk should be marked as final
    if len(chunks) > 0:
        last_chunk = chunks[-1]
        assert getattr(last_chunk, "is_final_chunk", True) is True


def test_streaming_user_empty_state_initialization(
    streaming_user: TextStreamingUserSimulator,
):
    """Test that state initializes with empty input_turn_taking_buffer and output_streaming_queue."""
    state = streaming_user.get_init_state()

    assert hasattr(state, "input_turn_taking_buffer")
    assert hasattr(state, "output_streaming_queue")
    assert state.input_turn_taking_buffer == []
    assert state.output_streaming_queue == []
    assert len(state.input_turn_taking_buffer) == 0
    assert len(state.output_streaming_queue) == 0


def test_streaming_user_state_serialization(streaming_user: TextStreamingUserSimulator):
    """Test that state with input_turn_taking_buffer and output_streaming_queue exists and can be used."""
    user_state = streaming_user.get_init_state()

    # Test that pending chunk fields exist
    assert hasattr(user_state, "input_turn_taking_buffer")
    assert hasattr(user_state, "output_streaming_queue")
    assert isinstance(user_state.input_turn_taking_buffer, list)
    assert isinstance(user_state.output_streaming_queue, list)

    # Test that model_dump includes pending chunk fields
    state_dict = user_state.model_dump()
    assert "input_turn_taking_buffer" in state_dict
    assert "output_streaming_queue" in state_dict

    # Test that pending chunk fields can hold messages
    # (actual serialization tested elsewhere in message tests)


def test_streaming_user_tool_calls_not_chunked(
    streaming_user: TextStreamingUserSimulator,
):
    """Test that if user makes tool calls, they're sent as single chunks."""
    user_state = streaming_user.get_init_state()

    # Create a message that might trigger a tool call
    agent_msg = AssistantMessage(
        role="assistant", content="What would you like me to do?"
    )

    # Get first chunk
    chunk, user_state = streaming_user.get_next_chunk(user_state, agent_msg)

    # If it's a tool call, it should be complete (not chunked)
    if chunk and chunk.tool_calls:
        # Tool call should be in a single chunk
        assert chunk.is_final_chunk is True or not hasattr(chunk, "is_final_chunk")
        assert chunk.tool_calls is not None


def test_streaming_user_role_flipping(streaming_user: TextStreamingUserSimulator):
    """Test that user state properly flips roles for LLM interaction."""
    user_state = streaming_user.get_init_state()

    agent_msg = AssistantMessage(
        role="assistant", content="Hello!", is_final_chunk=True, contains_speech=True
    )

    # Generate a response
    _, user_state = streaming_user.get_next_chunk(user_state, agent_msg)

    # Test flip_roles method
    flipped = user_state.flip_roles()
    # Should flip roles properly (implementation specific)
    assert flipped is not None


def test_streaming_user_time_counters(streaming_user: TextStreamingUserSimulator):
    """Test that time_since_last_talk and time_since_last_other_talk update correctly."""
    user_state = streaming_user.get_init_state()

    # Initial values should be 0
    assert user_state.time_since_last_talk == 0
    assert user_state.time_since_last_other_talk == 0

    # Send a speech chunk from agent
    agent_speech = AssistantMessage(
        role="assistant", content="Hello", contains_speech=True
    )
    chunk, user_state = streaming_user.get_next_chunk(user_state, agent_speech)

    # After receiving speech, time_since_last_other_talk should reset
    assert user_state.time_since_last_other_talk == 0

    # Send silence chunks to increment counters
    silence = AssistantMessage(role="assistant", content=None, contains_speech=False)
    for i in range(3):
        chunk, user_state = streaming_user.get_next_chunk(user_state, silence)
        # time_since_last_other_talk should increment with each silence
        assert user_state.time_since_last_other_talk == i + 1
        print(
            f"After silence chunk {i}: time_since_last_other_talk={user_state.time_since_last_other_talk}"
        )


def test_streaming_user_contains_speech_on_all_chunks(
    streaming_user: TextStreamingUserSimulator,
):
    """Test that all chunks returned have contains_speech explicitly set."""
    user_state = streaming_user.get_init_state()

    # Send various types of chunks
    test_chunks = [
        AssistantMessage(role="assistant", content="Hello", contains_speech=True),
        AssistantMessage(role="assistant", content=None, contains_speech=False),
        AssistantMessage(role="assistant", content="Test", contains_speech=True),
    ]

    for incoming in test_chunks:
        response, user_state = streaming_user.get_next_chunk(user_state, incoming)

        # Every response should have contains_speech set
        assert response is not None
        assert hasattr(response, "contains_speech")
        assert response.contains_speech is not None
        assert isinstance(response.contains_speech, bool)
        print(f"Received {incoming.contains_speech} → Sent {response.contains_speech}")


def test_streaming_user_speech_detection(streaming_user: TextStreamingUserSimulator):
    """Test that speech_detection method works correctly."""
    # Test with speech chunk
    speech_chunk = AssistantMessage(
        role="assistant", content="Hello", contains_speech=True
    )
    assert streaming_user.speech_detection(speech_chunk) is True

    # Test with silence chunk
    silence_chunk = AssistantMessage(
        role="assistant", content=None, contains_speech=False
    )
    assert streaming_user.speech_detection(silence_chunk) is False

    # Test with non-AssistantMessage (should return False)
    user_chunk = UserMessage(role="user", content="Hi", contains_speech=True)
    assert streaming_user.speech_detection(user_chunk) is False
