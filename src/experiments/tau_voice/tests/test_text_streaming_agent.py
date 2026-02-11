import pytest

from experiments.tau_voice.agents.llm_streaming_agent import TextStreamingLLMAgent
from tau2.data_model.message import AssistantMessage, UserMessage


@pytest.fixture
def streaming_agent(get_environment) -> TextStreamingLLMAgent:
    """Create a streaming agent with default (immediate response) behavior."""
    return TextStreamingLLMAgent(
        llm="gpt-4o-mini",
        tools=get_environment().get_tools(),
        domain_policy=get_environment().get_policy(),
        chunk_by="words",
        chunk_size=5,
    )


@pytest.fixture
def waiting_agent(get_environment) -> TextStreamingLLMAgent:
    """Create a streaming agent that waits for complete messages."""

    class WaitingAgent(TextStreamingLLMAgent):
        def _should_respond_to_chunk(self, incoming_chunk, state):
            # Only respond to final chunks
            return getattr(incoming_chunk, "is_final_chunk", True)

    return WaitingAgent(
        llm="gpt-4o-mini",
        tools=get_environment().get_tools(),
        domain_policy=get_environment().get_policy(),
        chunk_by="words",
        chunk_size=5,
    )


@pytest.fixture
def first_user_message():
    return UserMessage(content="Hello can you help me create a task?", role="user")


def test_streaming_agent_has_both_methods(streaming_agent: TextStreamingLLMAgent):
    """Test that streaming agent has get_next_chunk (full-duplex only)."""
    # Streaming agents are pure full-duplex, they don't have generate_next_message
    assert hasattr(streaming_agent, "get_next_chunk")
    assert callable(streaming_agent.get_next_chunk)


def test_streaming_agent_is_full_duplex_only(
    streaming_agent: TextStreamingLLMAgent, first_user_message: UserMessage
):
    """Test that streaming agent is full-duplex only (no generate_next_message)."""
    # Streaming agents are pure full-duplex, they don't have generate_next_message
    assert not hasattr(streaming_agent, "generate_next_message")
    # They use get_next_chunk for communication
    assert hasattr(streaming_agent, "get_next_chunk")


def test_streaming_agent_get_next_chunk_always_returns_message(
    streaming_agent: TextStreamingLLMAgent, first_user_message: UserMessage
):
    """Test that get_next_chunk always returns a message (never None)."""
    agent_state = streaming_agent.get_init_state()

    # Send a user message chunk
    first_user_message.contains_speech = True
    chunk, agent_state = streaming_agent.get_next_chunk(agent_state, first_user_message)

    # Should always get a chunk (never None)
    assert chunk is not None
    assert isinstance(chunk, AssistantMessage)
    assert hasattr(chunk, "contains_speech")
    assert chunk.contains_speech is not None


def test_streaming_agent_chunk_accumulation(streaming_agent: TextStreamingLLMAgent):
    """Test that chunks accumulate in state and contains_speech is properly tracked."""
    agent_state = streaming_agent.get_init_state()

    # Send chunk 0 (not final, contains speech)
    chunk_0 = UserMessage(
        role="user",
        content="Hello ",
        chunk_id=0,
        is_final_chunk=False,
        contains_speech=True,
    )
    response_0, agent_state = streaming_agent.get_next_chunk(agent_state, chunk_0)

    # Always returns a chunk (may have contains_speech=False if waiting)
    assert response_0 is not None
    assert isinstance(response_0, AssistantMessage)
    # Chunk should be in input_turn_taking_buffer
    assert len(agent_state.input_turn_taking_buffer) >= 1

    # Send chunk 1 (not final, contains speech)
    chunk_1 = UserMessage(
        role="user",
        content="world ",
        chunk_id=1,
        is_final_chunk=False,
        contains_speech=True,
    )
    response_1, agent_state = streaming_agent.get_next_chunk(agent_state, chunk_1)

    # Always returns a chunk
    assert response_1 is not None
    # Chunks should be accumulated
    assert len(agent_state.input_turn_taking_buffer) >= 2

    # Send chunk 2 (final, contains speech)
    _ = UserMessage(
        role="user",
        content="today!",
        chunk_id=2,
        is_final_chunk=True,
        contains_speech=True,
    )

    # Keep calling get_next_chunk until agent generates a response with speech
    max_attempts = 10
    for _ in range(max_attempts):
        # Send empty silence chunks to advance time
        silence_chunk = UserMessage(
            role="user", content=None, contains_speech=False, is_final_chunk=True
        )
        response, agent_state = streaming_agent.get_next_chunk(
            agent_state, silence_chunk
        )

        # Check if agent started talking (has pending output chunks or sent speech)
        if response.contains_speech or len(agent_state.output_streaming_queue) > 0:
            break


def test_streaming_agent_tool_calls_not_chunked(streaming_agent: TextStreamingLLMAgent):
    """Test that tool calls are sent as single chunks, not split."""
    agent_state = streaming_agent.get_init_state()

    # Create a message asking for a tool call
    user_msg = UserMessage(role="user", content="Create a task called 'Test Task'")

    # Get first chunk
    chunk, agent_state = streaming_agent.get_next_chunk(agent_state, user_msg)

    # If it's a tool call, it should be complete (not chunked)
    if chunk and chunk.tool_calls:
        # Tool call should be in a single chunk with is_final_chunk=True
        assert chunk.is_final_chunk is True or not hasattr(chunk, "is_final_chunk")
        # Should not have partial content
        assert chunk.tool_calls is not None
        assert len(chunk.tool_calls) > 0


def test_streaming_agent_state_independence(streaming_agent: TextStreamingLLMAgent):
    """Test that agent input_turn_taking_buffer are properly isolated per state."""
    # Create two independent states
    state1 = streaming_agent.get_init_state()
    state2 = streaming_agent.get_init_state()

    # Verify states start independent
    assert id(state1) != id(state2)
    assert id(state1.input_turn_taking_buffer) != id(state2.input_turn_taking_buffer)

    # Manually add different pending chunks to each state
    state1.input_turn_taking_buffer.append(
        UserMessage(role="user", content="Chunk for state 1")
    )
    state2.input_turn_taking_buffer.append(
        UserMessage(role="user", content="Chunk for state 2")
    )

    # Verify independence
    assert len(state1.input_turn_taking_buffer) == 1
    assert len(state2.input_turn_taking_buffer) == 1
    assert state1.input_turn_taking_buffer[0].content == "Chunk for state 1"
    assert state2.input_turn_taking_buffer[0].content == "Chunk for state 2"


def test_streaming_agent_chunk_metadata(waiting_agent: TextStreamingLLMAgent):
    """Test that chunks have proper metadata (chunk_id, is_final_chunk)."""
    agent_state = waiting_agent.get_init_state()

    # Trigger a response
    user_msg = UserMessage(
        role="user", content="Tell me about your services", is_final_chunk=True
    )

    chunks = []
    chunk, agent_state = waiting_agent.get_next_chunk(agent_state, user_msg)

    # Collect all chunks
    while chunk is not None:
        chunks.append(chunk)
        if getattr(chunk, "is_final_chunk", True):
            break
        chunk, agent_state = waiting_agent.get_next_chunk(agent_state, None)

    # Should have at least one chunk
    assert len(chunks) > 0

    # Last chunk should be marked as final
    if len(chunks) > 0:
        last_chunk = chunks[-1]
        assert getattr(last_chunk, "is_final_chunk", True) is True


def test_streaming_agent_cost_on_final_chunk_only(
    waiting_agent: TextStreamingLLMAgent,
):
    """Test that cost and usage are only on the final chunk."""
    agent_state = waiting_agent.get_init_state()

    user_msg = UserMessage(
        role="user", content="Tell me a long story", is_final_chunk=True
    )

    chunks = []
    chunk, agent_state = waiting_agent.get_next_chunk(agent_state, user_msg)

    while chunk is not None:
        chunks.append(chunk)
        if getattr(chunk, "is_final_chunk", True):
            break
        chunk, agent_state = waiting_agent.get_next_chunk(agent_state, None)

    # If we have multiple chunks, only the last should have cost
    if len(chunks) > 1:
        for i, chunk in enumerate(chunks[:-1]):
            # Non-final chunks should have zero or no cost
            assert chunk.cost == 0.0 or chunk.cost is None

        # Final chunk should have cost
        final_chunk = chunks[-1]
        assert final_chunk.cost is not None


def test_streaming_agent_empty_state_initialization(
    streaming_agent: TextStreamingLLMAgent,
):
    """Test that state initializes with empty input_turn_taking_buffer and output_streaming_queue."""
    state = streaming_agent.get_init_state()

    assert hasattr(state, "input_turn_taking_buffer")
    assert hasattr(state, "output_streaming_queue")
    assert state.input_turn_taking_buffer == []
    assert state.output_streaming_queue == []
    assert len(state.input_turn_taking_buffer) == 0
    assert len(state.output_streaming_queue) == 0


def test_streaming_agent_state_serialization(streaming_agent: TextStreamingLLMAgent):
    """Test that state with input_turn_taking_buffer and output_streaming_queue exists and can be used."""
    agent_state = streaming_agent.get_init_state()

    # Test that pending chunk fields exist
    assert hasattr(agent_state, "input_turn_taking_buffer")
    assert hasattr(agent_state, "output_streaming_queue")
    assert isinstance(agent_state.input_turn_taking_buffer, list)
    assert isinstance(agent_state.output_streaming_queue, list)

    # Test that model_dump includes pending chunk fields
    state_dict = agent_state.model_dump()
    assert "input_turn_taking_buffer" in state_dict
    assert "output_streaming_queue" in state_dict

    # Test that pending chunk fields can hold messages
    # (actual serialization tested elsewhere in message tests)


def test_streaming_agent_time_counters(streaming_agent: TextStreamingLLMAgent):
    """Test that time_since_last_talk and time_since_last_other_talk update correctly."""
    agent_state = streaming_agent.get_init_state()

    # Initial values should be 0
    assert agent_state.time_since_last_talk == 0
    assert agent_state.time_since_last_other_talk == 0

    # Send a speech chunk from user
    user_speech = UserMessage(role="user", content="Hello", contains_speech=True)
    chunk, agent_state = streaming_agent.get_next_chunk(agent_state, user_speech)

    # After receiving speech, time_since_last_other_talk should reset
    assert agent_state.time_since_last_other_talk == 0

    # Send silence chunks to increment counters
    silence = UserMessage(role="user", content=None, contains_speech=False)
    for i in range(3):
        chunk, agent_state = streaming_agent.get_next_chunk(agent_state, silence)
        # time_since_last_other_talk should increment with each silence
        assert agent_state.time_since_last_other_talk == i + 1
        print(
            f"After silence chunk {i}: time_since_last_other_talk={agent_state.time_since_last_other_talk}"
        )


def test_streaming_agent_contains_speech_on_all_chunks(
    streaming_agent: TextStreamingLLMAgent,
):
    """Test that all chunks returned have contains_speech explicitly set."""
    agent_state = streaming_agent.get_init_state()

    # Send various types of chunks
    test_chunks = [
        UserMessage(role="user", content="Hello", contains_speech=True),
        UserMessage(role="user", content=None, contains_speech=False),
        UserMessage(role="user", content="Test", contains_speech=True),
    ]

    for incoming in test_chunks:
        response, agent_state = streaming_agent.get_next_chunk(agent_state, incoming)

        # Every response should have contains_speech set
        assert response is not None
        assert hasattr(response, "contains_speech")
        assert response.contains_speech is not None
        assert isinstance(response.contains_speech, bool)
        print(f"Received {incoming.contains_speech} → Sent {response.contains_speech}")


def test_streaming_agent_speech_detection(streaming_agent: TextStreamingLLMAgent):
    """Test that speech_detection method works correctly."""
    # Test with speech chunk
    speech_chunk = UserMessage(role="user", content="Hello", contains_speech=True)
    assert streaming_agent.speech_detection(speech_chunk) is True

    # Test with silence chunk
    silence_chunk = UserMessage(role="user", content=None, contains_speech=False)
    assert streaming_agent.speech_detection(silence_chunk) is False

    # Test with non-UserMessage (should return False)
    agent_chunk = AssistantMessage(role="assistant", content="Hi", contains_speech=True)
    assert streaming_agent.speech_detection(agent_chunk) is False
