"""
Tests for text full-duplex streaming.

This test suite verifies that text streaming works correctly with:
- TextStreamingLLMAgent (receives text, sends text)
- TextStreamingUserSimulator (receives text, sends text)
- Proper state management
- contains_speech flag handling
"""

import pytest

from experiments.tau_voice.agents.llm_streaming_agent import TextStreamingLLMAgent
from experiments.tau_voice.users.text_streaming_user_simulator import (
    TextStreamingUserSimulator,
)
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.domains.mock.environment import get_environment, get_tasks
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator


@pytest.fixture
def text_orchestrator():
    """Create a text full-duplex orchestrator for testing."""
    # Get environment and tasks
    env = get_environment()
    tasks = get_tasks()
    task = tasks[0]

    # Create text streaming agent and user
    agent = TextStreamingLLMAgent(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        llm="gpt-4o-mini",
        chunk_by="words",
        chunk_size=5,
    )

    user = TextStreamingUserSimulator(
        tools=env.get_user_tools() if env.user_tools is not None else None,
        instructions=str(task.user_scenario),
        llm="gpt-4o-mini",
        chunk_by="words",
        chunk_size=5,
    )

    orchestrator = FullDuplexOrchestrator(
        domain="mock",
        agent=agent,
        user=user,
        environment=env,
        task=task,
        max_steps=20,
        max_errors=10,
        seed=42,
    )

    return orchestrator


def test_text_orchestrator_initialization(text_orchestrator):
    """Test that text orchestrator initializes correctly."""
    text_orchestrator.initialize()

    # Check that states are initialized
    assert text_orchestrator.agent_state is not None
    assert text_orchestrator.user_state is not None

    # Check that current chunks are initialized
    assert text_orchestrator.current_agent_chunk is not None
    assert text_orchestrator.current_user_chunk is not None

    # Check initial agent message
    assert text_orchestrator.current_agent_chunk.role == "assistant"
    assert text_orchestrator.current_agent_chunk.content is not None
    assert text_orchestrator.current_agent_chunk.contains_speech is not None
    assert text_orchestrator.current_agent_chunk.is_audio is False

    # Check initial user chunk (should be empty/silence)
    assert text_orchestrator.current_user_chunk.role == "user"
    assert text_orchestrator.current_user_chunk.contains_speech is False
    assert text_orchestrator.current_user_chunk.is_audio is False

    # Check pending chunks - initialization may add initial empty chunks to pending_input
    # This is expected behavior as initialization runs the first exchange
    assert len(text_orchestrator.agent_state.output_streaming_queue) == 0
    assert len(text_orchestrator.user_state.output_streaming_queue) == 0


def test_text_first_step_user_receives_greeting(text_orchestrator):
    """Test that user receives agent's greeting on first step."""
    text_orchestrator.initialize()

    # Get initial agent message
    initial_agent_message = text_orchestrator.current_agent_chunk
    assert initial_agent_message.contains_speech is True  # Agent said "Hi!"

    # Take first step
    text_orchestrator.step()

    # User should have received the agent's greeting
    user_state = text_orchestrator.user_state
    assert len(user_state.input_turn_taking_buffer) >= 1

    # Check that the agent's greeting is in user's pending input
    speech_chunks = [
        chunk
        for chunk in user_state.input_turn_taking_buffer
        if chunk.contains_speech and chunk.content
    ]
    assert len(speech_chunks) >= 1
    assert speech_chunks[0].role == "assistant"
    assert speech_chunks[0].contains_speech is True
    assert speech_chunks[0].content == initial_agent_message.content


def test_text_contains_speech_flag_consistency(text_orchestrator):
    """Test that contains_speech flag is consistently set across all chunks."""
    text_orchestrator.initialize()

    # Run several steps
    for _ in range(5):
        if text_orchestrator.done:
            break
        text_orchestrator.step()

        # Check agent chunks
        agent_chunk = text_orchestrator.current_agent_chunk
        assert hasattr(agent_chunk, "contains_speech")
        assert agent_chunk.contains_speech is not None
        assert isinstance(agent_chunk.contains_speech, bool)
        assert agent_chunk.is_audio is False  # Text chunks should not be audio

        # Check user chunks
        user_chunk = text_orchestrator.current_user_chunk
        assert hasattr(user_chunk, "contains_speech")
        assert user_chunk.contains_speech is not None
        assert isinstance(user_chunk.contains_speech, bool)
        assert user_chunk.is_audio is False  # Text chunks should not be audio

        # Check all pending chunks in agent state
        for chunk in text_orchestrator.agent_state.input_turn_taking_buffer:
            assert hasattr(chunk, "contains_speech")
            assert chunk.contains_speech is not None
            assert isinstance(chunk, (UserMessage, AssistantMessage))

        # Check all pending chunks in user state
        for chunk in text_orchestrator.user_state.input_turn_taking_buffer:
            assert hasattr(chunk, "contains_speech")
            assert chunk.contains_speech is not None
            assert isinstance(chunk, (UserMessage, AssistantMessage))


def test_text_speech_detection_logic(text_orchestrator):
    """Test that speech detection works correctly for text chunks."""
    text_orchestrator.initialize()

    initial_agent_chunk = text_orchestrator.current_agent_chunk
    assert initial_agent_chunk.contains_speech is True

    # User's speech detection should return True for agent's speech
    user_detects_speech = text_orchestrator.user.speech_detection(initial_agent_chunk)
    assert user_detects_speech is True

    # Take a step
    text_orchestrator.step()

    # Check that empty chunks have contains_speech=False
    if text_orchestrator.current_user_chunk.contains_speech is False:
        # Agent's speech detection should return False for user's silence
        agent_detects_speech = text_orchestrator.agent.speech_detection(
            text_orchestrator.current_user_chunk
        )
        assert agent_detects_speech is False


def test_text_ongoing_speech_duration(text_orchestrator):
    """Test that ongoing speech duration is calculated correctly."""
    text_orchestrator.initialize()

    # Run a few steps
    for step_num in range(3):
        if text_orchestrator.done:
            break

        text_orchestrator.step()

        # Check user state
        user_state = text_orchestrator.user_state
        total_speech = user_state.input_total_speech_duration()
        ongoing_speech = user_state.input_ongoing_speech_duration()

        print(
            f"\nStep {step_num}: User - Total speech: {total_speech}, Ongoing: {ongoing_speech}"
        )
        print(f"  Pending input chunks: {len(user_state.input_turn_taking_buffer)}")
        for i, chunk in enumerate(user_state.input_turn_taking_buffer):
            content_preview = (
                chunk.content[:30] + "..."
                if chunk.content and len(chunk.content) > 30
                else chunk.content
            )
            print(
                f"    Chunk {i}: role={chunk.role}, contains_speech={chunk.contains_speech}, content='{content_preview}'"
            )

        # Ongoing speech should never exceed total speech
        assert ongoing_speech <= total_speech

        # Check agent state
        agent_state = text_orchestrator.agent_state
        total_speech = agent_state.input_total_speech_duration()
        ongoing_speech = agent_state.input_ongoing_speech_duration()

        print(
            f"Step {step_num}: Agent - Total speech: {total_speech}, Ongoing: {ongoing_speech}"
        )
        print(f"  Pending input chunks: {len(agent_state.input_turn_taking_buffer)}")
        for i, chunk in enumerate(agent_state.input_turn_taking_buffer):
            content_preview = (
                chunk.content[:30] + "..."
                if chunk.content and len(chunk.content) > 30
                else chunk.content
            )
            print(
                f"    Chunk {i}: role={chunk.role}, contains_speech={chunk.contains_speech}, content='{content_preview}'"
            )

        # Ongoing speech should never exceed total speech
        assert ongoing_speech <= total_speech


def test_text_turn_taking_responds_to_speech(text_orchestrator):
    """Test that participants respond when they detect speech after silence."""
    text_orchestrator.initialize()

    # Track whether we see a response generated
    response_generated = False
    max_steps_to_check = 10

    for step_num in range(max_steps_to_check):
        if text_orchestrator.done:
            break

        prev_user_pending_output = len(
            text_orchestrator.user_state.output_streaming_queue
        )
        prev_agent_pending_output = len(
            text_orchestrator.agent_state.output_streaming_queue
        )

        text_orchestrator.step()

        # Check if user or agent generated new output chunks
        if (
            len(text_orchestrator.user_state.output_streaming_queue)
            > prev_user_pending_output
        ):
            response_generated = True
            print(f"User generated response at step {step_num}")
            # Print the user's response
            if text_orchestrator.user_state.output_streaming_queue:
                chunk = text_orchestrator.user_state.output_streaming_queue[0]
                print(f"  User response: {chunk.content}")
            break

        if (
            len(text_orchestrator.agent_state.output_streaming_queue)
            > prev_agent_pending_output
        ):
            response_generated = True
            print(f"Agent generated response at step {step_num}")
            # Print the agent's response
            if text_orchestrator.agent_state.output_streaming_queue:
                chunk = text_orchestrator.agent_state.output_streaming_queue[0]
                print(f"  Agent response: {chunk.content}")
            break

    # We should see at least one response generated within reasonable steps
    assert response_generated, (
        f"No response generated within {max_steps_to_check} steps"
    )


def test_text_chunks_always_have_contains_speech(text_orchestrator):
    """Test that ALL chunks always have contains_speech set (never None)."""
    text_orchestrator.initialize()

    for step_num in range(5):
        if text_orchestrator.done:
            break

        text_orchestrator.step()

        # Check message chunks
        for i, chunk in enumerate(text_orchestrator.get_messages()):
            assert hasattr(chunk, "contains_speech"), (
                f"Chunk {i} in messages missing contains_speech"
            )
            assert chunk.contains_speech is not None, (
                f"Chunk {i} in messages has contains_speech=None"
            )


def test_text_empty_chunks_have_correct_flag(text_orchestrator):
    """Test that empty/silence chunks have contains_speech=False."""
    text_orchestrator.initialize()

    for step_num in range(3):
        if text_orchestrator.done:
            break

        text_orchestrator.step()

        # Check current chunks
        agent_chunk = text_orchestrator.current_agent_chunk
        user_chunk = text_orchestrator.current_user_chunk

        # If chunk has no content (or content is None/empty), contains_speech should be False
        if agent_chunk.content is None or (
            isinstance(agent_chunk.content, str) and not agent_chunk.content.strip()
        ):
            assert agent_chunk.contains_speech is False, (
                f"Empty agent chunk should have contains_speech=False"
            )

        if user_chunk.content is None or (
            isinstance(user_chunk.content, str) and not user_chunk.content.strip()
        ):
            assert user_chunk.contains_speech is False, (
                f"Empty user chunk should have contains_speech=False"
            )


def test_text_chunking_creates_multiple_chunks(text_orchestrator):
    """Test that text messages are properly chunked into multiple pieces."""
    text_orchestrator.initialize()

    # Run until we get a response
    max_steps = 10
    for _ in range(max_steps):
        if text_orchestrator.done:
            break

        prev_user_chunks = len(text_orchestrator.user_state.output_streaming_queue)
        text_orchestrator.step()

        # Check if user generated a response with multiple chunks
        if len(text_orchestrator.user_state.output_streaming_queue) > prev_user_chunks:
            # Check that chunks have proper metadata
            for i, chunk in enumerate(
                text_orchestrator.user_state.output_streaming_queue
            ):
                assert chunk.chunk_id is not None
                assert isinstance(chunk.is_final_chunk, bool)
                assert (
                    chunk.contains_speech is True
                )  # All text content chunks should have speech
                print(
                    f"User chunk {i}: id={chunk.chunk_id}, final={chunk.is_final_chunk}, content='{chunk.content}'"
                )

            # If there are multiple chunks, check that only the last is marked final
            if len(text_orchestrator.user_state.output_streaming_queue) > 1:
                for i, chunk in enumerate(
                    text_orchestrator.user_state.output_streaming_queue[:-1]
                ):
                    assert chunk.is_final_chunk is False, (
                        f"Non-final chunk {i} should have is_final_chunk=False"
                    )

                last_chunk = text_orchestrator.user_state.output_streaming_queue[-1]
                assert last_chunk.is_final_chunk is True, (
                    "Last chunk should have is_final_chunk=True"
                )

            break


def test_text_conversation_progresses(text_orchestrator):
    """Test that the text conversation progresses through multiple turns."""
    text_orchestrator.initialize()

    # Track message exchanges
    user_messages = []
    agent_messages = []

    max_steps = 20
    for step_num in range(max_steps):
        if text_orchestrator.done:
            break

        text_orchestrator.step()

        # Collect messages
        for msg in text_orchestrator.get_messages():
            if (
                isinstance(msg, UserMessage)
                and msg.content
                and msg.content not in [m.content for m in user_messages]
            ):
                user_messages.append(msg)
                print(f"Step {step_num}: User said: {msg.content[:50]}...")
            elif (
                isinstance(msg, AssistantMessage)
                and msg.content
                and msg.content not in [m.content for m in agent_messages]
            ):
                agent_messages.append(msg)
                print(f"Step {step_num}: Agent said: {msg.content[:50]}...")

    # We should have at least one exchange
    assert len(user_messages) > 0, "User should have sent at least one message"
    assert len(agent_messages) > 0, "Agent should have sent at least one message"

    print(
        f"\nTotal conversation: {len(user_messages)} user messages, {len(agent_messages)} agent messages"
    )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
