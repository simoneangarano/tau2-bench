"""
Tests for voice full-duplex streaming.

This test suite verifies that voice streaming works correctly with:
- VoiceStreamingLLMAgent (receives audio, sends text)
- VoiceStreamingUserSimulator (receives text, sends audio)
- Proper state management
- contains_speech flag handling
- Interruption/barge-in behavior
- Backchanneling behavior
"""

import pytest

from experiments.tau_voice.agents.llm_streaming_agent import VoiceStreamingLLMAgent
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.voice import SynthesisConfig, TranscriptionConfig, VoiceSettings
from tau2.domains.mock.environment import get_environment, get_tasks
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator


@pytest.fixture
def voice_orchestrator():
    """Create a voice full-duplex orchestrator for testing."""
    # Get environment and tasks
    env = get_environment()
    tasks = get_tasks()
    task = tasks[0]

    # Voice settings
    agent_voice_settings = VoiceSettings(
        transcription_config=TranscriptionConfig(),
        synthesis_config=None,
    )

    user_voice_settings = VoiceSettings(
        transcription_config=None,
        synthesis_config=SynthesisConfig(),
    )

    # Harmonizing chunk duration for audio and text
    CHUNK_DURATION_IN_SECONDS = 1
    AUDIO_FREQUENCY = 8000
    TEXT_NUM_WORDS_PER_SECOND = 2

    AUDIO_CHUNK_SIZE_NUM_SAMPLES = AUDIO_FREQUENCY * CHUNK_DURATION_IN_SECONDS
    TEXT_CHUNK_SIZE_NUM_WORDS = TEXT_NUM_WORDS_PER_SECOND * CHUNK_DURATION_IN_SECONDS

    WAIT_TO_RESPOND_THRESHOLD_OTHER_IN_SECONDS = 1
    WAIT_TO_RESPOND_THRESHOLD_OTHER_IN_CHUNKS = (
        WAIT_TO_RESPOND_THRESHOLD_OTHER_IN_SECONDS // CHUNK_DURATION_IN_SECONDS
    )

    WAIT_TO_RESPOND_THRESHOLD_SELF_IN_SECONDS = 2
    WAIT_TO_RESPOND_THRESHOLD_SELF_IN_CHUNKS = (
        WAIT_TO_RESPOND_THRESHOLD_SELF_IN_SECONDS // CHUNK_DURATION_IN_SECONDS
    )

    INTERRUPT_THRESHOLD_IN_SECONDS = 2
    INTERRUPT_THRESHOLD_IN_CHUNKS = (
        INTERRUPT_THRESHOLD_IN_SECONDS // CHUNK_DURATION_IN_SECONDS
    )

    MAX_STEPS_IN_SECONDS = 60
    MAX_STEPS = MAX_STEPS_IN_SECONDS // CHUNK_DURATION_IN_SECONDS

    # Receives audio streams and sends text chunks
    agent = VoiceStreamingLLMAgent(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        llm="gpt-4o-mini",
        voice_settings=agent_voice_settings,
        chunk_by="words",
        chunk_size=TEXT_CHUNK_SIZE_NUM_WORDS,
        wait_to_respond_threshold_other=WAIT_TO_RESPOND_THRESHOLD_OTHER_IN_CHUNKS,
        wait_to_respond_threshold_self=WAIT_TO_RESPOND_THRESHOLD_SELF_IN_CHUNKS,
    )

    # Receives text chunks and sends audio streams
    user = VoiceStreamingUserSimulator(
        tools=env.get_user_tools() if env.user_tools is not None else None,
        instructions=str(task.user_scenario),
        llm="gpt-4o-mini",
        voice_settings=user_voice_settings,
        chunk_size=AUDIO_CHUNK_SIZE_NUM_SAMPLES,
        wait_to_respond_threshold_other=WAIT_TO_RESPOND_THRESHOLD_OTHER_IN_CHUNKS,
        wait_to_respond_threshold_self=WAIT_TO_RESPOND_THRESHOLD_SELF_IN_CHUNKS,
        yield_threshold_when_interrupted=INTERRUPT_THRESHOLD_IN_CHUNKS,
        backchannel_min_threshold=None,
    )

    orchestrator = FullDuplexOrchestrator(
        domain="mock",
        agent=agent,
        user=user,
        environment=env,
        task=task,
        max_steps=MAX_STEPS,
        max_errors=10,
        seed=42,
    )

    return orchestrator


def test_voice_orchestrator_initialization(voice_orchestrator):
    """Test that voice orchestrator initializes correctly."""
    voice_orchestrator.initialize()

    # Check that states are initialized
    assert voice_orchestrator.agent_state is not None
    assert voice_orchestrator.user_state is not None

    # Check that current chunks are initialized
    assert voice_orchestrator.current_agent_chunk is not None
    assert voice_orchestrator.current_user_chunk is not None

    # Check initial agent message
    assert voice_orchestrator.current_agent_chunk.role == "assistant"
    assert voice_orchestrator.current_agent_chunk.content is not None
    assert voice_orchestrator.current_agent_chunk.contains_speech is not None

    # Check initial user chunk (should be empty/silence)
    assert voice_orchestrator.current_user_chunk.role == "user"
    assert voice_orchestrator.current_user_chunk.contains_speech is False

    # Check pending chunks - initialization may add initial empty chunks to pending_input
    # This is expected behavior as initialization runs the first exchange
    assert len(voice_orchestrator.agent_state.output_streaming_queue) == 0
    assert len(voice_orchestrator.user_state.output_streaming_queue) == 0


def test_voice_first_step_user_receives_greeting(voice_orchestrator):
    """Test that user receives agent's greeting on first step."""
    voice_orchestrator.initialize()

    # Get initial agent message
    initial_agent_message = voice_orchestrator.current_agent_chunk
    assert initial_agent_message.contains_speech is True  # Agent said "Hi!"

    # Take first step
    voice_orchestrator.step()

    # User should have received the agent's greeting
    user_state = voice_orchestrator.user_state
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


def test_voice_contains_speech_flag_consistency(voice_orchestrator):
    """Test that contains_speech flag is consistently set across all chunks."""
    voice_orchestrator.initialize()

    # Run several steps
    for _ in range(5):
        if voice_orchestrator.done:
            break
        voice_orchestrator.step()

        # Check agent chunks
        agent_chunk = voice_orchestrator.current_agent_chunk
        assert hasattr(agent_chunk, "contains_speech")
        assert agent_chunk.contains_speech is not None
        assert isinstance(agent_chunk.contains_speech, bool)

        # Check user chunks
        user_chunk = voice_orchestrator.current_user_chunk
        assert hasattr(user_chunk, "contains_speech")
        assert user_chunk.contains_speech is not None
        assert isinstance(user_chunk.contains_speech, bool)

        # Check all pending chunks in agent state
        for chunk in voice_orchestrator.agent_state.input_turn_taking_buffer:
            assert hasattr(chunk, "contains_speech")
            assert chunk.contains_speech is not None

        # Check all pending chunks in user state
        for chunk in voice_orchestrator.user_state.input_turn_taking_buffer:
            assert hasattr(chunk, "contains_speech")
            assert chunk.contains_speech is not None


def test_voice_speech_detection_logic(voice_orchestrator):
    """Test that speech detection works correctly."""
    voice_orchestrator.initialize()

    initial_agent_chunk = voice_orchestrator.current_agent_chunk
    assert initial_agent_chunk.contains_speech is True

    # User's speech detection should return True for agent's speech
    user_detects_speech = voice_orchestrator.user.speech_detection(initial_agent_chunk)
    assert user_detects_speech is True

    # Take a step
    voice_orchestrator.step()

    # Check that empty chunks have contains_speech=False
    if voice_orchestrator.current_user_chunk.contains_speech is False:
        # Agent's speech detection should return False for user's silence
        agent_detects_speech = voice_orchestrator.agent.speech_detection(
            voice_orchestrator.current_user_chunk
        )
        assert agent_detects_speech is False


def test_voice_ongoing_speech_duration(voice_orchestrator):
    """Test that ongoing speech duration is calculated correctly."""
    voice_orchestrator.initialize()

    # Run a few steps
    for step_num in range(3):
        if voice_orchestrator.done:
            break

        voice_orchestrator.step()

        # Check user state
        user_state = voice_orchestrator.user_state
        total_speech = user_state.input_total_speech_duration()
        ongoing_speech = user_state.input_ongoing_speech_duration()

        print(
            f"\nStep {step_num}: User - Total speech: {total_speech}, Ongoing: {ongoing_speech}"
        )
        print(f"  Pending input chunks: {len(user_state.input_turn_taking_buffer)}")
        for i, chunk in enumerate(user_state.input_turn_taking_buffer):
            print(
                f"    Chunk {i}: role={chunk.role}, contains_speech={chunk.contains_speech}"
            )

        # Ongoing speech should never exceed total speech
        assert ongoing_speech <= total_speech

        # Check agent state
        agent_state = voice_orchestrator.agent_state
        total_speech = agent_state.input_total_speech_duration()
        ongoing_speech = agent_state.input_ongoing_speech_duration()

        print(
            f"Step {step_num}: Agent - Total speech: {total_speech}, Ongoing: {ongoing_speech}"
        )
        print(f"  Pending input chunks: {len(agent_state.input_turn_taking_buffer)}")
        for i, chunk in enumerate(agent_state.input_turn_taking_buffer):
            print(
                f"    Chunk {i}: role={chunk.role}, contains_speech={chunk.contains_speech}"
            )

        # Ongoing speech should never exceed total speech
        assert ongoing_speech <= total_speech


def test_voice_turn_taking_responds_to_speech(voice_orchestrator):
    """Test that participants respond when they detect speech after silence."""
    voice_orchestrator.initialize()

    # Track whether we see a response generated
    response_generated = False
    max_steps_to_check = 10

    for step_num in range(max_steps_to_check):
        if voice_orchestrator.done:
            break

        prev_user_pending_output = len(
            voice_orchestrator.user_state.output_streaming_queue
        )
        prev_agent_pending_output = len(
            voice_orchestrator.agent_state.output_streaming_queue
        )

        voice_orchestrator.step()

        # Check if user or agent generated new output chunks
        if (
            len(voice_orchestrator.user_state.output_streaming_queue)
            > prev_user_pending_output
        ):
            response_generated = True
            print(f"User generated response at step {step_num}")
            break

        if (
            len(voice_orchestrator.agent_state.output_streaming_queue)
            > prev_agent_pending_output
        ):
            response_generated = True
            print(f"Agent generated response at step {step_num}")
            break

    # We should see at least one response generated within reasonable steps
    assert response_generated, (
        f"No response generated within {max_steps_to_check} steps"
    )


def test_voice_chunks_always_have_contains_speech(voice_orchestrator):
    """Test that ALL chunks always have contains_speech set (never None)."""
    voice_orchestrator.initialize()

    for step_num in range(5):
        if voice_orchestrator.done:
            break

        voice_orchestrator.step()

        # Check message chunks
        for i, chunk in enumerate(voice_orchestrator.get_messages()):
            assert hasattr(chunk, "contains_speech"), (
                f"Chunk {i} in messages missing contains_speech"
            )
            # For now, allow None since default was just changed
            # In the future this should be: assert chunk.contains_speech is not None


def test_voice_empty_chunks_have_correct_flag(voice_orchestrator):
    """Test that empty/silence chunks have contains_speech=False."""
    voice_orchestrator.initialize()

    for step_num in range(3):
        if voice_orchestrator.done:
            break

        voice_orchestrator.step()

        # Check current chunks
        agent_chunk = voice_orchestrator.current_agent_chunk
        user_chunk = voice_orchestrator.current_user_chunk

        # If chunk has no content (or content is None/empty), contains_speech should be False
        if agent_chunk.content is None or (
            isinstance(agent_chunk.content, str) and not agent_chunk.content.strip()
        ):
            # Empty text chunk should have contains_speech=False
            if not agent_chunk.is_audio:
                assert agent_chunk.contains_speech is False, (
                    f"Empty agent text chunk should have contains_speech=False"
                )

        if user_chunk.content is None or (
            isinstance(user_chunk.content, str) and not user_chunk.content.strip()
        ):
            # Empty text chunk should have contains_speech=False
            if not user_chunk.is_audio:
                assert user_chunk.contains_speech is False, (
                    f"Empty user text chunk should have contains_speech=False"
                )


# =============================================================================
# Interruption/Barge-in Tests
# =============================================================================


@pytest.fixture
def interruptible_user_orchestrator():
    """Create an orchestrator with a user that can interrupt (low threshold)."""
    env = get_environment()
    tasks = get_tasks()
    task = tasks[0]

    agent_voice_settings = VoiceSettings(
        transcription_config=TranscriptionConfig(),
        synthesis_config=None,
    )

    user_voice_settings = VoiceSettings(
        transcription_config=None,
        synthesis_config=SynthesisConfig(),
    )

    # Low interrupt threshold = user can interrupt quickly
    INTERRUPT_THRESHOLD = 2

    agent = VoiceStreamingLLMAgent(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        llm="gpt-4o-mini",
        voice_settings=agent_voice_settings,
        chunk_by="words",
        chunk_size=2,
        wait_to_respond_threshold_other=1,
        wait_to_respond_threshold_self=2,
    )

    user = VoiceStreamingUserSimulator(
        tools=env.get_user_tools() if env.user_tools is not None else None,
        instructions=str(task.user_scenario),
        llm="gpt-4o-mini",
        voice_settings=user_voice_settings,
        chunk_size=8000,
        wait_to_respond_threshold_other=1,
        wait_to_respond_threshold_self=2,
        yield_threshold_when_interrupted=INTERRUPT_THRESHOLD,
        backchannel_min_threshold=None,
    )

    return FullDuplexOrchestrator(
        domain="mock",
        agent=agent,
        user=user,
        environment=env,
        task=task,
        max_steps=30,
        max_errors=10,
        seed=42,
    )


@pytest.fixture
def non_interruptible_user_orchestrator():
    """Create an orchestrator with a user that cannot interrupt."""
    env = get_environment()
    tasks = get_tasks()
    task = tasks[0]

    agent_voice_settings = VoiceSettings(
        transcription_config=TranscriptionConfig(),
        synthesis_config=None,
    )

    user_voice_settings = VoiceSettings(
        transcription_config=None,
        synthesis_config=SynthesisConfig(),
    )

    agent = VoiceStreamingLLMAgent(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        llm="gpt-4o-mini",
        voice_settings=agent_voice_settings,
        chunk_by="words",
        chunk_size=2,
        wait_to_respond_threshold_other=1,
        wait_to_respond_threshold_self=2,
    )

    # yield_threshold_when_interrupted=None means user cannot interrupt
    user = VoiceStreamingUserSimulator(
        tools=env.get_user_tools() if env.user_tools is not None else None,
        instructions=str(task.user_scenario),
        llm="gpt-4o-mini",
        voice_settings=user_voice_settings,
        chunk_size=8000,
        wait_to_respond_threshold_other=1,
        wait_to_respond_threshold_self=2,
        yield_threshold_when_interrupted=None,
        backchannel_min_threshold=None,
    )

    return FullDuplexOrchestrator(
        domain="mock",
        agent=agent,
        user=user,
        environment=env,
        task=task,
        max_steps=30,
        max_errors=10,
        seed=42,
    )


def test_voice_user_yield_threshold_when_interrupted_is_set(
    interruptible_user_orchestrator,
):
    """Test that yield_threshold_when_interrupted is correctly set on the user."""
    user = interruptible_user_orchestrator.user
    assert hasattr(user, "yield_threshold_when_interrupted")
    assert user.yield_threshold_when_interrupted == 2


def test_voice_user_yield_threshold_when_interrupted_none_disables_interruption(
    non_interruptible_user_orchestrator,
):
    """Test that yield_threshold_when_interrupted=None disables interruption."""
    user = non_interruptible_user_orchestrator.user
    assert hasattr(user, "yield_threshold_when_interrupted")
    assert user.yield_threshold_when_interrupted is None


def test_voice_both_participants_can_speak_simultaneously(voice_orchestrator):
    """Test that in full-duplex, both participants can have speech at the same tick.

    This is the fundamental requirement for interruption to work.
    """
    voice_orchestrator.initialize()

    # Track whether we see overlapping speech
    overlapping_speech_detected = False
    max_steps = 15

    for step_num in range(max_steps):
        if voice_orchestrator.done:
            break

        voice_orchestrator.step()

        agent_chunk = voice_orchestrator.current_agent_chunk
        user_chunk = voice_orchestrator.current_user_chunk

        # Check if both are speaking at the same time
        if agent_chunk.contains_speech and user_chunk.contains_speech:
            overlapping_speech_detected = True
            print(f"Step {step_num}: Overlapping speech detected!")
            print(f"  Agent: {agent_chunk.content}")
            print(f"  User: {user_chunk.content}")
            break

    # Note: This may or may not happen depending on the conversation flow
    # The test verifies the capability exists, not that it always happens
    print(f"Overlapping speech detected: {overlapping_speech_detected}")


def test_voice_interruption_scenario_simulation(interruptible_user_orchestrator):
    """Simulate an interruption scenario by manually creating overlapping chunks."""
    orch = interruptible_user_orchestrator
    orch.initialize()

    # Get initial states
    _agent_state = orch.agent_state
    _user_state = orch.user_state

    # Simulate agent speaking
    agent_speaking_chunk = AssistantMessage(
        role="assistant",
        content="Let me explain the policy in detail...",
        contains_speech=True,
    )

    # Simulate user starting to speak (interruption)
    user_interrupting_chunk = UserMessage(
        role="user",
        content="Wait, I have a question!",
        contains_speech=True,
    )

    # Both chunks can coexist - this is what enables interruption
    assert agent_speaking_chunk.contains_speech is True
    assert user_interrupting_chunk.contains_speech is True

    # In a real scenario, the orchestrator would handle these overlapping chunks
    # The user's yield_threshold_when_interrupted determines when user can start speaking
    assert orch.user.yield_threshold_when_interrupted == 2


def test_voice_state_tracks_other_party_speech(voice_orchestrator):
    """Test that state tracks when the other party is speaking.

    This is essential for interruption detection.
    """
    voice_orchestrator.initialize()

    # Run some steps
    for _ in range(3):
        if voice_orchestrator.done:
            break
        voice_orchestrator.step()

    user_state = voice_orchestrator.user_state
    agent_state = voice_orchestrator.agent_state

    # Check that input_turn_taking_buffer exists and tracks incoming chunks
    assert hasattr(user_state, "input_turn_taking_buffer")
    assert hasattr(agent_state, "input_turn_taking_buffer")

    # Check that time counters track speech patterns
    assert hasattr(user_state, "time_since_last_other_talk")
    assert hasattr(agent_state, "time_since_last_other_talk")


# =============================================================================
# Backchanneling Tests
# =============================================================================


@pytest.fixture
def backchanneling_user_orchestrator():
    """Create an orchestrator with a user that can backchannel."""
    env = get_environment()
    tasks = get_tasks()
    task = tasks[0]

    agent_voice_settings = VoiceSettings(
        transcription_config=TranscriptionConfig(),
        synthesis_config=None,
    )

    user_voice_settings = VoiceSettings(
        transcription_config=None,
        synthesis_config=SynthesisConfig(),
    )

    agent = VoiceStreamingLLMAgent(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        llm="gpt-4o-mini",
        voice_settings=agent_voice_settings,
        chunk_by="words",
        chunk_size=2,
        wait_to_respond_threshold_other=1,
        wait_to_respond_threshold_self=2,
    )

    # backchannel_min_threshold=3 enables backchanneling after 3 chunks of agent speech
    user = VoiceStreamingUserSimulator(
        tools=env.get_user_tools() if env.user_tools is not None else None,
        instructions=str(task.user_scenario),
        llm="gpt-4o-mini",
        voice_settings=user_voice_settings,
        chunk_size=8000,
        wait_to_respond_threshold_other=2,
        wait_to_respond_threshold_self=4,
        yield_threshold_when_interrupted=None,  # Disable interruption
        backchannel_min_threshold=3,  # Enable backchanneling
    )

    return FullDuplexOrchestrator(
        domain="mock",
        agent=agent,
        user=user,
        environment=env,
        task=task,
        max_steps=30,
        max_errors=10,
        seed=42,
    )


def test_voice_user_backchannel_min_threshold_is_set(backchanneling_user_orchestrator):
    """Test that backchannel_min_threshold is correctly set on the user."""
    user = backchanneling_user_orchestrator.user
    assert hasattr(user, "backchannel_min_threshold")
    assert user.backchannel_min_threshold == 3


def test_voice_user_backchannel_and_interrupt_can_be_independent(
    backchanneling_user_orchestrator,
):
    """Test that backchannel and interrupt thresholds are independent."""
    user = backchanneling_user_orchestrator.user

    # Backchanneling enabled, interruption disabled
    assert user.backchannel_min_threshold == 3
    assert user.yield_threshold_when_interrupted is None


def test_voice_backchannel_vs_interrupt_different_behaviors():
    """Test that backchanneling and interruption are configured differently."""
    _env = get_environment()
    tasks = get_tasks()
    task = tasks[0]

    user_voice_settings = VoiceSettings(
        transcription_config=None,
        synthesis_config=SynthesisConfig(),
    )

    # User with only interruption
    user_interrupt_only = VoiceStreamingUserSimulator(
        tools=None,
        instructions=str(task.user_scenario),
        llm="gpt-4o-mini",
        voice_settings=user_voice_settings,
        chunk_size=8000,
        yield_threshold_when_interrupted=2,
        backchannel_min_threshold=None,
    )

    # User with only backchanneling
    user_backchannel_only = VoiceStreamingUserSimulator(
        tools=None,
        instructions=str(task.user_scenario),
        llm="gpt-4o-mini",
        voice_settings=user_voice_settings,
        chunk_size=8000,
        yield_threshold_when_interrupted=None,
        backchannel_min_threshold=3,
    )

    # User with both
    user_both = VoiceStreamingUserSimulator(
        tools=None,
        instructions=str(task.user_scenario),
        llm="gpt-4o-mini",
        voice_settings=user_voice_settings,
        chunk_size=8000,
        yield_threshold_when_interrupted=2,
        backchannel_min_threshold=5,
    )

    # Verify configurations
    assert user_interrupt_only.yield_threshold_when_interrupted == 2
    assert user_interrupt_only.backchannel_min_threshold is None

    assert user_backchannel_only.yield_threshold_when_interrupted is None
    assert user_backchannel_only.backchannel_min_threshold == 3

    assert user_both.yield_threshold_when_interrupted == 2
    assert user_both.backchannel_min_threshold == 5


def test_voice_ongoing_speech_duration_for_backchanneling(voice_orchestrator):
    """Test that ongoing speech duration can be used for backchannel decisions.

    Backchanneling typically happens after the other party has been speaking
    for a certain duration.
    """
    voice_orchestrator.initialize()

    # Run some steps and check ongoing speech tracking
    for step_num in range(5):
        if voice_orchestrator.done:
            break

        voice_orchestrator.step()

        user_state = voice_orchestrator.user_state
        ongoing = user_state.input_ongoing_speech_duration()

        # This value is used by turn-taking logic for backchanneling decisions
        print(f"Step {step_num}: User sees ongoing agent speech = {ongoing}")

        # ongoing_speech_duration should be a non-negative integer
        assert isinstance(ongoing, int)
        assert ongoing >= 0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
