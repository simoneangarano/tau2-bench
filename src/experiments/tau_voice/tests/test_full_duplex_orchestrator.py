from typing import Callable

import pytest

from experiments.tau_voice.agents.llm_streaming_agent import TextStreamingLLMAgent
from experiments.tau_voice.users.text_streaming_user_simulator import (
    TextStreamingUserSimulator,
)
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.orchestrator.modes import CommunicationMode


@pytest.fixture
def streaming_user() -> TextStreamingUserSimulator:
    return TextStreamingUserSimulator(
        instructions="You are a user simulator.",
        llm="gpt-3.5-turbo",
        llm_args={"temperature": 0.0},
        chunk_by="words",
        chunk_size=5,
    )


@pytest.fixture
def streaming_agent(
    get_environment: Callable[[], Environment],
) -> TextStreamingLLMAgent:
    environment = get_environment()
    return TextStreamingLLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm="gpt-3.5-turbo",
        llm_args={"temperature": 0.0},
        chunk_by="words",
        chunk_size=5,
    )


def test_full_duplex_orchestrator_validation(
    domain_name: str,
    streaming_user: TextStreamingUserSimulator,
    streaming_agent: TextStreamingLLMAgent,
    get_environment: Callable[[], Environment],
    base_task: Task,
):
    """Test that FullDuplexOrchestrator requires streaming-capable agent and user."""
    # Should succeed with streaming agent and user
    orchestrator = FullDuplexOrchestrator(
        domain=domain_name,
        user=streaming_user,
        agent=streaming_agent,
        environment=get_environment(),
        task=base_task,
    )
    assert orchestrator.mode == CommunicationMode.FULL_DUPLEX


def test_full_duplex_orchestrator_initialization(
    domain_name: str,
    streaming_user: TextStreamingUserSimulator,
    streaming_agent: TextStreamingLLMAgent,
    get_environment: Callable[[], Environment],
    base_task: Task,
):
    """Test FullDuplexOrchestrator initialization."""
    orchestrator = FullDuplexOrchestrator(
        domain=domain_name,
        user=streaming_user,
        agent=streaming_agent,
        environment=get_environment(),
        task=base_task,
    )
    orchestrator.initialize()

    # Check initialization
    assert not orchestrator.done
    assert orchestrator.termination_reason is None
    # In FULL_DUPLEX mode, ticks should be tracked
    assert len(orchestrator.get_trajectory()) >= 0


def test_full_duplex_orchestrator_step(
    domain_name: str,
    streaming_user: TextStreamingUserSimulator,
    streaming_agent: TextStreamingLLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that step() function works correctly in FullDuplexOrchestrator.

    In full-duplex mode, both participants always return a chunk (even if empty).
    An empty chunk has contains_speech=False and may have content=None.
    """
    orchestrator = FullDuplexOrchestrator(
        domain=domain_name,
        environment=get_environment(),
        user=streaming_user,
        agent=streaming_agent,
        task=base_task,
    )
    orchestrator.initialize()

    # Check initial state
    assert orchestrator.step_count == 0
    assert not orchestrator.done
    # Agent starts first with a greeting
    assert orchestrator.current_agent_chunk is not None
    assert orchestrator.current_agent_chunk.contains_speech is True
    # User returns an empty chunk (waiting) - not None, but contains_speech=False
    assert orchestrator.current_user_chunk is not None
    assert orchestrator.current_user_chunk.contains_speech is False
    initial_ticks_len = len(orchestrator.get_trajectory())

    # Execute one step
    orchestrator.step()

    # Check that step incremented
    assert orchestrator.step_count > 0

    # Check that ticks are added
    assert len(orchestrator.get_trajectory()) > initial_ticks_len

    # Check that current chunks are being tracked
    # Both participants always return chunks in full-duplex mode
    assert orchestrator.current_agent_chunk is not None
    assert orchestrator.current_user_chunk is not None

    # States should be updated
    assert orchestrator.agent_state is not None
    assert orchestrator.user_state is not None

    # Agent should have spoken (greeting in messages)
    assert len(orchestrator.agent_state.messages) > 0


def test_full_duplex_orchestrator_multiple_steps(
    domain_name: str,
    streaming_user: TextStreamingUserSimulator,
    streaming_agent: TextStreamingLLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test multiple step() calls in FullDuplexOrchestrator.

    Note: The user simulator may "wait" for several ticks before responding,
    depending on turn-taking thresholds. We verify the orchestrator mechanics
    work correctly, not that the user has spoken.
    """
    orchestrator = FullDuplexOrchestrator(
        domain=domain_name,
        environment=get_environment(),
        user=streaming_user,
        agent=streaming_agent,
        task=base_task,
    )
    orchestrator.initialize()

    initial_ticks_len = len(orchestrator.get_trajectory())

    # Execute multiple steps
    for i in range(3):
        if not orchestrator.done:
            orchestrator.step()

    # Should have progressed
    assert orchestrator.step_count >= 1
    # Ticks should have grown (ticks are added each step)
    assert len(orchestrator.get_trajectory()) > initial_ticks_len
    # Agent should have activity (at least the greeting)
    assert len(orchestrator.agent_state.messages) > 0
    # User state should exist and have ticks recorded
    assert orchestrator.user_state is not None
    assert len(orchestrator.user_state.ticks) > 0


def test_full_duplex_orchestrator_run(
    domain_name: str,
    streaming_user: TextStreamingUserSimulator,
    streaming_agent: TextStreamingLLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test running FullDuplexOrchestrator to completion.

    Verifies that the orchestrator can run a full simulation and produce
    a valid SimulationRun with messages.
    """
    orchestrator = FullDuplexOrchestrator(
        domain=domain_name,
        environment=get_environment(),
        user=streaming_user,
        agent=streaming_agent,
        task=base_task,
        max_steps=10,
    )

    simulation_run = orchestrator.run()

    # Should complete successfully
    assert simulation_run is not None
    assert orchestrator.done
    # Should have messages
    assert len(simulation_run.messages) > 0


def test_full_duplex_state_preservation(
    domain_name: str,
    streaming_user: TextStreamingUserSimulator,
    streaming_agent: TextStreamingLLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that agent and user states preserve input_turn_taking_buffer and output_streaming_queue."""
    orchestrator = FullDuplexOrchestrator(
        domain=domain_name,
        user=streaming_user,
        agent=streaming_agent,
        environment=get_environment(),
        task=base_task,
    )
    orchestrator.initialize()

    # Check initial states have pending chunk fields
    assert hasattr(orchestrator.agent_state, "input_turn_taking_buffer")
    assert hasattr(orchestrator.agent_state, "output_streaming_queue")
    assert hasattr(orchestrator.user_state, "input_turn_taking_buffer")
    assert hasattr(orchestrator.user_state, "output_streaming_queue")
    # After initialization, input_turn_taking_buffer may contain initial chunks
    # (initialization runs the first exchange through get_next_chunk)
    assert orchestrator.agent_state.output_streaming_queue == []
    assert orchestrator.user_state.output_streaming_queue == []


def test_full_duplex_orchestrator_with_max_steps(
    domain_name: str,
    streaming_user: TextStreamingUserSimulator,
    streaming_agent: TextStreamingLLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that FullDuplexOrchestrator respects max_steps."""
    orchestrator = FullDuplexOrchestrator(
        domain=domain_name,
        environment=get_environment(),
        user=streaming_user,
        agent=streaming_agent,
        task=base_task,
        max_steps=3,  # Very low to test termination
    )

    simulation_run = orchestrator.run()

    # Should terminate (either by completion or max steps)
    assert simulation_run is not None
    assert orchestrator.done


def test_orchestrator_streaming_agent_chunking_config(
    streaming_agent: TextStreamingLLMAgent,
):
    """Test that streaming agent has proper chunking configuration."""
    assert hasattr(streaming_agent, "chunk_by")
    assert hasattr(streaming_agent, "chunk_size")
    assert streaming_agent.chunk_by in ["chars", "words", "sentences"]
    assert streaming_agent.chunk_size > 0


def test_orchestrator_streaming_user_chunking_config(
    streaming_user: TextStreamingUserSimulator,
):
    """Test that streaming user has proper chunking configuration."""
    assert hasattr(streaming_user, "chunk_by")
    assert hasattr(streaming_user, "chunk_size")
    assert streaming_user.chunk_by in ["chars", "words", "sentences"]
    assert streaming_user.chunk_size > 0


def test_orchestrator_streaming_components_have_get_next_chunk(
    streaming_agent: TextStreamingLLMAgent,
    streaming_user: TextStreamingUserSimulator,
):
    """Test that streaming components have get_next_chunk method."""
    # Agent should have get_next_chunk
    assert hasattr(streaming_agent, "get_next_chunk")
    assert callable(streaming_agent.get_next_chunk)

    # User should have get_next_chunk
    assert hasattr(streaming_user, "get_next_chunk")
    assert callable(streaming_user.get_next_chunk)


def test_orchestrator_streaming_components_have_generate_next_message(
    streaming_agent: TextStreamingLLMAgent,
    streaming_user: TextStreamingUserSimulator,
):
    """Test that streaming components are pure full-duplex (no generate_next_message)."""
    # Agent should NOT have generate_next_message (full-duplex only)
    assert not hasattr(streaming_agent, "generate_next_message")
    assert hasattr(streaming_agent, "get_next_chunk")

    # User should NOT have generate_next_message (full-duplex only)
    assert not hasattr(streaming_user, "generate_next_message")
    assert hasattr(streaming_user, "get_next_chunk")


def test_streaming_agents_not_compatible_with_half_duplex():
    """Test that streaming agents cannot be used with half-duplex Orchestrator.

    Streaming agents (TextStreamingLLMAgent, VoiceStreamingLLMAgent) are designed
    for full-duplex mode only. They do not have generate_next_message() and
    should be rejected by the half-duplex Orchestrator.

    Note: This is by design - streaming agents use get_next_chunk() for
    tick-based communication, which is incompatible with turn-based half-duplex.
    """
    # This is a documentation test - the validation is tested in test_streaming_integration.py
    # (test_full_duplex_requires_streaming_agent and test_full_duplex_requires_streaming_user)
    pass
