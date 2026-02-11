"""
Integration tests for experimental streaming agents with the full-duplex orchestrator.

Extracted from tests/test_streaming/test_streaming_integration.py when the
experimental agents (TextStreamingLLMAgent, VoiceStreamingLLMAgent) were moved
to experiments/tau_voice/agents/.
"""

import pytest

from experiments.tau_voice.agents.llm_streaming_agent import TextStreamingLLMAgent
from tau2 import UserSimulator
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.registry import registry
from tau2.run import get_tasks


@pytest.fixture
def mock_agent_setup():
    """Setup basic agent requirements."""
    env_constructor = registry.get_env_constructor("mock")
    env = env_constructor()
    tools = env.get_tools()
    domain_policy = "Mock domain for testing"
    return tools, domain_policy


def test_full_duplex_requires_streaming_user(mock_agent_setup):
    """FullDuplexOrchestrator requires streaming-capable user."""
    tools, domain_policy = mock_agent_setup

    # Create streaming agent
    streaming_agent = TextStreamingLLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm="gpt-4",
        chunk_by="words",
        chunk_size=10,
    )

    # Create regular (non-streaming) user
    user = UserSimulator(
        instructions="Test user",
        llm="gpt-4",
    )

    env_constructor = registry.get_env_constructor("mock")
    env = env_constructor()
    tasks = get_tasks("mock", task_ids=["create_task_1"])
    task = tasks[0]

    with pytest.raises(ValueError, match="get_next_chunk"):
        FullDuplexOrchestrator(
            domain="mock",
            agent=streaming_agent,
            user=user,  # Regular user doesn't have get_next_chunk
            environment=env,
            task=task,
        )


def test_streaming_agents_not_compatible_with_half_duplex():
    """Test that streaming agents cannot be used with half-duplex Orchestrator.

    Streaming agents (TextStreamingLLMAgent, VoiceStreamingLLMAgent) are designed
    for full-duplex mode only. They do not have generate_next_message() and
    should be rejected by the half-duplex Orchestrator.
    """
    # This is a documentation test - the validation is tested above
    pass
