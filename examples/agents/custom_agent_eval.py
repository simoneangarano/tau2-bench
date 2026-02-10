#!/usr/bin/env python3
"""
Custom agent evaluation example.

This example shows a more complete workflow:
    1. Build components manually (environment, agent, user, orchestrator)
    2. Run a simulation with full control
    3. Inspect the results in detail

This is the "power user" path -- useful for development and debugging.

Usage:
    python examples/agents/custom_agent_eval.py
"""

from copy import deepcopy
from typing import Optional

from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.message import AssistantMessage, Message, UserMessage
from tau2.environment.toolkit import Tool
from tau2.utils.llm_utils import generate

# =============================================================================
# A slightly more sophisticated agent
# =============================================================================


class VerboseAgent(HalfDuplexAgent[list[dict]]):
    """An agent that logs its reasoning before acting.

    This demonstrates how to add custom behavior (logging, pre-processing,
    post-processing) around the core LLM call.
    """

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "openai/gpt-4.1-mini",
        llm_args: Optional[dict] = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = llm_args or {}
        self.call_count = 0

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> list[dict]:
        system_prompt = (
            f"You are a helpful customer service agent.\n\n"
            f"## Policy\n{self.domain_policy}\n\n"
            f"Always follow the policy. Use tools when needed."
        )
        state = [{"role": "system", "content": system_prompt}]

        # Replay message history if provided (e.g., for tasks with prior context)
        if message_history:
            for msg in message_history:
                state.append(msg.model_dump())

        return state

    def generate_next_message(
        self, message: UserMessage, state: list[dict]
    ) -> tuple[AssistantMessage, list[dict]]:
        self.call_count += 1
        state = deepcopy(state)
        state.append({"role": "user", "content": str(message.content)})

        print(
            f"  [VerboseAgent] Turn {self.call_count}: received '{str(message.content)[:80]}...'"
        )

        response = generate(
            model=self.llm,
            tools=self.tools,
            messages=state,
            **self.llm_args,
        )

        # Log what the agent decided to do
        assistant_msg = AssistantMessage.from_llm_response(response)
        if assistant_msg.tool_calls:
            tool_names = [tc.function.name for tc in assistant_msg.tool_calls]
            print(f"  [VerboseAgent] -> Calling tools: {tool_names}")
        else:
            print(
                f"  [VerboseAgent] -> Responding with text: '{str(assistant_msg.content)[:80]}...'"
            )

        state.append(response.model_dump())
        return assistant_msg, state


# =============================================================================
# Build and run manually (no registry needed)
# =============================================================================

if __name__ == "__main__":
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner import (
        build_environment,
        build_user,
        get_tasks,
        run_simulation,
    )

    # Load a task from the mock domain
    tasks = get_tasks("mock", task_ids=["create_task_1"])
    task = tasks[0]
    print(f"Task: {task.id}")
    print(f"User scenario: {task.user_scenario.instructions[:100]}...")
    print()

    # Build the environment
    env = build_environment("mock")
    print(f"Domain policy: {env.get_policy()[:100]}...")
    print(f"Available tools: {[t.name for t in env.get_tools()]}")
    print()

    # Build our custom agent (no registry involved)
    agent = VerboseAgent(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        llm="openai/gpt-4.1-mini",
    )

    # Build the user simulator (from the registry, since we're not customizing it)
    user = build_user(
        "user_simulator",
        env,
        task,
        llm="openai/gpt-4.1-mini",
    )

    # Wire into an orchestrator
    orchestrator = Orchestrator(
        domain="mock",
        agent=agent,
        user=user,
        environment=env,
        task=task,
        max_steps=20,
        max_errors=5,
        seed=42,
    )

    # Run the simulation
    print("=" * 60)
    print("Running simulation...")
    print("=" * 60)
    result = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)

    # Inspect the results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Reward: {result.reward_info.reward}")
    print(f"Total messages: {len(result.messages)}")
    print(f"Agent calls: {agent.call_count}")

    # Print the conversation
    print()
    print("Conversation:")
    for i, msg in enumerate(result.messages):
        role = msg.role.value if hasattr(msg.role, "value") else msg.role
        content = str(msg.content)[:120] if msg.content else "(tool call/result)"
        print(f"  [{role}] {content}")

    # Print evaluation details
    if result.reward_info:
        print()
        print("Evaluation:")
        print(f"  Reward: {result.reward_info.reward}")
        if result.reward_info.reward_by_type:
            for reward_type, score in result.reward_info.reward_by_type.items():
                print(f"  {reward_type}: {score}")
