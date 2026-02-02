
# Agent Developer Guide

## Understanding the Environment

To develop an agent for a specific domain, you first need to understand the domain's policy and available tools. Start by running the environment server for your target domain:

```bash
tau2 domain <domain>
```

This will start a server and automatically open your browser to the API documentation page (ReDoc). Here you can:
- Review the available tools (API endpoints) for the domain
- Understand the policy requirements and constraints
- Test API calls directly through the documentation interface

## Developing an Agent

Implement the `HalfDuplexAgent` class for turn-based agents or `FullDuplexAgent` for streaming agents.

Register your agent in `src/tau2/agent/registry.py`
```python
registry.register_agent(MyAgent, "my_agent")
```

## Available Agent Types

### LLMAgent (Half-Duplex)
The basic LLM-powered agent for turn-based communication:

```python
from tau2.agent import LLMAgent

agent = LLMAgent(
    tools=tools,
    domain_policy=policy,
    llm="gpt-4",
    llm_args={"temperature": 0.7}
)

# Use in half-duplex mode
message, state = agent.generate_next_message(user_msg, state)
```

### TextStreamingLLMAgent (Full-Duplex)
Streaming agent with full-duplex capabilities:

```python
from tau2.agent import TextStreamingLLMAgent

agent = TextStreamingLLMAgent(
    tools=tools,
    domain_policy=policy,
    llm="gpt-4",
    chunk_by="words",  # "chars", "words", or "sentences"
    chunk_size=10      # Number of units per chunk
)

# FULL_DUPLEX mode (symmetric chunk-based streaming)
state = agent.get_init_state()
chunk, state = agent.get_next_chunk(state, incoming_chunk)
```

#### Customizing Turn-Taking Logic

Override `_next_turn_taking_action()` to implement custom turn-taking:

```python
class MyStreamingAgent(TextStreamingLLMAgent):
    def _next_turn_taking_action(self, state):
        # Custom logic for deciding when to speak
        return basic_turn_taking_policy(
            state,
            wait_to_respond_threshold_other=2,
            wait_to_respond_threshold_self=4,
        )
```

## Testing Your Agent
You can now use the command:
```bash
tau2 run \
  --domain <domain> \
  --agent my_agent \
  --agent-llm <llm_name> \
  --user-llm <llm_name> \
  ...
```

## Communication Protocols

### Half-Duplex (HalfDuplexAgent)
Traditional turn-based communication where one party speaks while the other listens. Uses `generate_next_message()`.

### Full-Duplex (FullDuplexAgent)
Symmetric chunk-based streaming where both parties can send and receive chunks simultaneously, enabling truly interactive conversations with interruptions and overlapping speech. Uses `get_next_chunk()`.
