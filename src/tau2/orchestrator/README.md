# Orchestrator Module

This module provides orchestrators for managing simulations between agents, users, and environments. Each orchestrator offers different communication patterns and capabilities.

## Overview

| Orchestrator | Communication Mode | Tool Execution | Participant Interface | Primary Use Case |
|--------------|-------------------|----------------|----------------------|------------------|
| `Orchestrator` | Half-duplex (turn-based) | Synchronous | `generate_next_message()` | Standard benchmarking |
| `FullDuplexOrchestrator` | Full-duplex (streaming) | Synchronous | `get_next_chunk()` | Real-time streaming |
| `AsyncToolFullDuplexOrchestrator` | Full-duplex (streaming) | Async (configurable latency) | `process_incoming_messages()` or `get_next_chunk()` | Realistic tool timing |
| `EventDrivenOrchestrator` | Full-duplex (event-based) | Async with timeouts | `process_event_batch()` | Complex async scenarios |

---

## 1. Orchestrator (Half-Duplex)

The standard orchestrator for turn-based communication.

### Communication Pattern

```
Agent ──message──> User ──message──> Agent ──tool_call──> Environment ──result──> Agent
```

Each participant takes turns sending complete messages. Only one party "speaks" at a time.

### Participant Interface

```python
class MyAgent(HalfDuplexAgent):
    def generate_next_message(
        self, 
        message: ValidAgentInputMessage, 
        state: AgentState
    ) -> tuple[AssistantMessage, AgentState]:
        """Generate a complete response to the incoming message."""
        ...
```

### Tool Execution

- **Synchronous**: Tool calls block until complete
- **Immediate**: Results returned in the same step

### Trajectory Structure

```python
trajectory: list[Message]  # Flat list of messages
```

### Compatible Classes

| Role | Classes |
|------|---------|
| Agent | `LLMAgent`, `LLMAgentGT`, `LLMSoloAgent`, `VoiceLLMAgent` |
| User | `UserSimulator`, `VoiceUserSimulator`, `DummyUser` |

### Usage

```python
from tau2.orchestrator.orchestrator import Orchestrator

orchestrator = Orchestrator(
    domain="airline",
    agent=LLMAgent(tools, policy, llm="gpt-4"),
    user=UserSimulator(instructions, llm="gpt-4"),
    environment=environment,
    task=task,
    max_steps=100,
    seed=42,
)
result = orchestrator.run()
```

---

## 2. FullDuplexOrchestrator

Orchestrator for real-time streaming communication where both parties can "speak" simultaneously.

### Communication Pattern

```
     Tick 0          Tick 1          Tick 2
Agent: [chunk0] ───> [chunk1] ───> [chunk2] ───>
User:  [chunk0] ───> [chunk1] ───> [chunk2] ───>
```

Both agent and user generate chunks each tick. Chunks can overlap (simultaneous speech).

### Participant Interface

```python
class MyStreamingAgent(FullDuplexAgent):
    def get_next_chunk(
        self,
        state: StreamingState,
        incoming_chunk: Optional[Message] = None,
    ) -> tuple[AssistantMessage, StreamingState]:
        """Generate the next chunk based on incoming chunk."""
        ...
```

### Tool Execution

- **Synchronous**: Tool calls block and execute immediately
- **Within tick**: Results returned before tick completes

### Trajectory Structure

```python
ticks: list[Tick]  # Tick-grouped events

@dataclass
class Tick:
    tick_id: int
    timestamp: str
    agent_chunk: Optional[AssistantMessage]
    user_chunk: Optional[UserMessage]
    agent_tool_results: list[ToolMessage]
    user_tool_results: list[ToolMessage]
```

### Compatible Classes

| Role | Classes |
|------|---------|
| Agent | `TextStreamingLLMAgent`, `VoiceStreamingLLMAgent`, `DiscreteTimeAudioNativeAgent` |
| User | `TextStreamingUserSimulator`, `VoiceStreamingUserSimulator` |

### Usage

```python
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator

orchestrator = FullDuplexOrchestrator(
    domain="airline",
    agent=TextStreamingLLMAgent(tools, policy, llm="gpt-4", chunk_by="words", chunk_size=5),
    user=TextStreamingUserSimulator(instructions, llm="gpt-4", chunk_by="words", chunk_size=5),
    environment=environment,
    task=task,
    tick_duration=0.1,  # Optional: real-time pacing
)
orchestrator.initialize()
while not orchestrator.done:
    orchestrator.step()
```

---

## 3. AsyncToolFullDuplexOrchestrator

Full-duplex orchestrator with realistic async tool execution timing.

### Communication Pattern

```
Tick 0: Agent makes tool call → queued (completes at tick 3)
Tick 1: Agent receives user chunk (no tool result yet)
Tick 2: Agent receives user chunk (no tool result yet)
Tick 3: Agent receives user chunk + tool result
```

Tool calls take multiple ticks to complete, simulating real-world latency.

### Participant Interface

**Native interface (recommended):**
```python
class MyAsyncToolAgent(BaseAsyncToolStreamingParticipant):
    def process_incoming_messages(
        self,
        state: AsyncToolStreamingAgentState,
        incoming: IncomingMessages,
    ) -> tuple[AsyncToolOutput, AsyncToolStreamingAgentState]:
        """Process participant chunk + environment responses together."""
        ...
```

**Legacy interface (also supported):**
```python
class MyStreamingAgent(FullDuplexAgent):
    def get_next_chunk(self, state, incoming_chunk):
        """Called sequentially for each env message, then participant chunk."""
        ...
```

### IncomingMessages Structure

```python
@dataclass
class IncomingMessages:
    participant_chunk: Optional[ChunkType]  # From other participant
    environment_messages: list[EnvironmentMessage]  # Completed tool results
```

### Tool Execution

- **Asynchronous**: Tool calls queued with configurable latency
- **Deferred**: Results delivered in future ticks
- **Configurable**: Per-tool latency via `tool_execution_ticks_fn`

### Trajectory Structure

```python
ticks: list[AsyncTick]

@dataclass
class AsyncTick:
    tick_id: int
    timestamp: str
    agent_chunk: Optional[AssistantMessage]
    user_chunk: Optional[UserMessage]
    # Tool calls SUBMITTED this tick
    agent_tool_calls_submitted: list[ToolCall]
    user_tool_calls_submitted: list[ToolCall]
    # Tool results RECEIVED this tick
    agent_tool_results_received: list[ToolMessage]
    user_tool_results_received: list[ToolMessage]
```

### Compatible Classes

| Role | Interface | Classes |
|------|-----------|---------|
| Agent | Native | `AsyncToolTextStreamingAgent` |
| Agent | Legacy | `TextStreamingLLMAgent`, `VoiceStreamingLLMAgent` |
| User | Native | `AsyncToolTextStreamingUser` |
| User | Legacy | `TextStreamingUserSimulator`, `VoiceStreamingUserSimulator` |

### Usage

```python
from tau2.orchestrator.full_duplex_orchestrator_async_tools import AsyncToolFullDuplexOrchestrator
from tau2.agent.llm_async_tool_streaming_agent import AsyncToolTextStreamingAgent
from tau2.user.user_simulator_async_tool_streaming import AsyncToolTextStreamingUser

orchestrator = AsyncToolFullDuplexOrchestrator(
    domain="airline",
    agent=AsyncToolTextStreamingAgent(tools, policy, llm="gpt-4"),
    user=AsyncToolTextStreamingUser(instructions, llm="gpt-4"),
    environment=environment,
    task=task,
    tool_execution_ticks=3,  # Default latency
    tool_execution_ticks_fn=lambda tc: 5 if tc.name == "slow_api" else 1,  # Per-tool latency
)
```

---

## 4. EventDrivenOrchestrator

Most flexible orchestrator with a unified event system for all async behaviors.

### Communication Pattern

```
     ┌─────────────────────────────────────────────────────┐
     │                   Event Queue                        │
     │  [tool_result, timeout, notification, chunk, ...]   │
     └───────────────────────┬─────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
    Agent Batch         User Batch          (future ticks)
```

All async behaviors (tool results, timeouts, notifications) are modeled as events.

### Participant Interface

```python
class MyEventDrivenAgent(BaseEventDrivenParticipant):
    def process_event_batch(
        self,
        state: EventDrivenAgentState,
        batch: EventBatch,
    ) -> tuple[ParticipantOutput, EventDrivenAgentState]:
        """Process all events for this tick."""
        # batch.tool_results - completed tool calls
        # batch.participant_chunks - messages from other participant
        # batch.timeouts - timed out operations
        # batch.notifications - external notifications
        ...
```

### EventBatch Structure

```python
@dataclass
class EventBatch:
    tick: int
    target: Literal["agent", "user"]
    events: list[Event]
    
    @property
    def tool_results(self) -> list[ToolMessage]: ...
    @property
    def participant_chunks(self) -> list[Message]: ...
    @property
    def timeouts(self) -> list[Event]: ...
    @property
    def notifications(self) -> list[Event]: ...
```

### Event Types

```python
class EventType(str, Enum):
    PARTICIPANT_CHUNK = "participant_chunk"
    TOOL_RESULT = "tool_result"
    TIMEOUT = "timeout"
    NOTIFICATION = "notification"
    SYSTEM_MESSAGE = "system_message"
    CANCEL = "cancel"
    INTERRUPT = "interrupt"
    # ... extensible
```

### Tool Execution

- **Event-based**: Tool calls scheduled as events
- **Timeout support**: Optional timeout events
- **Cancellable**: Events can be cancelled

### Trajectory Structure

```python
ticks: list[EventTick]

@dataclass
class EventTick:
    tick_id: int
    timestamp: str
    # Events DELIVERED this tick
    agent_events_received: Optional[EventBatch]
    user_events_received: Optional[EventBatch]
    # Outputs PRODUCED this tick
    agent_chunk: Optional[AssistantMessage]
    user_chunk: Optional[UserMessage]
    # Tool calls SCHEDULED this tick
    agent_tool_calls_scheduled: list[ToolCall]
    user_tool_calls_scheduled: list[ToolCall]
```

### Compatible Classes

| Role | Classes |
|------|---------|
| Agent | `EventDrivenTextAgent` |
| User | `EventDrivenTextUser` |

### Usage

```python
from tau2.orchestrator.event_driven_orchestrator import EventDrivenOrchestrator
from tau2.agent.llm_event_driven_agent import EventDrivenTextAgent
from tau2.user.user_simulator_event_driven import EventDrivenTextUser

orchestrator = EventDrivenOrchestrator(
    domain="airline",
    agent=EventDrivenTextAgent(tools, policy, llm="gpt-4"),
    user=EventDrivenTextUser(instructions, llm="gpt-4"),
    environment=environment,
    task=task,
    default_tool_latency=3,
    tool_timeout_ticks=10,  # Enable timeouts
)

# Inject external events
orchestrator.inject_notification("agent", {"type": "system_update", "message": "..."})
```

---

## Choosing the Right Orchestrator

| Scenario | Recommended Orchestrator |
|----------|-------------------------|
| Standard benchmarking, simple evaluation | `Orchestrator` |
| Real-time voice/streaming demo | `FullDuplexOrchestrator` |
| Testing with realistic API latencies | `AsyncToolFullDuplexOrchestrator` |
| Complex scenarios with timeouts, retries, external events | `EventDrivenOrchestrator` |

---

## Output Types Comparison

| Orchestrator | Participant Output | Tool Calls |
|--------------|-------------------|------------|
| `Orchestrator` | `Message` | Embedded in message (`msg.tool_calls`) |
| `FullDuplexOrchestrator` | `Message` | Embedded in message |
| `AsyncToolFullDuplexOrchestrator` | `AsyncToolOutput` | Separate (`output.tool_calls`) |
| `EventDrivenOrchestrator` | `ParticipantOutput` | Separate (`output.tool_calls`) |

### AsyncToolOutput / ParticipantOutput

```python
@dataclass
class ParticipantOutput:
    chunk: Optional[Message] = None      # Speech/content
    tool_calls: list[ToolCall] = []      # Tool calls to schedule
    should_stop: bool = False            # Stop signal
```

This separation provides cleaner handling of tool calls vs speech content.

---

## Migration Guide

### From `Orchestrator` to `FullDuplexOrchestrator`

1. Replace `LLMAgent` with `TextStreamingLLMAgent`
2. Replace `UserSimulator` with `TextStreamingUserSimulator`
3. Add chunking parameters (`chunk_by`, `chunk_size`)

### From `FullDuplexOrchestrator` to `AsyncToolFullDuplexOrchestrator`

1. (Optional) Replace with native async-tool classes for better batch processing
2. Add `tool_execution_ticks` parameter

### From `AsyncToolFullDuplexOrchestrator` to `EventDrivenOrchestrator`

1. Replace with event-driven classes (`EventDrivenTextAgent`, `EventDrivenTextUser`)
2. Update from `process_incoming_messages()` to `process_event_batch()`
3. Handle additional event types (timeouts, notifications)

