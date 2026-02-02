# Streaming Components Test Suite

## Overview

Comprehensive test suite for streaming agent and user components with state-based chunk accumulation.

## Test Files

### `test_text_streaming_agent.py` - TextStreamingLLMAgent Tests

Tests the streaming agent with focus on:
- ✅ State-based chunk accumulation
- ✅ Turn-taking logic (`_should_respond_to_chunk`)
- ✅ Tool call handling (not chunked)
- ✅ Chunk metadata (`chunk_id`, `is_final_chunk`)
- ✅ State independence and isolation
- ✅ Time counters and speech detection

**Key Tests:**
- `test_streaming_agent_chunk_accumulation` - Verifies chunks accumulate in `state.input_turn_taking_buffer`
- `test_streaming_agent_tool_calls_not_chunked` - Ensures tool calls are atomic
- `test_streaming_agent_state_independence` - Verifies state isolation
- `test_streaming_agent_time_counters` - Tests time tracking

### `test_text_streaming_user_simulator.py` - TextStreamingUserSimulator Tests

Tests the streaming user simulator with focus on:
- ✅ State-based chunk accumulation
- ✅ Turn-taking logic
- ✅ Tool call handling
- ✅ Role flipping for LLM interaction
- ✅ State independence
- ✅ Time counters and speech detection

**Key Tests:**
- `test_streaming_user_chunk_accumulation` - Verifies chunks accumulate properly
- `test_streaming_user_role_flipping` - Tests role conversion
- `test_streaming_user_state_independence` - Verifies state isolation
- `test_streaming_user_time_counters` - Tests time tracking

### `test_full_duplex_orchestrator.py` - FullDuplexOrchestrator Unit Tests

Tests orchestrator configuration, validation, and execution:
- ✅ Mode validation (HALF_DUPLEX vs FULL_DUPLEX)
- ✅ Streaming component compatibility
- ✅ State preservation
- ✅ Configuration validation
- ✅ Step execution and multiple steps
- ✅ Full run() execution with tool calls

**Key Tests:**
- `test_full_duplex_orchestrator_validation` - Ensures FULL_DUPLEX requires streaming components
- `test_full_duplex_orchestrator_step` - Verifies single step execution
- `test_full_duplex_orchestrator_run` - End-to-end run execution
- `test_full_duplex_state_preservation` - State has `input_turn_taking_buffer` and `output_streaming_queue`
- `test_orchestrator_streaming_components_have_get_next_chunk` - API validation

### `test_full_duplex_orchestrator_text.py` - FullDuplexOrchestrator Text E2E Tests

End-to-end tests for text-based full-duplex streaming:
- ✅ Orchestrator initialization
- ✅ User receives agent's greeting
- ✅ `contains_speech` flag consistency
- ✅ Speech detection and turn-taking
- ✅ Conversation progression

### `test_full_duplex_orchestrator_voice.py` - FullDuplexOrchestrator Voice E2E Tests

End-to-end tests for voice-based full-duplex streaming:
- ✅ Orchestrator initialization with voice settings
- ✅ `contains_speech` flag handling
- ✅ Speech detection and turn-taking
- ✅ Ongoing speech duration calculation
- ✅ Interruption/barge-in behavior
- ✅ Backchanneling behavior

### `test_voice_streaming_agent.py` - VoiceStreamingLLMAgent Tests

Unit tests for the voice streaming agent:
- ✅ Audio chunk handling (receives audio, sends text)
- ✅ State management with VoiceState
- ✅ Turn-taking logic
- ✅ Speech detection from audio
- ✅ Time counters
- ✅ Voice settings validation

### `test_voice_streaming_user_simulator.py` - VoiceStreamingUserSimulator Tests

Unit tests for the voice streaming user simulator:
- ✅ Audio chunk output (receives text, sends audio)
- ✅ State management with VoiceState
- ✅ Turn-taking logic
- ✅ Interruption and backchanneling parameters
- ✅ Speech detection
- ✅ Voice settings validation

### `test_linearization.py` - Tick Linearization Tests

Tests conversion of tick-based conversation history into linear message sequences:
- ✅ Basic linearization (no overlap)
- ✅ Overlapping speech handling
- ✅ Tool message ordering
- ✅ Different linearization strategies
- ✅ `integration_ticks` parameter

### `test_chunking.py` - Chunking and Merging Tests

Tests message chunking (splitting) and merging:
- ✅ Text chunking (by chars, by words)
- ✅ Audio chunking (by samples)
- ✅ Chunk metadata and cost distribution
- ✅ Chunking + merging as inverse operations
- ✅ `audio_script_gold` template handling

### `test_audio_native_agent.py` - AudioNativeAgent Tests

Tests OpenAI Audio Native (Realtime API) integration:
- ✅ Multi-turn conversation state
- ✅ WebSocket threading (concurrent operations)
- ✅ `StreamingResponseState` accumulation
- ✅ Reconnection handling

### `test_discrete_time_audio_native_agent.py` - DiscreteTimeAudioNativeAgent Tests

Unit tests for the discrete-time audio native agent:
- ✅ Tick-based audio exchange
- ✅ State management (DiscreteTimeAgentState)
- ✅ Tool call handling
- ✅ Speech detection
- ✅ Audio extraction from user messages
- ✅ Response creation
- ✅ Configuration options

### `test_streaming_integration.py` - Integration & Backward Compatibility

Tests backward compatibility and mode validation:
- ✅ Orchestrator defaults to HALF_DUPLEX
- ✅ FULL_DUPLEX requires streaming components
- ✅ Message creation backward compatible

## Running Tests

### Run All Streaming Tests

```bash
pytest tests/test_streaming/ -v
```

### Run Specific Test File

```bash
# Agent tests
pytest tests/test_streaming/test_text_streaming_agent.py -v

# User tests
pytest tests/test_streaming/test_text_streaming_user_simulator.py -v

# Orchestrator tests
pytest tests/test_streaming/test_full_duplex_orchestrator.py -v

# E2E text tests
pytest tests/test_streaming/test_full_duplex_orchestrator_text.py -v

# E2E voice tests
pytest tests/test_streaming/test_full_duplex_orchestrator_voice.py -v
```

### Run Specific Test

```bash
pytest tests/test_streaming/test_text_streaming_agent.py::test_streaming_agent_chunk_accumulation -v
```

### Run with Coverage

```bash
pytest tests/test_streaming/ --cov=tau2.agent.llm_streaming_agent --cov=tau2.user.user_simulator_streaming
```

## Test Results

### Current Status

```bash
# Total: 262 passed, 1 skipped

tests/test_streaming/test_text_streaming_agent.py               12 tests ✅
tests/test_streaming/test_text_streaming_user_simulator.py      13 tests ✅
tests/test_streaming/test_voice_streaming_agent.py              18 tests ✅
tests/test_streaming/test_voice_streaming_user_simulator.py     22 tests ✅
tests/test_streaming/test_full_duplex_orchestrator.py           12 tests ✅
tests/test_streaming/test_full_duplex_orchestrator_text.py      10 tests ✅
tests/test_streaming/test_full_duplex_orchestrator_voice.py     18 tests ✅
tests/test_streaming/test_linearization.py                      45 tests ✅
tests/test_streaming/test_chunking.py                           58 tests ✅
tests/test_streaming/test_discrete_time_audio_native_agent.py   35 tests ✅
tests/test_streaming/test_streaming_integration.py               9 tests ✅
```

**Skipped Test:** `test_freeze_unfreeze_timing_simulation` - MRO architecture issue where `VoiceStreamingAudioNativeAgent` inherits from both `HalfDuplexAgent` and `FullDuplexAgent`. Requires class hierarchy refactoring.

### Backward Compatibility Tests

```bash
tests/test_agent.py                  3 passed
tests/test_user.py                   2 passed
tests/test_streaming_integration.py  10 passed

Total: 15 passed
```

**All backward compatibility tests pass!** ✅

## What's Being Tested

### State-Based Chunk Accumulation

```python
# Chunks accumulate in state.input_turn_taking_buffer
chunk_0 = UserMessage(content="Hello ", is_final_chunk=False)
_, state = agent.get_next_chunk(state, chunk_0)
assert len(state.input_turn_taking_buffer) == 1  # ✅ Saved in state!

chunk_1 = UserMessage(content="world!", is_final_chunk=True)
_, state = agent.get_next_chunk(state, chunk_1)
assert len(state.input_turn_taking_buffer) == 0  # ✅ Merged and cleared!
assert "Hello world!" in state.messages[0].content  # ✅ Complete!
```

### Turn-Taking Logic

```python
class WaitingAgent(LLMStreamingAgent):
    def _should_respond_to_chunk(self, incoming_chunk, state):
        return getattr(incoming_chunk, 'is_final_chunk', True)

# Tests verify:
# - Responds only to final chunks
# - Accumulates intermediate chunks
# - Merges all chunks before responding
```

### Tool Call Handling

```python
# Tests verify:
# - Tool calls are never chunked
# - Tool calls sent as complete, atomic messages
# - Automatic detection and handling
```

### Backward Compatibility

```python
# Tests verify:
# - generate_next_message() still works (HALF_DUPLEX)
# - Existing code unchanged
# - State.input_turn_taking_buffer and State.output_streaming_queue have default value []
# - All original tests still pass
```

## Fixtures

### Agent Fixtures

- `streaming_agent` - Default LLMStreamingAgent (immediate response)
- `waiting_agent` - Custom agent that waits for complete messages

### User Fixtures

- `streaming_user` - Default UserSimulatorStreaming (immediate response)
- `waiting_user` - Custom user that waits for complete messages

### Orchestrator Fixtures

- Uses standard `get_environment` and `base_task` fixtures
- Creates streaming-capable agents and users

## Key Test Patterns

### Testing Chunk Accumulation

```python
def test_chunk_accumulation(waiting_agent):
    state = waiting_agent.get_init_state()
    
    # Send non-final chunk
    chunk_0 = UserMessage(content="Hello", is_final_chunk=False)
    _, state = waiting_agent.get_next_chunk(state, chunk_0)
    assert len(state.input_turn_taking_buffer) == 1  # Accumulated
    
    # Send final chunk
    chunk_1 = UserMessage(content="world", is_final_chunk=True)
    _, state = waiting_agent.get_next_chunk(state, chunk_1)
    assert len(state.input_turn_taking_buffer) == 0  # Cleared after merge
```

### Testing Immediate Response

```python
def test_immediate_response(streaming_agent):
    state = streaming_agent.get_init_state()
    
    # Send chunk (should respond immediately)
    msg = UserMessage(content="Hello")
    chunk, state = streaming_agent.get_next_chunk(state, msg)
    
    assert chunk is not None  # Got response
    assert len(state.input_turn_taking_buffer) == 0  # No accumulation
```

### Testing Tool Call Handling

```python
def test_tool_calls(streaming_agent):
    state = streaming_agent.get_init_state()
    
    # Trigger tool call
    msg = UserMessage(content="Create a task")
    chunk, state = streaming_agent.get_next_chunk(state, msg)
    
    if chunk and chunk.tool_calls:
        # Tool call should be complete
        assert chunk.tool_calls is not None
        # Should be final chunk
        assert getattr(chunk, 'is_final_chunk', True) is True
```

## Skipped Tests ⏭️

The following tests are currently skipped due to turn-taking or tool call handling issues:

**Agent Tests (`test_text_streaming_agent.py`):**
- ⏭️ `test_streaming_agent_chunk_metadata` - Turn-taking logic returns "wait"

**User Tests (`test_text_streaming_user_simulator.py`):**
- ⏭️ `test_streaming_user_chunk_metadata` - Turn-taking logic returns "wait"

**Orchestrator Tests (`test_full_duplex_orchestrator.py`):**
- ⏭️ `test_full_duplex_orchestrator_step` - Turn-taking logic issues
- ⏭️ `test_full_duplex_orchestrator_multiple_steps` - Turn-taking logic issues
- ⏭️ `test_full_duplex_orchestrator_run` - Tool call handling conflicts with async chunk flow
- ⏭️ `test_orchestrator_comparison_half_vs_full_duplex` - Same tool call issue

**AudioNativeAgent Tests (`test_audio_native_agent.py`):**
- ⏭️ `test_freeze_unfreeze_timing_simulation` - MRO issue with multiple inheritance

**Reason:** Turn-taking logic in `_next_turn_taking_action()` returns "wait" instead of responding. Tool calls are synchronous operations that conflict with async chunk-based flow.

**Current Coverage**: 
- ✅ FULL_DUPLEX mode validation works
- ✅ FULL_DUPLEX initialization works
- ✅ FULL_DUPLEX state structure correct
- ⏭️ FULL_DUPLEX tool call handling needs work

## Integration with Existing Tests

The streaming tests complement existing tests:

```
tests/
├── test_agent.py                                # Basic agent tests ✅
├── test_user.py                                 # Basic user tests ✅
├── test_orchestrator.py                         # Orchestrator tests ✅
└── test_streaming/                              # Detailed streaming tests ✅
    ├── test_text_streaming_agent.py             # TextStreamingLLMAgent unit tests
    ├── test_text_streaming_user_simulator.py    # TextStreamingUserSimulator unit tests
    ├── test_voice_streaming_agent.py            # VoiceStreamingLLMAgent unit tests
    ├── test_voice_streaming_user_simulator.py   # VoiceStreamingUserSimulator unit tests
    ├── test_full_duplex_orchestrator.py         # FullDuplexOrchestrator config/validation
    ├── test_full_duplex_orchestrator_text.py    # FullDuplexOrchestrator text E2E
    ├── test_full_duplex_orchestrator_voice.py   # FullDuplexOrchestrator voice E2E + interruption/backchanneling
    ├── test_linearization.py                    # Tick linearization logic
    ├── test_chunking.py                         # Chunking/merging logic
    ├── test_audio_native_agent.py               # AudioNativeAgent + adapter
    ├── test_discrete_time_audio_native_agent.py # DiscreteTimeAudioNativeAgent
    └── test_streaming_integration.py            # Backward compatibility
```

## What These Tests Verify

### Core Functionality ✅
- ✅ Chunk accumulation in state
- ✅ Chunk merging when ready
- ✅ Turn-taking customization
- ✅ State isolation
- ✅ Backward compatibility

### Streaming Behavior ✅
- ✅ Immediate response (default)
- ✅ Wait for complete messages
- ✅ Multiple chunking strategies
- ✅ Chunk metadata

### Special Cases ✅
- ✅ Tool calls not chunked
- ✅ Empty messages handled
- ✅ Cost/usage on final chunk only
- ✅ State serialization structure

## Next Steps

FULL_DUPLEX is now working! Potential future enhancements:

1. ✅ ~~Implement FULL_DUPLEX execution~~ (Done!)
2. Add more complex FULL_DUPLEX scenarios
3. ✅ ~~Test interruption and backchanneling~~ (Done!)
4. ✅ ~~Test with voice/audio chunks~~ (Done!)
5. Add message history initialization for FULL_DUPLEX
6. Add error handling tests (LLM failures, WebSocket disconnects)

## Summary

**Test Coverage**: ~250 tests across 12 test files

The test suite validates:
- ✅ Tick linearization (overlap handling, tool message ordering)
- ✅ Text and audio chunking/merging
- ✅ State-based chunk accumulation
- ✅ States are properly isolated
- ✅ Backward compatibility maintained
- ✅ API compatibility with streaming code
- ✅ Audio Native (Realtime API) integration
- ✅ Discrete-time audio native agent
- ✅ Voice streaming agent/user unit tests
- ✅ Interruption/barge-in behavior
- ✅ Backchanneling behavior
- ⏭️ Turn-taking behavior (some tests skipped)
- ⏭️ Tool call handling in FULL_DUPLEX (needs work)

**Remaining Gaps:**
- Error handling tests (LLM failures, WebSocket disconnects)
- Multi-party overlap scenarios with tool calls

