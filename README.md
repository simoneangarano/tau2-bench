# $\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment

[![python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![arXiv](http://img.shields.io/badge/cs.AI-arXiv%3A2506.07982-B31B1B.svg?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2506.07982)
[![blog](https://img.shields.io/badge/blog-tau2--bench-green)](https://sierra.ai/blog/benchmarking-agents-in-collaborative-real-world-scenarios)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/sierra.svg?style=social&label=Follow%20%40SierraPlatform)](https://x.com/SierraPlatform/status/1932464265207889974)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/posts/sierra_last-year-we-introduced-%F0%9D%9C%8F-bench-a-benchmark-activity-7338229693898231809-F8L4?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAdc8goBmhEsiEo1_t_XSJbAnY4_zMfAWcE)
[![Leaderboard](https://img.shields.io/badge/🏆_Live_Leaderboard-taubench.com-brightgreen?style=flat)](https://taubench.com)

<div align="center">
<img src="figs/overview.png" width="95%" alt="System Overview"><br>
<em>Figure 1: τ²-bench allows users to interact with the agent and the environment</em>
</div>

<div align="center">
<img src="figs/traj.png" width="95%" alt="Trajectory"><br>
<em>Figure 2: Trajectory of a conversation between an agent and a user</em>
</div>

## 🆕 What's New

### 🎙️ Voice Full-Duplex Now Supported (New!)
τ²-bench now supports **voice full-duplex** via Audio Native mode with multiple real-time providers:

```bash
tau2 run --domain retail --audio-native --num-tasks 1 --verbose-logs
```

Supported providers: OpenAI Realtime, Gemini Live, Nova Sonic, xAI, Deepgram, Qwen

[**→ See Audio Native Documentation**](src/tau2/voice/audio_native/README.md)

### 🤖 Reinforcement Learning Support (New!)
τ²-bench now supports RL training with a Gymnasium-compatible interface:

- **🏋️ Train RL Agents**: Use the gym interface to train agents with popular RL frameworks. 
- **🎮 Play as Agent or User**: Interactive mode lets you control either the agent or the user in conversations
- **📊 Train/Test Splits**: To help support experiments around training Agents and evaluating them, all domains include standardized task splits for proper train/test evaluation.

> **⚠️ IMPORTANT FOR BACKWARD COMPATIBILITY**: If you are just evaluating an agent (not training), you **MUST** use the `base` task split to evaluate on the complete task set that matches the original τ²-bench structure. This ensures your results are comparable to previous evaluations and maintains consistency with the established benchmark. (If you don't specify a task split, it will default to `base`.)
- **🔧 Gymnasium Compatible**: Standard gym interface works with existing RL tools and libraries

[**→ See Gym Documentation**](src/tau2/gym/README.md) | [**→ Try CLI Play Mode**](#interactive-play-mode)

### 🏆 Live Leaderboard (v0.2.0)
The τ²-bench leaderboard is now live at **[taubench.com](https://taubench.com)**! 

- **📊 Interactive Rankings**: Compare model performance across all domains
- **📱 Mobile-Friendly**: View results on any device  
- **🔍 Detailed Analysis**: Explore trajectories and conversation flows
- **📥 Easy Submission**: Submit your results directly through the interface

[**→ Visit the Leaderboard**](https://taubench.com) | [**→ Submit Your Results**](#leaderboard-submission)

## Overview

$\tau^2$-bench implements a simulation framework for evaluating customer service agents across various domains.

**$\tau^2$-bench is the new iteration of the original $\tau$-bench**, featuring code fixes and an additional telecom domain.

### 🎙️ Full-Duplex and Voice Support (New!)

τ²-bench now supports **full-duplex communication modes and voice interactions**:

- **🔄 Communication Modes**:
  - **HALF_DUPLEX** (default): Turn-based communication (one speaks, then the other)
  - **FULL_DUPLEX**: Simultaneous bidirectional communication (both can speak at once)
- **Current Implementation**:
  - ✅ **Text Full-Duplex**: Chunk-based text with interruption support
  - ✅ **Voice Half-Duplex**: Turn-based voice conversations with synthesis and transcription
  - ✅ **Voice Full-Duplex**: Audio native mode with multiple providers (OpenAI, Gemini, xAI, Nova, Deepgram)
- **🔧 Easy Integration**: Use pre-built full-duplex/voice classes or wrap existing agents
- **⚡ Opt-in Feature**: Existing code works unchanged

**Quick Start - Text Full-Duplex:**
```bash
tau2 run --domain mock --agent text_streaming_llm_agent \
  --user text_streaming_user_simulator --num-tasks 1 --streaming-enabled
```

**Quick Start - Voice Half-Duplex:**
```bash
tau2 run --domain mock --agent voice_llm_agent \
  --user voice_user_simulator --voice-enabled \
  --voice-transcription-model nova-3 --num-tasks 1
```

[**→ See Full-Duplex Documentation**](#full-duplex-mode) | [**→ See Voice Documentation**](#voice-mode)

### 🎤 Audio Native Mode (Experimental)

τ²-bench supports **end-to-end voice processing** using realtime multimodal models:

- **🧠 Native Multimodal**: Uses models like `gpt-4o-realtime-preview` that handle audio input/output directly
- **🔄 Full-Duplex Voice**: Tick-based discrete-time simulation with real-time audio streaming
- **🎯 Server VAD**: OpenAI's built-in Voice Activity Detection for speech start/end
- **⚡ Fast-Forward Mode**: Run simulations faster than real-time for efficient evaluation

**Quick Start - Audio Native via CLI:**
```bash
tau2 run --domain retail --audio-native --num-tasks 1 --verbose-logs
```

**With custom configuration:**
```bash
tau2 run --domain retail --audio-native \
  --tick-duration 0.2 \
  --max-steps-seconds 240 \
  --speech-complexity low \
  --verbose-logs
```

[**→ See Audio Native Documentation**](#audio-native-mode)

Each domain specifies:
- a policy that the agent must follow
- a set of tools that the agent can use
- a set of tasks to evaluate the agent's performance
- Optionally: A set of tools that the user simulator can use

Domains are:
- `mock`
- `airline`
- `retail`
- `telecom`

All the information that an agent developer needs to build an agent for a domain can be accessed through the domain's API docs. See [View domain documentation](#view-domain-documentation) for more details.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench
```

2. Create a new environment (optional)

$\tau^2$-bench (voice) requires Python 3.10, but only upto 3.12 (due to support needed for `pyaudioop`). You may create and activate a new environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

3. Install ffmpeg (required for voice features)

If you plan to use voice-enabled features, you need to install ffmpeg:

**macOS:**
```bash
brew install ffmpeg
```

4. Install tau2 

```bash
pip install -e .
```

This will enable you to run the `tau2` command.

**Note:** If you use `pip install .` (without `-e`), you'll need to set the `TAU2_DATA_DIR` environment variable to point to your data directory:

```bash
export TAU2_DATA_DIR=/path/to/your/tau2-bench/data
```

**Check your data directory setup:**

After installation, you can verify that your data directory is correctly configured by running:

```bash
tau2 check-data
```

This command will check if the data directory exists and print instructions if it is missing.

To remove all the generated files and the virtual environment, run:
```bash
make clean
```

## Quick Start

### Setup LLM API keys

We use [LiteLLM](https://github.com/BerriAI/litellm) to manage LLM APIs, so you can use any LLM provider supported by LiteLLM.

To provide your API keys, copy `.env.example` as `.env` and edit it to include your API keys.

**Voice API Keys (for voice-enabled features):**

If you're using voice features, you'll need to set up the following API keys in your `.env` file:
- `ELEVENLABS_API_KEY`: For voice synthesis
- `DEEPGRAM_API_KEY`: For voice transcription

### Run agent evaluation

**Standard text-based evaluation (half-duplex):**
```bash
tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1 \
  --num-trials 1 --num-tasks 5
```

**Text full-duplex mode:**
```bash
tau2 run --domain mock --agent text_streaming_llm_agent \
  --user text_streaming_user_simulator --num-tasks 1 --streaming-enabled
```

**Voice half-duplex mode:**
```bash
tau2 run --domain mock --agent voice_llm_agent \
  --user voice_user_simulator --voice-enabled \
  --voice-transcription-model nova-3 --num-tasks 1
```

Results will be saved in `data/simulations/`.

### Simulation Directory Structure

Each simulation run creates a directory with the following structure:
```
data/simulations/<timestamp>_<domain>_<agent>_<user>/
├── results.json             # The simulation results and metrics
└── voice/                   # Only present if --voice-enabled
    └── sim_<simulation_id>/
        └── turn_<turn_uuid>/
            ├── speech.mp3                # Synthesized audio file
            ├── text_to_synthesize.txt   # Original text content
            ├── transcribed_text.txt     # Transcribed text from audio
            └── metadata.json            # Simulation and turn metadata
```

**Note:** Voice data is only generated when using `--voice-enabled` flag with voice agent/user.

> **💡 Tip**: For full agent evaluation that matches the original τ²-bench methodology, remove `--num-tasks` and use `--task-split base` to evaluate on the complete task set.

## Command Line Interface

The `tau2` command provides a unified interface for all functionality:

### Running Benchmark 
```bash
tau2 run \
  --domain <domain> \
  --agent-llm <llm_name> \
  --user-llm <llm_name> \
  --num-trials <trial_count> \
  --task-ids <task_ids> \
  --max-concurrency <concurrent_sims> \
  ...
```

### Interactive Play Mode
```bash
tau2 play
```
Experience τ²-bench from either perspective! The play mode allows you to:
- **Play as Agent**: Manually control the agent's responses and tool calls
- **Play as User**: Control the user while an LLM agent handles requests (available in domains with user tools like telecom)
- **Understand tasks** by walking through scenarios step-by-step
- **Test strategies** before implementing them in code
- **Choose task splits** to practice on training data or test on held-out tasks

This is perfect for:
- Getting familiar with domain policies and tools from both perspectives
- Debugging task scenarios and conversation flows
- Developing intuition for agent strategies
- Testing user behavior and agent responses
- Training yourself before training your model!

See the [Gym Documentation](src/tau2/gym/README.md) for more details on using the gymnasium interface programmatically, including the `AgentGymEnv` (play as agent) and `UserGymEnv` (play as user).

### Viewing Results
```bash
tau2 view
```
This tool allows you to:
- Browse simulation files (in `data/simulations/`)
- View agent performance metrics
- View a particular simulation
- View task details

### View domain documentation
```bash
tau2 domain <domain>
```
Visit http://127.0.0.1:8004/redoc to see the domain policy and API documentation.

![domain_viewer1](figs/domain_viewer.png)

### Check data configuration
```bash
tau2 check-data
```
This command checks if your data directory is properly configured and all required files are present.

## Leaderboard Submission

To submit your agent results to the τ²-bench leaderboard, you need to prepare a valid submission package that meets specific requirements.

### Requirements for Valid Submissions

Your trajectory runs must follow these constraints:

1. **Complete domain coverage**: Include results for all three domains:
   - `retail`
   - `airline` 
   - `telecom`

2. **Consistent model configuration**: All trajectory files must use:
   - The same agent LLM with identical arguments across all domains
   - The same user simulator LLM with identical arguments across all domains

3. **One result per domain**: Each domain should appear exactly once in your submission

4. **All tasks completed**: Run evaluation on all tasks within each domain (don't use `--task-ids` or `--num-tasks` filters)

> **📝 Note**: For consistency with the original τ²-bench evaluation methodology, use the `base` task split when evaluating your agent to ensure you're testing on the complete, standard task set.

### Preparing Your Submission

#### Step 1: Run Evaluations
First, run your agent evaluation on all domains with consistent settings:

```bash
# Example: Run complete evaluation for all domains
tau2 run --domain retail --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4 --save-to my_model_retail
tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4 --save-to my_model_airline  
tau2 run --domain telecom --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4 --save-to my_model_telecom
```

**Important**: Use identical `--agent-llm`, `--user-llm`, and their arguments across all runs.

#### Step 2: Prepare Submission Package
Use the submission preparation tool to create your leaderboard submission:

```bash
tau2 submit prepare data/simulations/my_model_*.json --output ./my_submission
```

This command will:
- Verify all trajectory files are valid
- Check that submission requirements are met
- Compute performance metrics (Pass^k rates)
- Prompt for required metadata (model name, organization, contact email)
- Create a structured submission directory with:
  - `submission.json`: Metadata and metrics
  - `trajectories/`: Your trajectory files

#### Step 3: Validate Your Submission
Before submitting, validate your submission package:

```bash
tau2 submit validate ./my_submission
```

This will verify:
- All required files are present
- Trajectory files are valid
- Domain coverage is complete
- Model configurations are consistent

### Additional Options

#### Skip Verification (if needed)
```bash
tau2 submit prepare data/simulations/my_model_*.json --output ./my_submission --no-verify
```

#### Verify Individual Trajectory Files
```bash
tau2 submit verify-trajs data/simulations/my_model_*.json
```

### Submitting to the Leaderboard

Once your submission package is prepared and validated:

1. Review the generated `submission.json` file
2. Follow the submission guidelines in [web/leaderboard/public/submissions/README.md](web/leaderboard/public/submissions/README.md) to create a Pull Request
3. Keep your `trajectories/` directory for reference

The leaderboard will display your model's Pass^k success rates (k=1,2,3,4) across all domains.

## Experiments

### Experimental Code Directory

The `@experiments/` directory contains experimental features and research code that extends beyond the core tau2 benchmark. This directory is designed for community contributions of innovative approaches, prototypes, and new features that are not part of the core evaluation framework.

- **Purpose**: Research code and experimental features
- **Location**: `src/experiments/`
- **Usage**: Each experimental component has its own README with documentation
- **Status**: Experimental code is provided as-is and may not be fully tested or supported

For more details, see the [experiments README](src/experiments/README.md).

### Running Ablation Studies (No User, or Agent with Oracle Plan)
`telecom` domain enables running ablation studies.

1. Running an LLM in `no-user` mode. In this mode, the LLM is given all the tools and the information upfront.
Just choose `llm_agent_solo` as the agent and `dummy_user` as the user.

```bash
tau2 run \
  --domain telecom \
  --agent llm_agent_solo \
  --agent-llm gpt-4.1 \
  --user dummy_user \
  ...
```

2. Running an LLM in `oracle-plan` mode. In this mode, the LLM is given an oracle plan ahead of time alleviating the need for action planning.
Just choose `llm_agent_gt` as the agent.

```bash
tau2 run \
  --domain telecom \
  --agent llm_agent_gt \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  ...
```

### Running Telecom Domain with Workflow Policy
To test the impact of policy format, we provide an additional "workflow" policy for the telecom domain.
To run using this policy, use the `telecom-workflow` domain.

```bash
tau2 run \
  --domain telecom-workflow \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  ...
```

## Domains

For all the details see the domains [README](src/tau2/domains/README.md).

### Basics

- Code is located in `src/tau2/domains/`
- Data is located in `data/tau2/domains/`
- Each domain has its own configuration and task definitions

#### View domain-specific policy and API docs:
Run the following command to see the domain policy and API documentation.
```bash
tau2 env <domain>
```

Then visit http://127.0.0.1:8004/redoc

### Environment CLI (beta)

An interactive command-line interface for directly querying and testing domain environments. Features:
- Interactive query interface with domain-specific tools
- Support for multiple domains (airline, mock, etc.)
- Session management with history

To use:
```bash
make env-cli
```

Available commands:
- `:q` - quit the program
- `:d` - change domain
- `:n` - start new session (clears history)

Example usage:
```bash
$ make env-cli

Welcome to the Environment CLI!
Connected to airline domain.

Query (:n new session, :d change domain, :q quit)> What flights are available from SF to LA tomorrow?
Assistant: Let me check the flight availability for you...
[Flight details will appear here]
```

The Environment CLI is useful for:
- Testing domain tools and queries
- Debugging environment responses
- Exploring available domain functionality
- Quick domain interaction without starting the full server stack


## Run tests
To run the test suite use the command

```sh
make test
```

## Config

To configure the framework, see the [config](src/tau2/config.py) file.

### LLM Calls caching
LLM call caching is disabled by default.

To enable LLM calls caching:
    - Make sure `redis` is running.
    - Update the redis config in `config.py` if necessary.
    - Set `LLM_CACHE_ENABLED` to `True` in `config.py`


## Communication Modes

τ²-bench supports two fundamentally different communication modes that change how participants interact.

### Half-Duplex vs Full-Duplex

**HALF_DUPLEX** (default): Turn-based communication
- One participant speaks at a time (like walkie-talkie)
- Complete messages only
- Traditional request-response pattern
- Backward compatible with all existing code
- **Implemented for**: Text ✅ and Voice ✅
- Best for: Standard chat agents, tool-heavy workflows, testing

**FULL_DUPLEX**: Simultaneous bidirectional communication
- Both participants can communicate at the same time (like phone call)
- Messages sent in chunks as they're generated
- Supports interruptions and overlapping speech
- Realistic conversation dynamics
- **Implemented for**: Text ✅ and Voice ✅ (via Audio Native)
- Best for: Realistic conversations, interruption handling, real-time interactions

### Implementation Status

| Modality | Half-Duplex | Full-Duplex |
|----------|-------------|-------------|
| **Text** | ✅ Available | ✅ Available |
| **Voice** | ✅ Available | ✅ Audio Native |

## Full-Duplex Mode

Full-duplex mode enables simultaneous bidirectional communication, where both participants can generate and receive content at the same time. This is achieved through **streaming** - breaking messages into chunks that are sent incrementally.

### Key Concepts

- **Full-Duplex**: Communication mode where both sides can talk simultaneously
- **Streaming**: The mechanism used to enable full-duplex (sending messages in chunks)
- **Chunks**: Small pieces of a message (e.g., words, sentences) sent incrementally
- **Interruptions**: One participant can interrupt the other mid-message

### Text Full-Duplex

Currently implemented for text-based interactions using chunk-based streaming.

### CLI Usage

```bash
# Text full-duplex mode
tau2 run --domain mock --agent text_streaming_llm_agent \
  --user text_streaming_user_simulator --num-tasks 1 --streaming-enabled
```

### Programmatic Usage

**Option 1: Use Pre-built Streaming Classes**

```python
from tau2 import (
    Orchestrator, 
    LLMStreamingAgent, 
    UserSimulatorStreaming,
    CommunicationMode
)

# Create streaming agent and user
agent = LLMStreamingAgent(
    tools=tools,
    domain_policy=policy,
    llm="gpt-4",
    chunk_by="words",      # "chars", "words", or "sentences"
    chunk_size=10          # number of units per chunk
)

user = UserSimulatorStreaming(
    instructions="You are a customer...",
    llm="gpt-4",
    chunk_by="sentences",
    chunk_size=2
)

# Run in full-duplex mode
orchestrator = Orchestrator(
    domain="airline",
    agent=agent,
    user=user,
    environment=env,
    task=task,
    mode=CommunicationMode.FULL_DUPLEX
)

# Everything else works the same!
simulation_run = orchestrator.run()
```

**Option 2: Wrap Existing Agents**

```python
from tau2 import LLMAgent, UserSimulator, make_streaming_agent, make_streaming_user

# Use your existing agents
agent = LLMAgent(tools, policy, "gpt-4")
user = UserSimulator(instructions="...", llm="gpt-4")

# Wrap them to add streaming
streaming_agent = make_streaming_agent(agent, chunk_size=30, chunk_by="words")
streaming_user = make_streaming_user(user, chunk_size=50, chunk_by="chars")

# Now they work in FULL_DUPLEX mode!
orchestrator = Orchestrator(..., mode=CommunicationMode.FULL_DUPLEX)
```

**Option 3: Keep Using Existing Code (No Changes)**

```python
from tau2 import Orchestrator, LLMAgent, UserSimulator

# Works exactly as before - HALF_DUPLEX is the default
agent = LLMAgent(tools, policy, "gpt-4")
orchestrator = Orchestrator(domain, agent, user, env, task)  # mode defaults to HALF_DUPLEX
simulation_run = orchestrator.run()  # No changes needed!
```

### Streaming Strategies

In full-duplex mode, messages are streamed as chunks. Control the chunking strategy:

- **`chunk_by="chars"`**: Split by character count (precise control)
- **`chunk_by="words"`**: Split by word boundaries (natural for text)
- **`chunk_by="sentences"`**: Split by sentences (best for readability)

```python
agent = LLMStreamingAgent(
    tools=tools,
    domain_policy=policy,
    llm="gpt-4",
    chunk_by="words",
    chunk_size=10  # Stream 10 words at a time
)
```

### Benefits of Full-Duplex

- **Realistic Simulations**: Model natural conversations with interruptions
- **Simultaneous Communication**: Both sides can communicate at once
- **Interrupt Handling**: Test how agents handle being interrupted
- **Backward Compatible**: Existing half-duplex code works unchanged
- **Flexible**: Use pre-built classes or wrap existing agents

### When to Use Full-Duplex

- ✅ Realistic conversation testing
- ✅ Interrupt handling validation
- ✅ Simulating phone-like interactions
- ✅ Research on conversation dynamics

### When to Use Half-Duplex (Default)

- ✅ Standard chat agents
- ✅ Tool-heavy workflows
- ✅ Testing and debugging
- ✅ Backward compatibility needed
- ✅ Traditional request-response patterns

## Voice Mode

Voice mode enables speech synthesis and transcription for realistic voice conversations. Currently implemented as half-duplex (turn-based).

### CLI Usage

```bash
# Voice half-duplex
tau2 run --domain mock --agent voice_llm_agent \
  --user voice_user_simulator --voice-enabled \
  --voice-transcription-model nova-3 --num-tasks 1
```

### Key Features

- **🎤 Speech Synthesis**: Uses ElevenLabs or other TTS providers
- **📝 Transcription**: Supports multiple models (Nova-3, Whisper, etc.)
- **🔊 Background Noise**: Realistic audio with configurable noise profiles
  - Continuous noise: car, café, train, village, etc.
  - Burst noise: door closing, dog barking, sirens
- **🎛️ Audio Processing**: Type-safe audio preprocessing pipeline
- **💾 Audio Storage**: All voice data saved for analysis

### Voice Data Structure

Each voice simulation creates audio files organized by turn:

```
data/simulations/<timestamp>_<domain>_voice_*/
├── results.json
└── voice/
    └── sim_<simulation_id>/
        └── turn_<turn_uuid>/
            ├── speech.mp3                # Synthesized audio
            ├── text_to_synthesize.txt   # Original text
            ├── transcribed_text.txt     # Transcribed text
            └── metadata.json            # Turn metadata
```

### Programmatic Usage

```python
from tau2 import Orchestrator, VoiceAgent, VoiceUserSimulator

# Create voice participants
agent = VoiceAgent(
    tools=tools,
    domain_policy=policy,
    llm="gpt-4",
    voice_config={
        "synthesis_provider": "elevenlabs",
        "transcription_model": "nova-3"
    }
)

user = VoiceUserSimulator(
    instructions="You are a customer...",
    llm="gpt-4",
    voice_config={
        "add_background_noise": True,
        "noise_type": "continuous"
    }
)

# Run in half-duplex mode (default for voice)
orchestrator = Orchestrator(
    domain="airline",
    agent=agent,
    user=user,
    environment=env,
    task=task
)

simulation_run = orchestrator.run()
```

### Voice Configuration Options

- `--voice-enabled`: Enable voice synthesis and transcription
- `--voice-transcription-model`: Choose transcription model (nova-3, whisper, etc.)
- `--voice-synthesis-provider`: Choose TTS provider (elevenlabs, etc.)
- `--voice-add-noise`: Add background noise for realism
- `--voice-noise-type`: continuous or burst noise

### Voice Full-Duplex (Audio Native)

Voice full-duplex is now available via **Audio Native mode**, enabling phone-like conversations where both sides can speak simultaneously using real-time multimodal models.

**Supported Providers:**
- OpenAI Realtime API (`gpt-4o-realtime`)
- Google Gemini Live
- Amazon Nova Sonic  
- xAI Grok Voice Agent
- Deepgram Voice Agent (cascaded STT→LLM→TTS)
- Qwen Omni Flash (⚠️ tool calling broken)

```bash
# Run voice full-duplex with OpenAI (default)
tau2 run --domain retail --audio-native --num-tasks 1 --verbose-logs

# Run with a specific provider
tau2 run --domain retail --audio-native --audio-native-provider gemini --num-tasks 1
```

See the [Audio Native Documentation](src/tau2/voice/audio_native/README.md) for detailed provider configuration.

## Audio Native Mode

Audio native mode uses OpenAI's Realtime API with `gpt-4o-realtime-preview` for end-to-end voice processing in a discrete-time simulation. The agent receives and produces audio directly, enabling realistic full-duplex voice conversations.

### CLI Usage

**Basic usage:**
```bash
tau2 run --domain retail --audio-native --num-tasks 1
```

**With verbose logging (saves audio files, LLM logs, and ticks):**
```bash
tau2 run --domain retail --audio-native --num-tasks 1 --verbose-logs
```

**Full configuration example:**
```bash
tau2 run --domain retail --audio-native \
  --num-tasks 5 \
  --seed 42 \
  --tick-duration 0.2 \
  --max-steps-seconds 240 \
  --speech-complexity low \
  --wait-to-respond-other 1.0 \
  --wait-to-respond-self 5.0 \
  --verbose-logs \
  --save-to my_audio_native_run
```

### Audio Native CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--audio-native` | `false` | Enable audio-native mode |
| `--tick-duration` | `0.2` | Tick duration in seconds (simulation timestep) |
| `--max-steps-seconds` | `240` | Maximum conversation duration in seconds |
| `--speech-complexity` | `low` | Speech complexity: `low`, `medium`, `high` |
| `--pcm-sample-rate` | `16000` | User simulator PCM synthesis rate |
| `--telephony-rate` | `8000` | API/agent telephony rate |

**Turn-taking thresholds:**

| Option | Default | Description |
|--------|---------|-------------|
| `--wait-to-respond-other` | `1.0` | Min seconds since agent spoke before user responds |
| `--wait-to-respond-self` | `5.0` | Min seconds since user spoke before responding again |
| `--yield-when-interrupted` | `1.0` | How long user keeps speaking when agent interrupts |
| `--yield-when-interrupting` | `5.0` | How long user keeps speaking when interrupting agent |
| `--interruption-check-interval` | `2.0` | Interval for checking interruptions |
| `--integration-duration` | `0.5` | Integration duration for linearization |
| `--silence-annotation-threshold` | `4.0` | Silence threshold for annotations |

**Agent behavior flags:**

| Option | Default | Description |
|--------|---------|-------------|
| `--no-buffer-until-complete` | `false` | Don't buffer audio until complete utterance |
| `--no-fast-forward` | `false` | Disable fast-forward mode (run in real-time) |
| `--no-send-audio-instant` | `false` | Simulate streaming audio instead of sending instantly |

### Output Structure

When using `--verbose-logs`, audio native runs create detailed output:

```
data/simulations/<timestamp>_<domain>_<agent>_<user>/
├── results.json                    # Simulation results and metrics
├── llm_logs/                       # LLM call logs (if verbose)
│   └── task_<task_id>/
│       └── *.json                  # Individual LLM call logs
├── ticks/                          # Tick data (if verbose)
│   └── task_<task_id>.json         # Tick-by-tick conversation data
└── audio/                          # Audio files (if verbose)
    └── task_<task_id>/
        ├── combined.wav            # Full conversation audio
        ├── agent.wav               # Agent-only audio track
        └── user.wav                # User-only audio track
```

### Viewing Results and Tick Data

Use `tau2 view` to analyze simulation results and full-duplex conversations:

```bash
# View simulation results
tau2 view --file data/simulations/<run>/results.json

# View with expanded tick view for full-duplex simulations
tau2 view --file data/simulations/<run>/results.json --expanded-ticks
```

### Key Concepts

- **Ticks**: Discrete time steps (default 200ms) that drive the simulation
- **Fast-forward mode**: Skip wall-clock waiting when enough audio is buffered
- **Turn-taking thresholds**: Control when participants can speak based on silence duration
- **Integration duration**: Time window for linearizing overlapping speech

## Evaluate Your Own Agent
For local or remote agent evaluation, see our [agent developer guide](src/tau2/agent/README.md).

## Contributing

We welcome contributions to τ²-bench! Whether you're fixing bugs, adding new features, creating new domains, or contributing experimental research code, please see our [Contributing Guide](CONTRIBUTING.md) for detailed guidelines on:

- **Opening issues** before starting work
- **Branch naming conventions** and development workflow  
- **Code quality standards** and testing requirements
- **Pull request guidelines** for clean, reviewable contributions
- **Domain and experimental contributions** specific guidelines

For experimental features and research code, check out the [`@experiments/`](src/experiments/) directory.

## Orchestration Sequence Diagram

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant A as Agent
    participant U as UserSimulator
    participant E as Environment

    Note over O: Initialize(task)
    rect rgb(100, 150, 150)
        O->>A: get_init_state_info(message_history)
        A->>O: agent_state_info
        O->>U: get_init_state_info(message_history)
        U->>O: user_state_info
        O->>E: set_state(initialization_data, initialization_actions, message_history)
    end
    Note over O: Start simulation
    loop Pass messages between Agent, User, and Environment

        alt Agent/Env to User
            rect rgb(200, 150, 150)
            O->>U: generate_next_message(msg, user_state_info)
            U-->>O: (user_msg, user_state_info)
            end
            Note over O: Check if user_msg is STOP
        else User/Env to Agent
            rect rgb(100, 200, 100)
            O->>A: generate_next_message(msg, agent_state_info)
            A-->>O: (assistant_msg, agent_state_info)
            Note over O: Check if too many errors
            end
        else User/Agent to Environment
            rect rgb(150, 150, 200)
            O->>E: get_response(tool_call)
            E-->>O: tool_message
            end
        end
        Note over O: Check if max turns reached.
    end
    Note over O: Return simulation run
```

## Citation

```bibtex
@misc{barres2025tau2,
      title={$\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment}, 
      author={Victor Barres and Honghua Dong and Soham Ray and Xujie Si and Karthik Narasimhan},
      year={2025},
      eprint={2506.07982},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.07982}, 
}
```
