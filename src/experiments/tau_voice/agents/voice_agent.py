from typing import List, Optional

from tau2.agent.base.voice import VoiceMixin, VoiceState
from tau2.agent.base_agent import HalfDuplexVoiceAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_GT,
    LLMAgent,
    LLMAgentState,
    LLMGTAgent,
)
from tau2.data_model.audio import TELEPHONY_SAMPLE_RATE
from tau2.data_model.message import AssistantMessage, Message, UserMessage
from tau2.data_model.tasks import Task
from tau2.data_model.voice import VoiceSettings
from tau2.environment.tool import Tool
from tau2.voice.synthesis.audio_effects import BackgroundNoiseGenerator
from tau2.voice.synthesis.synthesize import create_background_noise_generator
from tau2.voice_config import resolve_background_noise_path

VOICE_AGENT_INSTRUCTION = """
You are a customer service agent handling a VOICE CALL with a customer. You are receiving TRANSCRIBED TEXT from the customer's speech.

Important Voice Call Considerations:
- This is transcribed speech, not written text. Expect:
  - Missing or incorrect punctuation (periods, commas)
  - Run-on sentences or incomplete thoughts
  - Misspellings of names, emails, or technical terms
  - Natural speech patterns with fillers ("um", "uh", "you know")
  - Non-uniform pauses and chunking
- Users may spell out special characters verbally:
  - Email: "john underscore doe at gmail dot com" means "john_doe@gmail.com"
  - Name: "J O H N D O E" means "John Doe"
- Ask for clarification if something is unclear
- If you mishear or are unsure about critical information (names, emails, IDs), ask the user to repeat it or spell it out letter by letter
- Respond naturally and conversationally as you would in a phone call
- Keep responses concise and clear for voice communication

In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()


VOICE_AGENT_GT_INSTRUCTION = """
You are testing that our user simulator is working correctly in a VOICE CALL scenario.
User simulator will have an issue for you to solve through spoken conversation.
You are receiving TRANSCRIBED TEXT from the user's speech.

Important Voice Call Considerations:
- This is transcribed speech, not written text. Expect:
  - Missing or incorrect punctuation (periods, commas)
  - Run-on sentences or incomplete thoughts
  - Misspellings of names, emails, or technical terms
  - Natural speech patterns with fillers ("um", "uh", "you know")
  - Non-uniform pauses and chunking
- Users will spell out special characters verbally:
  - Email: "john underscore doe at gmail dot com" means "john_doe@gmail.com"
  - Name: "J O H N D O E" means "John Doe"
- Ask for clarification if something is unclear
- If you mishear or are unsure about critical information (names, emails, IDs), ask the user to repeat it or spell it out letter by letter
- Respond naturally and conversationally as you would in a phone call
- Keep responses concise and clear for voice communication

You must behave according to the <policy> provided below.
To make following the policy easier, we give you the list of resolution steps you are expected to take.
These steps involve either taking an action or asking the user to take an action.

In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()


class VoiceLLMAgentState(LLMAgentState, VoiceState):
    """State for the VoiceLLMAgent."""


class VoiceLLMAgent(
    VoiceMixin[UserMessage, AssistantMessage, VoiceLLMAgentState],
    LLMAgent[VoiceLLMAgentState],
    HalfDuplexVoiceAgent[VoiceLLMAgentState],
):
    """LLM Agent with voice transcription capabilities."""

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        voice_settings: VoiceSettings,
        llm_args: Optional[dict] = None,
    ):
        """Initialize the VoiceLLMAgent."""
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            voice_settings=voice_settings,
            llm=llm,
            llm_args=llm_args,
        )
        self.validate_voice_settings()

    def validate_voice_settings(self) -> None:
        """Validate the voice settings."""
        if self.voice_settings is None:
            raise ValueError("Voice settings must be provided")
        if not self.voice_settings.transcription_enabled:
            raise ValueError("Voice transcription must be enabled")

    @property
    def system_prompt(self) -> str:
        """Override system prompt to use voice-specific instructions."""
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=VOICE_AGENT_INSTRUCTION
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> VoiceLLMAgentState:
        """
        Get the initial state of the voice agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the voice agent (LLMAgentVoiceState).
        """
        # Get the base state from parent
        base_state = super().get_init_state(message_history)

        # Create background noise generator if synthesis config is available
        synthesis_config = self.voice_settings.synthesis_config
        speech_env = self.voice_settings.speech_environment
        background_noise_file = resolve_background_noise_path(
            speech_env.background_noise_file
        )
        if synthesis_config is not None:
            noise_generator = create_background_noise_generator(
                config=synthesis_config.source_effects_config,
                sample_rate=TELEPHONY_SAMPLE_RATE,
                background_noise_file=background_noise_file,
            )
        else:
            noise_generator = BackgroundNoiseGenerator(
                sample_rate=TELEPHONY_SAMPLE_RATE,
                silent_mode=True,
            )

        # Create voice agent state with the base state's data
        return VoiceLLMAgentState(
            system_messages=base_state.system_messages,
            messages=base_state.messages,
            noise_generator=noise_generator,
        )

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: VoiceLLMAgentState
    ) -> AssistantMessage:
        """Respond to a user or tool message with audio transcription support."""
        # Handle audio transcription if present
        if isinstance(message, UserMessage):
            if not message.is_audio:
                raise ValueError("User message must be audio")
            message = self.transcribe_voice(message)
            message.is_audio = False
        assistant_message = super()._generate_next_message(message, state)
        return assistant_message


class VoiceLLMGTAgent(
    VoiceMixin[UserMessage, AssistantMessage, VoiceLLMAgentState],
    LLMGTAgent[VoiceLLMAgentState],
    HalfDuplexVoiceAgent[VoiceLLMAgentState],
):
    """Ground Truth Agent with voice transcription capabilities."""

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        llm: str,
        voice_settings: VoiceSettings,
        llm_args: Optional[dict] = None,
        provide_function_args: bool = True,
    ):
        """Initialize the VoiceLLMGTAgent."""
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            voice_settings=voice_settings,
            task=task,
            llm=llm,
            llm_args=llm_args,
            provide_function_args=provide_function_args,
        )
        self.validate_voice_settings()

    def validate_voice_settings(self) -> None:
        """Validate the voice settings."""
        if self.voice_settings is None:
            raise ValueError("Voice settings must be provided")
        if not self.voice_settings.transcription_enabled:
            raise ValueError("Voice transcription must be enabled")

    @property
    def system_prompt(self) -> str:
        """Override system prompt to use voice-specific instructions."""
        return SYSTEM_PROMPT_GT.format(
            agent_instruction=VOICE_AGENT_GT_INSTRUCTION,
            domain_policy=self.domain_policy,
            resolution_steps=self.make_agent_instructions_from_actions(),
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> VoiceLLMAgentState:
        """
        Get the initial state of the voice agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the voice agent (LLMAgentVoiceState).
        """
        # Get the base state from parent
        base_state = super().get_init_state(message_history)

        # Create background noise generator if synthesis config is available
        synthesis_config = self.voice_settings.synthesis_config
        speech_env = self.voice_settings.speech_environment
        background_noise_file = resolve_background_noise_path(
            speech_env.background_noise_file
        )
        if synthesis_config is not None:
            noise_generator = create_background_noise_generator(
                config=synthesis_config.source_effects_config,
                sample_rate=TELEPHONY_SAMPLE_RATE,
                background_noise_file=background_noise_file,
            )
        else:
            noise_generator = BackgroundNoiseGenerator(
                sample_rate=TELEPHONY_SAMPLE_RATE,
                silent_mode=True,
            )

        # Create voice agent state with the base state's data
        return VoiceLLMAgentState(
            system_messages=base_state.system_messages,
            messages=base_state.messages,
            noise_generator=noise_generator,
        )

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: VoiceLLMAgentState
    ) -> AssistantMessage:
        """Respond to a user or tool message with audio transcription support."""
        # Handle audio transcription if present
        if isinstance(message, UserMessage):
            if not message.is_audio:
                raise ValueError("User message must be audio")
            message = self.transcribe_voice(message)
            message.is_audio = False
        assistant_message = super()._generate_next_message(message, state)
        return assistant_message


# Note: VoiceLLMSoloAgent is not needed since LLMSoloAgent doesn't support user messages
