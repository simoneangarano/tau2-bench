"""
Text-based chunking mixin for streaming participants (experimental).

Extracted from tau2.agent.base.streaming when text streaming agents
were moved to experiments/tau_voice/. The production audio chunking mixin
(AudioChunkingMixin) remains in tau2.
"""

import re
import uuid

from tau2.agent.base.streaming import (
    InputMessageType,
    OutputMessageType,
    StateType,
    StreamingMixin,
)


class TextChunkingMixin(StreamingMixin[InputMessageType, OutputMessageType, StateType]):
    """
    Generic streaming mixin with text-based chunking.

    Chunks text messages by character count, word boundaries, or sentences.
    Works with any message type that has a 'content' attribute and supports
    chunk_id and is_final_chunk parameters.

    Type Parameters:
        InputMessageType: The type of messages this participant receives
        OutputMessageType: The type of messages this participant produces (must support chunking)
        StateType: The type of internal streaming state
    """

    def __init__(self, *args, chunk_by: str = "chars", chunk_size: int = 50, **kwargs):
        """
        Initialize text chunking mixin.

        Args:
            chunk_by: Chunking strategy - "chars", "words", or "sentences"
            chunk_size: Size of each chunk (characters, words, or sentences)
        """
        super().__init__(*args, chunk_size=chunk_size, **kwargs)
        self.chunk_by = chunk_by

    def _create_chunk_messages(
        self,
        full_message: OutputMessageType,
    ) -> list[OutputMessageType]:
        """Generate chunks from text content.

        Assumes the message has 'content', 'role', 'cost', 'usage', 'raw_data' attributes
        and supports 'chunk_id' and 'is_final_chunk' parameters in its constructor.
        """
        content = getattr(full_message, "content", None) or ""

        if self.chunk_by == "chars":
            chunks = self._chunk_by_chars(content)
        elif self.chunk_by == "words":
            chunks = self._chunk_by_words(content)
        elif self.chunk_by == "sentences":
            chunks = self._chunk_by_sentences(content)
        else:
            chunks = [content]  # Fallback: single chunk

        chunk_messages = []
        message_class = type(full_message)

        # Generate a unique utterance ID for the message
        utterance_id = str(uuid.uuid4())

        for i, chunk_content in enumerate(chunks):
            is_final = i == len(chunks) - 1
            # Create chunk message using the same class as the full message
            chunk_message = message_class(
                role=full_message.role,
                content=chunk_content,
                cost=full_message.cost if i == 0 else 0.0,
                usage=full_message.usage if i == 0 else None,
                raw_data=full_message.raw_data if i == 0 else None,
                utterance_ids=[utterance_id],
                chunk_id=i,
                is_final_chunk=is_final,
            )
            chunk_messages.append(chunk_message)
        return chunk_messages

    def _chunk_by_chars(self, text: str) -> list[str]:
        """Chunk text by character count."""
        return [
            text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)
        ]

    def _chunk_by_sep(self, text: str, separator_pattern: str) -> list[str]:
        """Chunk text by separator pattern, preserving original separators.

        Chunks are created by cutting at every `chunk_size` separators.
        """
        # Find all separator positions
        sep_matches = list(re.finditer(separator_pattern, text))

        if not sep_matches:
            # No separators found, return whole text as single chunk
            return [text] if text else [""]

        # Build chunks by cutting at every chunk_size separators
        chunks = []
        start = 0
        for i in range(self.chunk_size - 1, len(sep_matches), self.chunk_size):
            end = sep_matches[i].end()
            chunks.append(text[start:end])
            start = end

        # Add remaining text after last cut
        if start < len(text):
            chunks.append(text[start:])

        return chunks if chunks else [""]

    def _chunk_by_words(self, text: str) -> list[str]:
        """Chunk text by word count, preserving original whitespace."""
        return self._chunk_by_sep(text, r"\s+")

    def _chunk_by_sentences(self, text: str) -> list[str]:
        """Chunk text by sentence count, preserving original whitespace."""
        return self._chunk_by_sep(text, r"[.!?]+\s+")
