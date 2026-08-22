"""SFT (Supervised Fine-Tuning) data formats."""

from typing import Any, Dict, List

from .base import (
    BaseDataFormat,
    DataFormatType,
    SFTSample,
    ConversationSample,
    ConversationMessage,
)


class InstructionFormat(BaseDataFormat):
    """
    Instruction format for SFT.

    Expected columns:
        - instruction: The instruction/prompt
        - input: Optional input context
        - output: The expected output/response
        - system: Optional system message
    """

    @property
    def format_type(self) -> DataFormatType:
        return DataFormatType.INSTRUCTION

    @property
    def required_columns(self) -> List[str]:
        return ["instruction", "output"]

    @property
    def optional_columns(self) -> List[str]:
        return ["input", "system"]

    def validate(self, sample: Dict[str, Any]) -> bool:
        if not self.validate_columns(sample):
            return False
        return bool(sample.get("instruction")) and bool(sample.get("output"))

    def parse(self, sample: Dict[str, Any]) -> SFTSample:
        return SFTSample(
            instruction=sample["instruction"],
            input=sample.get("input", ""),
            output=sample["output"],
            system=sample.get("system"),
        )


class AlpacaFormat(BaseDataFormat):
    """
    Alpaca format for SFT.

    Expected columns:
        - instruction: The instruction
        - input: Input context (can be empty)
        - output: The response
    """

    @property
    def format_type(self) -> DataFormatType:
        return DataFormatType.ALPACA

    @property
    def required_columns(self) -> List[str]:
        return ["instruction", "output"]

    @property
    def optional_columns(self) -> List[str]:
        return ["input"]

    def validate(self, sample: Dict[str, Any]) -> bool:
        if not self.validate_columns(sample):
            return False
        return bool(sample.get("instruction")) and bool(sample.get("output"))

    def parse(self, sample: Dict[str, Any]) -> SFTSample:
        return SFTSample(
            instruction=sample["instruction"],
            input=sample.get("input", ""),
            output=sample["output"],
        )


class ShareGPTFormat(BaseDataFormat):
    """
    ShareGPT format for multi-turn conversations.

    Expected columns:
        - conversations: List of conversation turns
            Each turn has: {"from": "human/gpt/system", "value": "..."}
    """

    @property
    def format_type(self) -> DataFormatType:
        return DataFormatType.SHAREGPT

    @property
    def required_columns(self) -> List[str]:
        return ["conversations"]

    def validate(self, sample: Dict[str, Any]) -> bool:
        if not self.validate_columns(sample):
            return False

        conversations = sample.get("conversations", [])
        if not isinstance(conversations, list) or len(conversations) < 2:
            return False

        for turn in conversations:
            if not isinstance(turn, dict):
                return False
            if "from" not in turn or "value" not in turn:
                return False

        return True

    def parse(self, sample: Dict[str, Any]) -> ConversationSample:
        conversations = sample["conversations"]
        messages = []

        role_mapping = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
            "system": "system",
        }

        for turn in conversations:
            role = role_mapping.get(turn["from"], turn["from"])
            messages.append(ConversationMessage(
                role=role,
                content=turn["value"]
            ))

        return ConversationSample(messages=messages)


class ConversationFormat(BaseDataFormat):
    """
    Generic conversation format.

    Expected columns:
        - messages: List of message objects
            Each message has: {"role": "system/user/assistant", "content": "..."}
    """

    @property
    def format_type(self) -> DataFormatType:
        return DataFormatType.CONVERSATION

    @property
    def required_columns(self) -> List[str]:
        return ["messages"]

    def validate(self, sample: Dict[str, Any]) -> bool:
        if not self.validate_columns(sample):
            return False

        messages = sample.get("messages", [])
        if not isinstance(messages, list) or len(messages) < 1:
            return False

        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg or "content" not in msg:
                return False

        return True

    def parse(self, sample: Dict[str, Any]) -> ConversationSample:
        messages = sample["messages"]
        return ConversationSample(
            messages=[
                ConversationMessage(role=msg["role"], content=msg["content"])
                for msg in messages
            ]
        )


# Format registry
SFT_FORMATS = {
    DataFormatType.INSTRUCTION: InstructionFormat,
    DataFormatType.ALPACA: AlpacaFormat,
    DataFormatType.SHAREGPT: ShareGPTFormat,
    DataFormatType.CONVERSATION: ConversationFormat,
}


def get_sft_format(format_type: DataFormatType) -> BaseDataFormat:
    """Get SFT format by type."""
    if format_type not in SFT_FORMATS:
        raise ValueError(f"Unknown SFT format: {format_type}")
    return SFT_FORMATS[format_type]()
