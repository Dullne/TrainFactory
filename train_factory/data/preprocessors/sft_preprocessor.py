"""SFT data preprocessor."""

import logging
from typing import Any, Dict, List, Optional

from datasets import Dataset

from .base import TokenizingPreprocessor

logger = logging.getLogger(__name__)


class SFTPreprocessor(TokenizingPreprocessor):
    """
    Preprocessor for SFT (Supervised Fine-Tuning) data.

    Handles conversion of instruction/conversation data into
    the format expected by transformers Trainer.
    """

    # Default chat template for models without one
    DEFAULT_TEMPLATE = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] }}\n\n"
        "{% set messages = messages[1:] %}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "User: {{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}"
        "Assistant: {{ message['content'] }}"
        "{% if not loop.last %}\n{% endif %}"
        "{% endif %}"
        "{% endfor %}"
    )

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 2048,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SFT preprocessor.

        Args:
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            config: Additional configuration including:
                - system_prompt: Default system prompt
                - instruction_template: Template for instruction format
                - response_template: Template for response
        """
        super().__init__(tokenizer, max_length, config)

        self.system_prompt = self.config.get("system_prompt", "")
        self.instruction_template = self.config.get(
            "instruction_template",
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        )
        self.response_template = self.config.get("response_template", "{output}")

    def preprocess(self, dataset: Dataset) -> Dataset:
        """
        Preprocess SFT dataset.

        Converts instruction/conversation data to input_ids and labels.
        """
        # Filter invalid samples first
        dataset = self.filter_invalid(dataset)

        # Apply preprocessing
        dataset = dataset.map(
            self._preprocess_function,
            batched=False,
            remove_columns=dataset.column_names,
            desc="Preprocessing SFT data",
        )

        return dataset

    def _preprocess_function(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a single sample."""
        # Check if it's conversation format
        if "messages" in sample:
            return self._preprocess_conversation(sample)
        elif "conversations" in sample:
            return self._preprocess_sharegpt(sample)
        else:
            return self._preprocess_instruction(sample)

    def _preprocess_instruction(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess instruction format."""
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output = sample.get("output", "")

        # Build prompt
        prompt = self.instruction_template.format(
            instruction=instruction,
            input=input_text
        )
        response = self.response_template.format(output=output)

        # Tokenize
        full_text = prompt + response

        # Tokenize full text
        tokenized = self.tokenize(full_text, truncation=True)

        # Calculate where the response starts for label masking
        prompt_tokens = self.tokenize(prompt, truncation=False, add_special_tokens=False)
        prompt_length = len(prompt_tokens["input_ids"])

        # Create labels (mask prompt tokens with -100)
        labels = tokenized["input_ids"].copy()
        labels[:prompt_length] = [-100] * prompt_length

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    def _preprocess_conversation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess conversation format (messages)."""
        messages = sample["messages"]

        # Add system prompt if not present and configured
        if self.system_prompt and messages[0].get("role") != "system":
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Fallback to default template
            text = self._apply_default_template(messages)

        tokenized = self.tokenize(text, truncation=True)

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].copy(),
        }

    def _preprocess_sharegpt(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess ShareGPT format."""
        conversations = sample["conversations"]

        # Convert to standard messages format
        role_mapping = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
            "system": "system",
        }

        messages = []
        for turn in conversations:
            role = role_mapping.get(turn["from"], turn["from"])
            messages.append({"role": role, "content": turn["value"]})

        return self._preprocess_conversation({"messages": messages})

    def _apply_default_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply default chat template."""
        result = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                result.append(f"{content}\n")
            elif role == "user":
                result.append(f"User: {content}\n")
            elif role == "assistant":
                result.append(f"Assistant: {content}\n")
        return "".join(result)

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate a single sample."""
        # Check for instruction format
        if "instruction" in sample:
            return bool(sample.get("instruction")) and bool(sample.get("output"))

        # Check for conversation format
        if "messages" in sample:
            messages = sample.get("messages", [])
            return isinstance(messages, list) and len(messages) >= 2

        # Check for ShareGPT format
        if "conversations" in sample:
            conversations = sample.get("conversations", [])
            return isinstance(conversations, list) and len(conversations) >= 2

        return False

    @property
    def preprocessor_type(self) -> str:
        return "sft"
