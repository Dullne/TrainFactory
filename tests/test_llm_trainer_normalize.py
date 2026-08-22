"""Tests for LLMTrainer SFT dataset normalization robustness (audit finding:
_normalize_sft_dataset crashed on dirty samples — non-dict ShareGPT turns,
None query/answer — taking down the whole dataset.map)."""
from datasets import Dataset

from train_factory.trainers.decoder.llm_trainer import LLMTrainer


def _trainer():
    return LLMTrainer(
        {"training_method": "sft", "dataset_configs": [], "max_length": 1024}
    )


def test_sft_qa_filters_rows_with_none_query_or_answer():
    trainer = _trainer()
    ds = Dataset.from_dict(
        {"query": ["q1", None, "q3", "q4"], "answer": ["a1", "a2", None, ""]}
    )
    out = trainer._normalize_sft_dataset(ds)
    # Only q1/a1 is fully valid
    assert len(out["messages"]) == 1
    roles = [m["role"] for m in out["messages"][0]]
    assert roles == ["user", "assistant"]
    assert out["messages"][0][0]["content"] == "q1"


def test_sft_sharegpt_filters_null_message_content():
    trainer = _trainer()
    ds = Dataset.from_dict(
        {
            "conversations": [
                [{"from": "human", "value": None}, {"from": "gpt", "value": "hello"}],
            ],
        }
    )
    out = trainer._normalize_sft_dataset(ds)
    assert len(out) == 0


def test_sft_instruction_filters_rows_with_none_instruction_or_output():
    trainer = _trainer()
    ds = Dataset.from_dict(
        {"instruction": ["do x", None], "output": ["result", "no-instruction"]}
    )
    out = trainer._normalize_sft_dataset(ds)
    assert len(out["messages"]) == 1
    assert out["messages"][0][-1]["content"] == "result"


def test_sft_messages_filters_rows_without_user_and_assistant():
    trainer = _trainer()
    ds = Dataset.from_dict(
        {
            "messages": [
                [{"role": "system", "content": "system only"}],
                [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
            ]
        }
    )

    out = trainer._normalize_sft_dataset(ds)

    assert len(out) == 1
    assert [turn["role"] for turn in out[0]["messages"]] == ["user", "assistant"]


def test_sft_sharegpt_filters_conversation_with_only_invalid_turns():
    trainer = _trainer()
    ds = Dataset.from_dict({"conversations": [[None]]})

    out = trainer._normalize_sft_dataset(ds)

    assert len(out) == 0


def test_sft_messages_filters_blank_message_content():
    trainer = _trainer()
    ds = Dataset.from_dict(
        {
            "messages": [[
                {"role": "user", "content": "   "},
                {"role": "assistant", "content": "answer"},
            ]]
        }
    )

    out = trainer._normalize_sft_dataset(ds)

    assert len(out) == 0


def test_sft_messages_preserves_tool_call_metadata_and_null_content():
    trainer = _trainer()
    tool_calls = [
        {
            "id": "call_weather",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": {"city": "Shanghai"},
            },
        }
    ]
    ds = Dataset.from_dict(
        {
            "messages": [[
                {"role": "user", "content": "What is the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                {
                    "role": "tool",
                    "content": "Sunny",
                    "tool_call_id": "call_weather",
                    "name": "get_weather",
                },
                {"role": "assistant", "content": "It is sunny."},
            ]]
        }
    )

    out = trainer._normalize_sft_dataset(ds)

    assert len(out) == 1
    messages = out[0]["messages"]
    assert messages[1]["content"] is None
    assert messages[1]["tool_calls"] == tool_calls
    assert messages[2]["tool_call_id"] == "call_weather"
    assert messages[2]["name"] == "get_weather"


def test_sft_messages_preserves_structured_content():
    trainer = _trainer()
    user_content = [{"type": "text", "text": "Describe this input"}]
    assistant_content = [{"type": "text", "text": "A structured answer"}]
    ds = Dataset.from_dict(
        {
            "messages": [[
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]]
        }
    )

    out = trainer._normalize_sft_dataset(ds)

    assert len(out) == 1
    assert out[0]["messages"][0]["content"] == user_content
    assert out[0]["messages"][1]["content"] == assistant_content


def test_sft_messages_rejects_entire_broken_tool_conversation():
    trainer = _trainer()
    ds = Dataset.from_dict(
        {
            "messages": [[
                {"role": "user", "content": "Run the tool"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call_1", "name": "lookup"}],
                },
                {
                    "role": "tool",
                    "content": "   ",
                    "tool_call_id": "call_1",
                },
                {"role": "assistant", "content": "Done"},
            ]]
        }
    )

    out = trainer._normalize_sft_dataset(ds)

    assert len(out) == 0


def test_sft_messages_preserves_legacy_function_call_conversation():
    trainer = _trainer()
    function_call = {
        "name": "get_weather",
        "arguments": '{"city":"Shanghai"}',
    }
    ds = Dataset.from_dict(
        {
            "messages": [[
                {"role": "user", "content": "What is the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": function_call,
                },
                {
                    "role": "function",
                    "name": "get_weather",
                    "content": '{"condition":"sunny"}',
                },
                {"role": "assistant", "content": "It is sunny."},
            ]]
        }
    )

    out = trainer._normalize_sft_dataset(ds)

    assert len(out) == 1
    messages = out[0]["messages"]
    assert messages[1]["content"] is None
    assert messages[1]["function_call"] == function_call
    assert messages[2]["role"] == "function"
    assert messages[2]["name"] == "get_weather"
    assert messages[2]["content"] == '{"condition":"sunny"}'


def test_sft_messages_rejects_legacy_function_result_without_name():
    trainer = _trainer()
    ds = Dataset.from_dict(
        {
            "messages": [[
                {"role": "user", "content": "Run the function"},
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": "lookup", "arguments": "{}"},
                },
                {"role": "function", "content": '{"result":1}'},
                {"role": "assistant", "content": "Done"},
            ]]
        }
    )

    out = trainer._normalize_sft_dataset(ds)

    assert len(out) == 0
