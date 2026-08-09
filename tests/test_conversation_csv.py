from conversation_csv import normalize_conversation


def test_normalize_conversation_differences_legacy_cumulative_step_offsets():
    conversation = normalize_conversation(
        {
            "session_id": "session",
            "doc_id": "final",
            "mode": "open",
            "data": (
                "Time on Each Step (min:sec): ['0:10', '0:25', '1:00']\n"
                "Conversation Transcript:\n-- Noa: Hello\n\n-- User: Hi"
            ),
        }
    )

    assert conversation["section_durations_seconds"] == [10, 15, 35]
    assert conversation["section_time_semantics"] == "cumulative_legacy"


def test_normalize_conversation_prefers_structured_turns_and_duration():
    conversation = normalize_conversation(
        {
            "session_id": "session",
            "doc_id": "current",
            "mode": "open",
            "duration_seconds": 14.2,
            "section_durations_seconds": [4.2],
            "section_time_semantics": "per_section_v2",
            "turns": [
                {
                    "role": "assistant",
                    "content": "Hello",
                    "elapsed_seconds": None,
                },
                {
                    "role": "user",
                    "content": "Hi",
                    "elapsed_seconds": 4.2,
                    "input_modality": "text",
                },
            ],
            "data": "",
        }
    )

    assert conversation["session_duration_seconds"] == 14.2
    assert conversation["conversation_content"] == "Noa: Hello\n\nUser: Hi"
    assert conversation["turns"][1]["input_modality"] == "text"
    assert conversation["section_durations_seconds"] == [4.2]
    assert conversation["section_time_semantics"] == "per_section_v2"


def test_normalize_conversation_parses_hebrew_legacy_speakers():
    conversation = normalize_conversation(
        {
            "session_id": "hebrew",
            "doc_id": "final",
            "mode": "open",
            "data": "תמלול השיחה:\n-- נועה: קשה לי\n\n-- משתמש: אני שומעת אותך",
        }
    )

    assert conversation["conversation_content"] == "Noa: קשה לי\n\nUser: אני שומעת אותך"
    assert conversation["user_messages"] == '["אני שומעת אותך"]'
