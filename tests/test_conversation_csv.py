from conversation_csv import (
    classify_closed_option,
    normalize_conversation,
    parse_closed_stage_results,
)


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


CLOSED_TRANSCRIPT = """\
Status: completed
Session Completed: True
Closed Script Completed: False
Number of questions: 5
Number of correct answers: 2
Current Stage: 5
--- Transcript ---

Noa: Prompt one
User: It's really unpleasant, but remember that conflict is a part of every relationship we have. How do you usually behave in situations like this?
Correct: Yes

Noa: Prompt two
User: Sounds rough, especially at the last minute... What did you end up doing?
Correct: No

Noa: Prompt three
User: Mystery answer that is not in the script
Correct: No

Noa: Closing line without scoring
User: thanks
--------------------------
"""


def test_parse_closed_stage_results_recovers_correctness():
    stages = parse_closed_stage_results(CLOSED_TRANSCRIPT)

    assert len(stages) == 3
    assert stages[0] == {
        "stage": 1,
        "noa_prompt": "Prompt one",
        "selected_answer": (
            "It's really unpleasant, but remember that conflict is a part of every "
            "relationship we have. How do you usually behave in situations like this?"
        ),
        "is_correct": True,
    }
    assert stages[1]["is_correct"] is False
    assert stages[2]["selected_answer"] == "Mystery answer that is not in the script"


def test_normalize_conversation_includes_closed_stage_results():
    conversation = normalize_conversation(
        {
            "session_id": "closed",
            "doc_id": "final_1",
            "mode": "closed",
            "status": "completed",
            "total_questions": 5,
            "correct_answers": 2,
            "session_language": "en",
            "data": CLOSED_TRANSCRIPT,
        }
    )

    assert len(conversation["closed_stage_results"]) == 3
    assert conversation["closed_stage_results"][0]["is_correct"] is True


def test_normalize_open_conversation_has_empty_closed_stage_results():
    conversation = normalize_conversation(
        {
            "session_id": "open",
            "doc_id": "final",
            "mode": "open",
            "data": "Conversation Transcript:\n-- Noa: Hi\n\n-- User: Hello",
        }
    )

    assert conversation["closed_stage_results"] == []


def test_classify_closed_option_matches_script_choices():
    script_entry = {
        "correct_answer": "Correct choice",
        "incorrect_answer_1": "Wrong A",
        "incorrect_answer_2": "Wrong B",
    }

    assert classify_closed_option("Correct choice", script_entry) == "correct"
    assert classify_closed_option("Wrong A", script_entry) == "incorrect_1"
    assert classify_closed_option("Wrong B", script_entry) == "incorrect_2"
    assert classify_closed_option("something else", script_entry) == "unknown"


def test_parse_closed_stage_results_handles_hebrew_speakers():
    data = (
        "--- Transcript ---\n"
        "נועה: שאלה\n"
        "משתמש: תשובה שגויה\n"
        "Correct: No\n"
    )
    stages = parse_closed_stage_results(data)

    assert len(stages) == 1
    assert stages[0]["noa_prompt"] == "שאלה"
    assert stages[0]["selected_answer"] == "תשובה שגויה"
    assert stages[0]["is_correct"] is False
