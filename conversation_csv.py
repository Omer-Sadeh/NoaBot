"""Pure helpers for flattening saved conversation records into CSV rows."""

import ast
import json
import re


CSV_FIELDNAMES = (
    "session_id",
    "doc_id",
    "timestamp",
    "session_created",
    "mode",
    "status",
    "session_completed",
    "closed_script_completed",
    "is_successful",
    "session_duration",
    "session_duration_seconds",
    "user_message_count",
    "completed_guidelines",
    "completed_criteria_total",
    "time_on_each_step",
    "tips_shown",
    "current_stage",
    "total_questions",
    "correct_answers",
    "session_language",
    "conversation_content",
    "noa_messages",
    "user_messages",
    "data",
)

METADATA_LABELS = {
    "status": "status",
    "סטטוס": "status",
    "completed": "open_success",
    "session completed": "session_completed",
    "closed script completed": "closed_script_completed",
    "session duration": "session_duration",
    "conversation duration": "session_duration",
    "משך השיחה": "session_duration",
    "number of user messages": "user_message_count",
    "מספר הודעות משתמש": "user_message_count",
    "number of completed criteria": "completed_guidelines",
    "number of criteria completed": "completed_guidelines",
    "מספר קריטריונים שהושלמו": "completed_guidelines",
    "time on each step (min:sec)": "time_on_each_step",
    "time on each step (min)": "time_on_each_step",
    "זמן בכל שלב (דקות)": "time_on_each_step",
    "number of tips shown": "tips_shown",
    "מספר טיפים שהוצגו": "tips_shown",
    "current stage": "current_stage",
    "number of questions": "total_questions",
    "number of correct answers": "correct_answers",
}
TRANSCRIPT_MARKERS = (
    "Conversation Transcript:",
    "Conversation Transcript",
    "--- Transcript ---",
    "תמלול השיחה:",
)
SPEAKER_PATTERN = re.compile(
    r"^\s*(?:--\s*)?(Noa|User|Therapist|נועה|משתמש)\s*:\s?(.*)$",
    re.IGNORECASE | re.MULTILINE,
)


def normalize_conversation(conversation):
    """Return one stable, analysis-friendly CSV row for a conversation record."""
    data = conversation.get("data") or ""
    metadata = parse_metadata(data)
    mode = conversation.get("mode") or "open"
    status = first_value(conversation.get("status"), metadata.get("status"))
    if status is None:
        status = "completed" if data else ""

    parsed_completed, parsed_total = split_completed_guidelines(
        metadata.get("completed_guidelines")
    )
    session_completed = first_value(
        conversation.get("session_finished"),
        conversation.get("completed"),
        boolean_value(metadata.get("session_completed")),
        status == "completed",
    )
    is_successful = first_value(
        conversation.get("is_successful"),
        success_from_metadata(mode, metadata),
    )
    closed_script_completed = (
        first_value(
            conversation.get("is_successful"),
            boolean_value(metadata.get("closed_script_completed")),
        )
        if mode == "closed"
        else None
    )
    transcript, turns = parse_transcript(data)
    noa_messages = [content for role, content in turns if role == "noa"]
    user_messages = [content for role, content in turns if role == "user"]

    return {
        "session_id": conversation.get("session_id"),
        "doc_id": conversation.get("doc_id"),
        "timestamp": conversation.get("timestamp"),
        "session_created": conversation.get("session_created"),
        "mode": mode,
        "status": status,
        "session_completed": session_completed,
        "closed_script_completed": closed_script_completed,
        "is_successful": is_successful,
        "session_duration": metadata.get("session_duration"),
        "session_duration_seconds": duration_seconds(metadata.get("session_duration")),
        "user_message_count": first_value(
            conversation.get("user_message_count"),
            integer_value(metadata.get("user_message_count")),
        ),
        "completed_guidelines": first_value(
            conversation.get("completed_guidelines"),
            parsed_completed,
        ),
        "completed_criteria_total": parsed_total,
        "time_on_each_step": normalized_step_times(metadata.get("time_on_each_step")),
        "tips_shown": integer_value(metadata.get("tips_shown")),
        "current_stage": first_value(
            conversation.get("current_stage"),
            integer_value(metadata.get("current_stage")),
        ),
        "total_questions": first_value(
            conversation.get("total_questions"),
            integer_value(metadata.get("total_questions")),
        ),
        "correct_answers": first_value(
            conversation.get("correct_answers"),
            integer_value(metadata.get("correct_answers")),
        ),
        "session_language": conversation.get("session_language"),
        "conversation_content": transcript,
        "noa_messages": json.dumps(noa_messages, ensure_ascii=False),
        "user_messages": json.dumps(user_messages, ensure_ascii=False),
        "data": data,
    }


def parse_metadata(data):
    """Extract known metadata lines without interpreting transcript messages."""
    metadata = {}
    for line in data.splitlines():
        normalized_line = line.strip().lower()
        for label, field in METADATA_LABELS.items():
            prefix = f"{label}:"
            if normalized_line.startswith(prefix):
                metadata[field] = line.strip()[len(prefix):].strip()
                break
    return metadata


def parse_transcript(data):
    """Return cleaned transcript text and ordered normalized speaker turns."""
    transcript = transcript_section(data)
    turns = []
    current_role = None
    current_lines = []

    def add_turn():
        if current_role is not None:
            content = "\n".join(current_lines).strip()
            if content:
                turns.append((current_role, content))

    for line in transcript.splitlines():
        match = SPEAKER_PATTERN.match(line)
        if match:
            add_turn()
            current_role = normalize_speaker(match.group(1))
            current_lines = [match.group(2)]
        elif line.strip().lower().startswith("correct:"):
            add_turn()
            current_role = None
            current_lines = []
        elif line.strip().startswith("---"):
            continue
        elif current_role is not None:
            current_lines.append(line)
    add_turn()

    if turns:
        transcript = "\n\n".join(
            f"{'Noa' if role == 'noa' else 'User'}: {content}"
            for role, content in turns
        )
    return transcript.strip(), turns


def transcript_section(data):
    for marker in TRANSCRIPT_MARKERS:
        marker_index = data.lower().find(marker.lower())
        if marker_index >= 0:
            return data[marker_index + len(marker):].lstrip("\n -")
    return data if SPEAKER_PATTERN.search(data) else ""


def normalize_speaker(speaker):
    return "noa" if speaker.lower() in {"noa", "נועה"} else "user"


def success_from_metadata(mode, metadata):
    if mode == "open":
        return boolean_value(metadata.get("open_success"))
    if mode == "closed":
        completed = boolean_value(metadata.get("closed_script_completed"))
        if completed is not None:
            return completed
        total = integer_value(metadata.get("total_questions"))
        correct = integer_value(metadata.get("correct_answers"))
        if total is not None and correct is not None:
            return correct == total
    return None


def split_completed_guidelines(value):
    if not value:
        return None, None
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", str(value))
    if not match:
        return integer_value(value), None
    return int(match.group(1)), int(match.group(2))


def duration_seconds(value):
    if not value:
        return None
    text = str(value)
    minutes = re.search(r"(\d+)\s*(?:minutes?|דקות)", text, re.IGNORECASE)
    seconds = re.search(r"(\d+)\s*(?:seconds?|שניות)", text, re.IGNORECASE)
    if not minutes and not seconds:
        return None
    return (int(minutes.group(1)) if minutes else 0) * 60 + (
        int(seconds.group(1)) if seconds else 0
    )


def normalized_step_times(value):
    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return str(value)
    return json.dumps(parsed, ensure_ascii=False)


def boolean_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def integer_value(value):
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def first_value(*values):
    return next((value for value in values if value is not None), None)
