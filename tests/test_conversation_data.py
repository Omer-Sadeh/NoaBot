from datetime import datetime

from conversation_data import filter_attempts


def test_filter_attempts_keeps_open_success_with_matching_language_and_date():
    attempts = [
        {
            "session_id": "match",
            "mode": "open",
            "status": "completed",
            "is_successful": True,
            "session_language": "en",
            "timestamp": datetime(2026, 8, 9),
        },
        {
            "session_id": "closed",
            "mode": "closed",
            "status": "completed",
            "is_successful": True,
            "session_language": "en",
            "timestamp": datetime(2026, 8, 9),
        },
    ]
    filters = {
        "mode": "open",
        "statuses": ["success"],
        "date_range": (datetime(2026, 8, 9).date(), datetime(2026, 8, 9).date()),
        "session_id": "",
        "language": "en",
    }

    filtered = filter_attempts(attempts, filters)

    assert [attempt["session_id"] for attempt in filtered] == ["match"]
