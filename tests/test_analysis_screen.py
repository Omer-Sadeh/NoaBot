from analysis_screen import analysis_filters, distribution_rows
from conversation_data import filter_attempts


def test_distribution_rows_keeps_metric_values_for_histogram_binning():
    rows = distribution_rows(
        [
            ({}, "Duration", 4),
            ({}, "Duration", 8),
            ({}, "Duration", None),
        ]
    )

    assert rows == [
        {"series": "Duration", "value": 4},
        {"series": "Duration", "value": 8},
    ]


def test_analysis_filters_include_all_languages_and_completed_open_outcomes():
    attempts = [
        {
            "session_id": "open-success",
            "mode": "open",
            "status": "completed",
            "is_successful": True,
            "session_language": "en",
        },
        {
            "session_id": "open-no-success",
            "mode": "open",
            "status": "completed",
            "is_successful": False,
            "session_language": "he",
        },
        {
            "session_id": "open-ongoing",
            "mode": "open",
            "status": "ongoing",
            "session_language": "en",
        },
        {
            "session_id": "closed-success",
            "mode": "closed",
            "status": "completed",
            "is_successful": True,
            "session_language": "en",
        },
    ]

    filtered = filter_attempts(attempts, analysis_filters(date_range=None))

    assert [attempt["session_id"] for attempt in filtered] == [
        "open-success",
        "open-no-success",
    ]
