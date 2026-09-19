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


def sample_attempts():
    return [
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
        {
            "session_id": "closed-no-success",
            "mode": "closed",
            "status": "completed",
            "is_successful": False,
            "session_language": "he",
        },
    ]


def test_analysis_filters_include_all_languages_and_completed_open_outcomes():
    filtered = filter_attempts(sample_attempts(), analysis_filters(date_range=None))

    assert [attempt["session_id"] for attempt in filtered] == [
        "open-success",
        "open-no-success",
    ]


def test_closed_filter_keeps_completed_closed_attempts_for_survey_insights():
    open_filters = analysis_filters(date_range=None)
    closed_filters = {**open_filters, "mode": "closed"}
    filtered = filter_attempts(sample_attempts(), closed_filters)

    assert [attempt["session_id"] for attempt in filtered] == [
        "closed-success",
        "closed-no-success",
    ]
