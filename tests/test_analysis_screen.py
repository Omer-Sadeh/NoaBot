from analysis_screen import distribution_rows


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
