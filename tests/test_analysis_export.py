import csv
import io
import zipfile
from datetime import datetime, timezone

from analysis_export import DATASET_DESCRIPTIONS, build_analysis_export
from conversation_analysis import ANALYZER_VERSION, deterministic_metrics
from script_analysis_explanations import MEASURE_DEFINITIONS
from survey_data import PII_COLUMNS


def open_attempt():
    return {
        "session_id": "session-open",
        "doc_id": "attempt-open",
        "mode": "open",
        "status": "completed",
        "session_language": "he",
        "session_duration_seconds": 42,
        "schema_version": 2,
        "section_durations_seconds": [20, 22],
        "section_user_turns": [1, 1],
        "section_completion_flags": [True, False],
        "section_time_semantics": "per_section_v2",
        "section_transition_events": [
            {"section": 1, "completed": True, "elapsed_seconds": 20, "user_turns": 1}
        ],
        "completed_guidelines": 1,
        "completed_criteria_total": 5,
        "is_successful": False,
        "turns": [
            {"role": "noa", "content": "אני כועסת", "elapsed_seconds": None},
            {
                "role": "user",
                "content": "=HYPERLINK(\"https://example.invalid\")",
                "elapsed_seconds": 10,
                "input_modality": "text",
            },
            {"role": "noa", "content": "לא אמרתי דבר", "elapsed_seconds": 20},
            {"role": "user", "content": "מה היית רוצה לומר?", "elapsed_seconds": 30},
        ],
    }


def closed_attempt():
    return {
        "session_id": "session-closed",
        "doc_id": "attempt-closed",
        "mode": "closed",
        "status": "completed",
        "session_language": "en",
        "total_questions": 1,
        "correct_answers": 0,
        "closed_stage_results": [
            {"stage": 1, "selected_answer": "+unsafe", "is_correct": False}
        ],
    }


def survey_row():
    row = {
        "session_id": "session-open",
        "S1_pre_calm_tense_conversation": 50,
        "S2_post_calm_tense_conversation": 60,
        "S1_pre_reflect_viewpoint": 50,
        "S2_post_reflect_viewpoint": 60,
        "S1_pre_constructive_next_step": 50,
        "S2_post_constructive_next_step": 60,
        "S1_pre_guiding_questions": 50,
        "S2_post_guiding_questions": 60,
        "S2_reuse_intention": 4,
        "username": "must-not-export",
    }
    for name in (
        "S2_UES_lost_myself",
        "S2_UES_time_slipped_away",
        "S2_UES_absorbed",
        "S2_UES_frustrated_raw",
        "S2_UES_confusing_raw",
        "S2_UES_taxing_raw",
        "S2_UES_worthwhile",
        "S2_UES_rewarding",
        "S2_UES_interested",
        "S2_agent_fake_natural",
        "S2_agent_machinelike_humanlike",
        "S2_agent_unconscious_conscious",
        "S2_agent_artificial_lifelike",
        "S2_agent_rigid_elegant",
        "S2_agent_dislike_like",
        "S2_agent_unfriendly_friendly",
        "S2_agent_unkind_kind",
        "S2_agent_unpleasant_pleasant",
        "S2_agent_awful_nice",
    ):
        row[name] = 3
    for name in (
        "S2_UEQ_obstructive_supportive",
        "S2_UEQ_complicated_easy",
        "S2_UEQ_inefficient_efficient",
        "S2_UEQ_confusing_clear",
        "S2_UEQ_boring_exciting",
        "S2_UEQ_not_interesting_interesting",
        "S2_UEQ_conventional_inventive",
        "S2_UEQ_usual_leading_edge",
    ):
        row[name] = 4
    return row


def export_bytes(*, semantic=True):
    attempt = open_attempt()
    key = "session-open/attempt-open"
    metrics = deterministic_metrics(attempt)
    semantic_result = (
        {
            "reference_trajectory": {
                "user": {
                    "reference_move_coverage": [0.2, None],
                    "median_nearest_reference_distance": 0.3,
                    "exploratory_monotonic_alignment": {
                        "matches": [{"reference_index": 0, "similarity": 0.4}],
                        "mean_similarity": 0.4,
                    },
                },
                "noa": {
                    "reference_move_coverage": [0.5],
                    "median_nearest_reference_distance": 0.2,
                    "exploratory_monotonic_alignment": {
                        "matches": [],
                        "mean_similarity": None,
                    },
                },
            },
            "domain_reference_coverage": {
                domain: {"coverage": 0.35, "available": True}
                for domain in (
                    "calm_deescalation",
                    "viewpoint_reflection",
                    "constructive_next_step",
                    "guiding_questions",
                )
            },
        }
        if semantic
        else None
    )
    return build_analysis_export(
        attempts=[attempt],
        metrics_by_attempt={key: metrics},
        semantic_by_attempt={key: semantic_result},
        closed_attempts=[closed_attempt()],
        survey_rows=[survey_row()],
        filters={
            "mode": "open",
            "statuses": ("success", "no success"),
            "date_range": None,
            "language": "All",
        },
        analyzer_version=ANALYZER_VERSION,
        embedding_model="test-embedding-model",
        generated_at=datetime(2026, 9, 20, 8, 0, tzinfo=timezone.utc),
    )


def read_rows(bundle: bytes, name: str) -> list[dict]:
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        text = archive.read(name).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def test_export_contains_all_tidy_csv_members_and_is_deterministic():
    first = export_bytes()
    second = export_bytes()

    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist() == list(DATASET_DESCRIPTIONS)
    assert first == second


def test_export_preserves_turn_granularity_and_neutralizes_formulas():
    rows = read_rows(export_bytes(), "turns.csv")
    formula = next(row for row in rows if row["role"] == "user")

    assert formula["content"].startswith("'=HYPERLINK")
    assert formula["content_csv_escaped"] == "True"
    assert any("אני כועסת" in row["content"] for row in rows)


def test_export_keeps_unavailable_semantics_explicit():
    attempts = read_rows(export_bytes(semantic=False), "attempts.csv")
    references = read_rows(export_bytes(semantic=False), "reference_coverage.csv")

    assert attempts[0]["semantic_available"] == "False"
    assert attempts[0]["user_mean_reference_coverage"] == ""
    assert references == []


def test_export_excludes_survey_pii_and_records_filters():
    survey_rows = read_rows(export_bytes(), "survey_cases.csv")
    metadata = {
        row["key"]: row["value"]
        for row in read_rows(export_bytes(), "export_metadata.csv")
    }

    assert not any(
        column in survey_rows[0]
        for column in PII_COLUMNS
    )
    assert "must-not-export" not in str(survey_rows)
    assert metadata["filter_language"] == "All"
    assert metadata["sensitive_data"] == "True"


def test_codebook_includes_every_shared_measure_definition():
    rows = read_rows(export_bytes(), "codebook.csv")
    defined = {
        row["variable"]
        for row in rows
        if row["dataset"] == "measure_registry"
    }

    assert defined == set(MEASURE_DEFINITIONS)
    assert all(row["derivation"] for row in rows if row["dataset"] == "measure_registry")
