"""Pure builders for the research analysis ZIP export."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import date, datetime, timezone
from statistics import mean

from script_analysis_explanations import measure_codebook_rows
from survey_analysis import (
    annotate_closed_stages,
    closed_accuracy_rate,
    closed_cohort_summary,
    closed_distractor_patterns,
    closed_stage_difficulty,
    concordance_quadrants,
    joined_cases,
    observed_value,
    open_stage_coverage,
    paired_summary,
    scale_reliability,
    select_closed_attempts,
    spearman_summary,
    triangulation_profiles,
)
from survey_data import DOMAIN_COLUMNS, PII_COLUMNS


EXPORT_VERSION = "1"
CURATED_ASSOCIATIONS = (
    ("Focused attention", "duration_minutes"),
    ("Focused attention", "trainee_turns"),
    ("Perceived usability", "tips_shown"),
    ("Reward", "guideline_completion"),
    ("Reuse intention", "guideline_completion"),
    ("Anthropomorphism", "style_alignment"),
)

DATASET_DESCRIPTIONS = {
    "export_metadata.csv": "Export provenance, active filters, versions, sample sizes, and sensitivity.",
    "attempts.csv": "One row per filtered open conversation attempt.",
    "turns.csv": "One row per recoverable open-conversation turn, including raw text.",
    "sections.csv": "One row per recoverable attempt goal section.",
    "reference_coverage.csv": "One row per attempt, speaker role, and authored reference move.",
    "survey_cases.csv": "One row per survey record matched to the selected open attempt.",
    "closed_attempts.csv": "One row per selected completed closed-script attempt.",
    "closed_stages.csv": "One row per recoverable closed-script stage response.",
    "statistical_results.csv": "One row per displayed aggregate estimate or inferential result.",
    "codebook.csv": "Variable and measure definitions for every exported CSV.",
}

FIELD_DESCRIPTIONS = {
    "session_id": "Internal application session identifier.",
    "attempt_id": "Saved attempt identifier, using attempt_id or document ID.",
    "doc_id": "Firestore conversation document identifier.",
    "timestamp": "Saved attempt timestamp in ISO 8601 form.",
    "language": "Recorded session language code.",
    "status": "Recorded attempt outcome status.",
    "mode": "Conversation mode.",
    "source_hash": "Hash of the analysis source fields used for cache invalidation.",
    "semantic_available": "Whether cached semantic analysis was available at export time.",
    "turn_index": "One-based turn order within the attempt.",
    "role": "Conversation speaker role.",
    "content": "Raw conversation text, spreadsheet-escaped only when content_csv_escaped is true.",
    "content_csv_escaped": "True when a leading apostrophe was added to prevent spreadsheet formula execution.",
    "section": "One-based goal-section or closed-script stage number.",
    "n": "Number of usable observations contributing to the result.",
    "estimate": "Primary numeric estimate for the named result.",
    "ci_low": "Lower endpoint of the reported 95% interval.",
    "ci_high": "Upper endpoint of the reported 95% interval.",
    "p_value": "Reported p-value; blank when not computed or not eligible.",
}


def analysis_attempt_key(attempt: dict) -> str:
    return f"{attempt.get('session_id')}/{attempt.get('doc_id')}"


def _json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _plain(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def spreadsheet_safe(value) -> tuple[object, bool]:
    """Neutralize spreadsheet formulas while retaining a reversible indicator."""
    value = _plain(value)
    if not isinstance(value, str):
        return value, False
    stripped = value.lstrip()
    if stripped.startswith(("=", "+", "-", "@", "\t", "\r")):
        return f"'{value}", True
    return value, False


def _fieldnames(rows: list[dict], preferred: tuple[str, ...] = ()) -> list[str]:
    keys = {key for row in rows for key in row}
    return list(preferred) + sorted(keys - set(preferred))


def _csv_bytes(rows: list[dict], fieldnames: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        safe_row = {key: spreadsheet_safe(row.get(key))[0] for key in fieldnames}
        writer.writerow(safe_row)
    return stream.getvalue().encode("utf-8-sig")


def _mean(values) -> float | None:
    usable = [value for value in values if value is not None]
    return round(float(mean(usable)), 6) if usable else None


def _attempt_rows(
    attempts: list[dict],
    metrics_by_attempt: dict[str, dict],
    semantic_by_attempt: dict[str, dict | None],
) -> list[dict]:
    rows = []
    for attempt in attempts:
        key = analysis_attempt_key(attempt)
        metrics = metrics_by_attempt[key]
        semantic = semantic_by_attempt.get(key) or {}
        length = metrics.get("length", {})
        register = metrics.get("therapeutic_register", {})
        style = metrics.get("style_alignment", {})
        completion = metrics.get("completion", {})
        row = {
            "session_id": attempt.get("session_id"),
            "doc_id": attempt.get("doc_id"),
            "attempt_id": metrics.get("attempt_id"),
            "timestamp": attempt.get("timestamp"),
            "mode": attempt.get("mode"),
            "status": attempt.get("status"),
            "language": attempt.get("session_language"),
            "schema_version": attempt.get("schema_version"),
            "app_version": attempt.get("instrument", {}).get("app_version"),
            "analysis_source_hash": metrics.get("source_hash"),
            "duration_seconds": length.get("duration_seconds"),
            "all_turn_count": length.get("all", {}).get("turn_count"),
            "all_word_count": length.get("all", {}).get("word_count"),
            "trainee_turn_count": length.get("user", {}).get("turn_count"),
            "trainee_word_count": length.get("user", {}).get("word_count"),
            "noa_turn_count": length.get("noa", {}).get("turn_count"),
            "noa_word_count": length.get("noa", {}).get("word_count"),
            "trainee_median_words_per_turn": length.get("user", {}).get("median_words_per_turn"),
            "trainee_median_words_per_sentence": length.get("user", {}).get("median_words_per_sentence"),
            "trainee_mattr_50": length.get("user", {}).get("mattr_50"),
            "trainee_repetition_rate": length.get("user", {}).get("repetition_rate"),
            "question_rate_per_turn": register.get("question_rate"),
            "reflection_marker_rate_per_word": register.get("reflection_marker_rate"),
            "second_person_rate_per_word": register.get("second_person_rate"),
            "first_person_rate_per_word": register.get("first_person_rate"),
            "style_alignment_available": style.get("available"),
            "trainee_to_noa_style_alignment": style.get("user_to_noa_mean_alignment"),
            "noa_to_trainee_style_alignment": style.get("noa_to_user_mean_alignment"),
            "style_alignment_shuffled_baseline": style.get("within_attempt_shuffled_null"),
            "style_alignment_pair_count": style.get("user_to_noa_pairs"),
            "guidelines_cleared": completion.get("guidelines_cleared"),
            "guidelines_total": completion.get("guidelines_total"),
            "is_llm_judged_success": completion.get("is_llm_judged_success"),
            "tips_shown": attempt.get("tips_shown"),
            "semantic_available": bool(semantic),
        }
        for role in ("user", "noa"):
            trajectory = semantic.get("reference_trajectory", {}).get(role, {})
            row[f"{role}_mean_reference_coverage"] = _mean(
                trajectory.get("reference_move_coverage", [])
            )
            row[f"{role}_median_nearest_reference_distance"] = trajectory.get(
                "median_nearest_reference_distance"
            )
            row[f"{role}_sequence_fidelity"] = trajectory.get(
                "exploratory_monotonic_alignment", {}
            ).get("mean_similarity")
        for domain in DOMAIN_COLUMNS:
            row[f"domain_{domain}_coverage"] = semantic.get(
                "domain_reference_coverage", {}
            ).get(domain, {}).get("coverage")
        rows.append(row)
    return rows


def _turn_rows(attempts: list[dict]) -> list[dict]:
    rows = []
    for attempt in attempts:
        for index, turn in enumerate(attempt.get("turns") or [], start=1):
            content, escaped = spreadsheet_safe(turn.get("content"))
            rows.append(
                {
                    "session_id": attempt.get("session_id"),
                    "doc_id": attempt.get("doc_id"),
                    "turn_index": index,
                    "role": turn.get("role"),
                    "elapsed_seconds": turn.get("elapsed_seconds"),
                    "input_modality": turn.get("input_modality"),
                    "content": content,
                    "content_csv_escaped": escaped,
                }
            )
    return rows


def _section_rows(attempts: list[dict]) -> list[dict]:
    rows = []
    for attempt in attempts:
        durations = attempt.get("section_durations_seconds") or []
        user_turns = attempt.get("section_user_turns") or []
        flags = attempt.get("section_completion_flags") or []
        transitions = attempt.get("section_transition_events") or []
        count = max(len(durations), len(user_turns), len(flags), len(transitions))
        for index in range(count):
            transition = transitions[index] if index < len(transitions) else {}
            rows.append(
                {
                    "session_id": attempt.get("session_id"),
                    "doc_id": attempt.get("doc_id"),
                    "section": index + 1,
                    "duration_seconds": durations[index] if index < len(durations) else None,
                    "user_turns": user_turns[index] if index < len(user_turns) else transition.get("user_turns"),
                    "completed": flags[index] if index < len(flags) else transition.get("completed"),
                    "transition_elapsed_seconds": transition.get("elapsed_seconds"),
                    "time_semantics": attempt.get("section_time_semantics"),
                }
            )
    return rows


def _reference_rows(
    attempts: list[dict], semantic_by_attempt: dict[str, dict | None]
) -> list[dict]:
    rows = []
    for attempt in attempts:
        semantic = semantic_by_attempt.get(analysis_attempt_key(attempt)) or {}
        for role, trajectory in semantic.get("reference_trajectory", {}).items():
            alignment = trajectory.get("exploratory_monotonic_alignment", {})
            matches = alignment.get("matches") or []
            similarities = {
                match.get("reference_index"): match.get("similarity") for match in matches
            }
            for index, coverage in enumerate(
                trajectory.get("reference_move_coverage") or [], start=1
            ):
                rows.append(
                    {
                        "session_id": attempt.get("session_id"),
                        "doc_id": attempt.get("doc_id"),
                        "language": attempt.get("session_language"),
                        "role": role,
                        "reference_move_index": index,
                        "coverage": coverage,
                        "ordered_match_similarity": similarities.get(index - 1),
                        "sequence_fidelity_mean": alignment.get("mean_similarity"),
                    }
                )
    return rows


def _survey_rows(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        attempt = case["attempt"]
        row = {
            "session_id": attempt.get("session_id"),
            "doc_id": attempt.get("doc_id"),
            "language": attempt.get("session_language"),
            "candidate_attempt_count": case.get("candidate_count"),
        }
        for name, value in case.get("survey", {}).items():
            if name not in PII_COLUMNS and name != "session_id":
                row[f"survey_{name}"] = value
        for domain, values in case.get("domains", {}).items():
            for statistic, value in values.items():
                row[f"{domain}_{statistic}"] = value
            row[f"{domain}_reference_coverage"] = observed_value(
                case, f"domain:{domain}"
            )
        for name, value in case.get("instruments", {}).items():
            row[f"scale_{name.lower().replace(' ', '_')}"] = value
        rows.append(row)
    return rows


def _closed_rows(closed_attempts: list[dict]) -> tuple[list[dict], list[dict]]:
    attempts, stages = [], []
    for attempt in select_closed_attempts(closed_attempts):
        attempts.append(
            {
                "session_id": attempt.get("session_id"),
                "doc_id": attempt.get("doc_id"),
                "timestamp": attempt.get("timestamp"),
                "mode": attempt.get("mode"),
                "status": attempt.get("status"),
                "language": attempt.get("session_language"),
                "total_questions": attempt.get("total_questions"),
                "correct_answers": attempt.get("correct_answers"),
                "accuracy_rate": closed_accuracy_rate(attempt),
            }
        )
        for stage in annotate_closed_stages(attempt):
            selected, escaped = spreadsheet_safe(stage.get("selected_answer"))
            stages.append(
                {
                    "session_id": attempt.get("session_id"),
                    "doc_id": attempt.get("doc_id"),
                    "section": stage.get("stage"),
                    "language": stage.get("language"),
                    "selected_answer": selected,
                    "selected_answer_csv_escaped": escaped,
                    "is_correct": stage.get("is_correct"),
                    "option_type": stage.get("option_type"),
                    "failure_mode": stage.get("failure_mode"),
                    "failure_mode_label": stage.get("failure_mode_label"),
                }
            )
    return attempts, stages


def _result_row(
    analysis: str,
    measure: str,
    *,
    estimate=None,
    n=None,
    domain=None,
    stage=None,
    group=None,
    ci_low=None,
    ci_high=None,
    p_value=None,
    denominator=None,
    notes=None,
) -> dict:
    return {
        "analysis": analysis,
        "measure": measure,
        "domain": domain,
        "stage": stage,
        "group": group,
        "estimate": estimate,
        "n": n,
        "denominator": denominator,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "notes": notes,
    }


def _statistical_rows(cases: list[dict], closed_attempts: list[dict]) -> list[dict]:
    rows = []
    for domain in DOMAIN_COLUMNS:
        paired = paired_summary(cases, domain)
        for measure in ("pre_median", "post_median", "change_median"):
            rows.append(
                _result_row(
                    "self_report_change",
                    measure,
                    estimate=paired.get(measure),
                    n=paired.get("n", 0),
                    domain=domain,
                    ci_low=paired.get("ci_low") if measure == "change_median" else None,
                    ci_high=paired.get("ci_high") if measure == "change_median" else None,
                    p_value=paired.get("p_value") if measure == "change_median" else None,
                )
            )
        association = spearman_summary(
            cases,
            lambda case, d=domain: case["domains"][d]["post"],
            lambda case, d=domain: observed_value(case, f"domain:{d}"),
        )
        rows.append(
            _result_row(
                "spearman_association",
                "post_confidence_vs_domain_coverage",
                estimate=association.get("rho"),
                n=association.get("n"),
                domain=domain,
                ci_low=association.get("ci_low"),
                ci_high=association.get("ci_high"),
                p_value=association.get("p_value"),
            )
        )
        calibration = concordance_quadrants(cases, domain)
        for measure, values in (
            ("post_vs_proxy", calibration.get("post_vs_proxy", {})),
            ("change_vs_proxy", calibration.get("change_vs_proxy", {})),
        ):
            if values:
                rows.append(
                    _result_row(
                        "calibration_association",
                        measure,
                        estimate=values.get("rho"),
                        n=values.get("n"),
                        domain=domain,
                        ci_low=values.get("ci_low"),
                        ci_high=values.get("ci_high"),
                        p_value=values.get("p_value"),
                    )
                )
        if not calibration.get("quadrants"):
            rows.append(
                _result_row(
                    "calibration_quadrants",
                    "proportion",
                    n=calibration.get("n", 0),
                    domain=domain,
                    notes="Insufficient complete matched values for quadrant estimates.",
                )
            )
        for quadrant, values in calibration.get("quadrants", {}).items():
            rows.append(
                _result_row(
                    "calibration_quadrants",
                    "proportion",
                    estimate=values.get("proportion"),
                    n=values.get("count"),
                    denominator=calibration.get("n"),
                    domain=domain,
                    group=quadrant,
                    ci_low=values.get("ci_low"),
                    ci_high=values.get("ci_high"),
                )
            )
    for scale, observed in CURATED_ASSOCIATIONS:
        association = spearman_summary(
            cases,
            lambda case, name=scale: case["instruments"].get(name),
            lambda case, key=observed: observed_value(case, key),
        )
        rows.append(
            _result_row(
                "spearman_association",
                f"{scale}_vs_{observed}",
                estimate=association.get("rho"),
                n=association.get("n"),
                ci_low=association.get("ci_low"),
                ci_high=association.get("ci_high"),
                p_value=association.get("p_value"),
            )
        )
    for scale, alpha in scale_reliability(cases).items():
        rows.append(
            _result_row("scale_reliability", "cronbach_alpha", estimate=alpha, group=scale)
        )

    closed_summary = closed_cohort_summary(closed_attempts)
    for measure in ("median_accuracy", "mean_accuracy", "perfect_rate"):
        rows.append(
            _result_row(
                "closed_accuracy",
                measure,
                estimate=closed_summary.get(measure),
                n=closed_summary.get("n", 0),
            )
        )
    difficulty = closed_stage_difficulty(closed_attempts)
    for item in difficulty:
        rows.append(
            _result_row(
                "closed_stage_difficulty",
                "correct_rate",
                estimate=item.get("correct_rate"),
                n=item.get("n"),
                stage=item.get("stage"),
                domain=item.get("primary_domain"),
                ci_low=item.get("ci_low"),
                ci_high=item.get("ci_high"),
            )
        )
    for item in closed_distractor_patterns(closed_attempts):
        rows.append(
            _result_row(
                "closed_distractor_pattern",
                "share_of_stage_errors",
                estimate=item.get("share_of_stage_errors"),
                n=item.get("count"),
                denominator=item.get("stage_error_n"),
                stage=item.get("stage"),
                domain=item.get("primary_domain"),
                group=item.get("failure_mode"),
                notes=item.get("failure_mode_label"),
            )
        )
    coverage = open_stage_coverage(cases)
    for item in coverage.get("stages", []):
        rows.append(
            _result_row(
                "reference_coverage",
                "median_stage_coverage",
                estimate=item.get("median_coverage"),
                n=item.get("n"),
                stage=item.get("stage"),
                domain=item.get("primary_domain"),
            )
        )
    for item in triangulation_profiles(difficulty, coverage, cases):
        for source, estimate_key, n_key, low_key, high_key in (
            ("closed_correct_rate", "closed_correct_rate", "closed_n", "closed_ci_low", "closed_ci_high"),
            ("open_median_coverage", "open_median_coverage", "open_n", None, None),
            ("survey_change_median", "survey_change_median", "survey_n", "survey_ci_low", "survey_ci_high"),
        ):
            rows.append(
                _result_row(
                    "triangulation",
                    source,
                    estimate=item.get(estimate_key),
                    n=item.get(n_key),
                    stage=item.get("stage"),
                    domain=item.get("primary_domain"),
                    ci_low=item.get(low_key) if low_key else None,
                    ci_high=item.get(high_key) if high_key else None,
                    notes="Independent cohorts; values are not person-linked.",
                )
            )
    return rows


def _metadata_rows(
    *,
    attempts: list[dict],
    closed_attempts: list[dict],
    cases: list[dict],
    semantic_by_attempt: dict[str, dict | None],
    filters: dict,
    join_info: dict,
    generated_at: datetime,
    analyzer_version: str,
    embedding_model: str,
) -> list[dict]:
    metadata = {
        "export_version": EXPORT_VERSION,
        "generated_at_utc": generated_at.astimezone(timezone.utc).isoformat(),
        "sensitive_data": True,
        "sensitivity_note": "Contains internal identifiers and raw conversation text; handle as identifiable research data.",
        "filter_mode": filters.get("mode"),
        "filter_statuses": filters.get("statuses"),
        "filter_date_range": filters.get("date_range"),
        "filter_language": filters.get("language"),
        "open_attempt_count": len(attempts),
        "open_unique_session_count": len(
            {attempt.get("session_id") for attempt in attempts if attempt.get("session_id")}
        ),
        "open_completed_attempt_count": sum(
            attempt.get("status") == "completed" for attempt in attempts
        ),
        "open_llm_judged_success_count": sum(
            bool(attempt.get("is_successful")) for attempt in attempts
        ),
        "closed_attempt_count_before_per_session_selection": len(closed_attempts),
        "matched_survey_case_count": len(cases),
        "unmatched_survey_count": join_info.get("survey_without_current_attempt", 0),
        "multi_attempt_open_session_count": join_info.get("multi_attempt_sessions", 0),
        "semantic_available_count": sum(
            semantic_by_attempt.get(analysis_attempt_key(attempt)) is not None
            for attempt in attempts
        ),
        "analyzer_version": analyzer_version,
        "embedding_model": embedding_model,
        "semantic_export_policy": "Cached results only; export does not trigger embedding analysis.",
    }
    return [{"key": key, "value": value} for key, value in metadata.items()]


def _codebook_rows(tables: dict[str, tuple[list[dict], list[str]]]) -> list[dict]:
    rows = []
    for dataset, (_, fields) in tables.items():
        if dataset == "codebook.csv":
            continue
        for field in fields:
            rows.append(
                {
                    "dataset": dataset,
                    "variable": field,
                    "description": FIELD_DESCRIPTIONS.get(
                        field, field.replace("_", " ").capitalize()
                    ),
                    "type": "mixed; inspect non-missing values",
                    "values_or_unit": "",
                    "missing": "Blank means unavailable, inapplicable, or not computed for this row.",
                    "source": DATASET_DESCRIPTIONS[dataset],
                    "derivation": "Direct export or named analysis result; see matching measure definitions below.",
                    "limitations": "",
                }
            )
    for definition in measure_codebook_rows():
        rows.append(
            {
                "dataset": "measure_registry",
                "variable": definition["measure_id"],
                "description": definition["summary"],
                "type": "derived measure",
                "values_or_unit": definition["unit"],
                "missing": definition["missingness"],
                "source": definition["source"],
                "derivation": definition["operationalization"],
                "limitations": definition["limitations"],
            }
        )
    return rows


def build_analysis_export(
    *,
    attempts: list[dict],
    metrics_by_attempt: dict[str, dict],
    semantic_by_attempt: dict[str, dict | None],
    closed_attempts: list[dict],
    survey_rows: list[dict],
    filters: dict,
    analyzer_version: str,
    embedding_model: str,
    generated_at: datetime | None = None,
) -> bytes:
    """Build a filter-aware research bundle without performing external I/O."""
    generated_at = generated_at or datetime.now(timezone.utc)
    cases, join_info = joined_cases(
        survey_rows, attempts, metrics_by_attempt, semantic_by_attempt
    )
    closed_rows, closed_stage_rows = _closed_rows(closed_attempts)
    table_rows = {
        "export_metadata.csv": _metadata_rows(
            attempts=attempts,
            closed_attempts=closed_attempts,
            cases=cases,
            semantic_by_attempt=semantic_by_attempt,
            filters=filters,
            join_info=join_info,
            generated_at=generated_at,
            analyzer_version=analyzer_version,
            embedding_model=embedding_model,
        ),
        "attempts.csv": _attempt_rows(attempts, metrics_by_attempt, semantic_by_attempt),
        "turns.csv": _turn_rows(attempts),
        "sections.csv": _section_rows(attempts),
        "reference_coverage.csv": _reference_rows(attempts, semantic_by_attempt),
        "survey_cases.csv": _survey_rows(cases),
        "closed_attempts.csv": closed_rows,
        "closed_stages.csv": closed_stage_rows,
        "statistical_results.csv": _statistical_rows(cases, closed_attempts),
    }
    preferred = {
        "export_metadata.csv": ("key", "value"),
        "attempts.csv": (
            "session_id", "doc_id", "attempt_id", "timestamp", "mode", "status",
            "language", "schema_version", "semantic_available",
        ),
        "turns.csv": ("session_id", "doc_id", "turn_index", "role", "elapsed_seconds", "input_modality", "content", "content_csv_escaped"),
        "sections.csv": (
            "session_id", "doc_id", "section", "duration_seconds", "user_turns",
            "completed", "transition_elapsed_seconds", "time_semantics",
        ),
        "reference_coverage.csv": (
            "session_id", "doc_id", "language", "role", "reference_move_index",
            "coverage", "ordered_match_similarity", "sequence_fidelity_mean",
        ),
        "survey_cases.csv": ("session_id", "doc_id", "language", "candidate_attempt_count"),
        "closed_attempts.csv": (
            "session_id", "doc_id", "timestamp", "mode", "status", "language",
            "total_questions", "correct_answers", "accuracy_rate",
        ),
        "closed_stages.csv": (
            "session_id", "doc_id", "section", "language", "selected_answer",
            "selected_answer_csv_escaped", "is_correct", "option_type",
            "failure_mode", "failure_mode_label",
        ),
        "statistical_results.csv": ("analysis", "measure", "domain", "stage", "group", "estimate", "n", "denominator", "ci_low", "ci_high", "p_value", "notes"),
    }
    tables = {
        name: (rows, _fieldnames(rows, preferred[name])) for name, rows in table_rows.items()
    }
    codebook = _codebook_rows(tables)
    tables["codebook.csv"] = (
        codebook,
        [
            "dataset",
            "variable",
            "description",
            "type",
            "values_or_unit",
            "missing",
            "source",
            "derivation",
            "limitations",
        ],
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in DATASET_DESCRIPTIONS:
            rows, fields = tables[name]
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, _csv_bytes(rows, fields))
    return buffer.getvalue()
