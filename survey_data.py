"""Validated loader for the de-identified survey/session workbook."""

from __future__ import annotations

import hashlib
from pathlib import Path

from openpyxl import load_workbook


SHEET_NAME = "Sessions + Survey 1 + Survey 2"
PII_COLUMNS = {"username", "S1_username", "S2_username", "S1_gender", "S1_age"}
DOMAIN_COLUMNS = {
    "calm_deescalation": ("S1_pre_calm_tense_conversation", "S2_post_calm_tense_conversation"),
    "viewpoint_reflection": ("S1_pre_reflect_viewpoint", "S2_post_reflect_viewpoint"),
    "constructive_next_step": (
        "S1_pre_constructive_next_step",
        "S2_post_constructive_next_step",
    ),
    "guiding_questions": ("S1_pre_guiding_questions", "S2_post_guiding_questions"),
}
UES_COLUMNS = (
    "S2_UES_lost_myself", "S2_UES_time_slipped_away", "S2_UES_absorbed",
    "S2_UES_frustrated_raw", "S2_UES_confusing_raw", "S2_UES_taxing_raw",
    "S2_UES_worthwhile", "S2_UES_rewarding", "S2_UES_interested",
)
UEQ_COLUMNS = (
    "S2_UEQ_obstructive_supportive", "S2_UEQ_complicated_easy",
    "S2_UEQ_inefficient_efficient", "S2_UEQ_confusing_clear",
    "S2_UEQ_boring_exciting", "S2_UEQ_not_interesting_interesting",
    "S2_UEQ_conventional_inventive", "S2_UEQ_usual_leading_edge",
)
AGENT_COLUMNS = (
    "S2_agent_fake_natural", "S2_agent_machinelike_humanlike",
    "S2_agent_unconscious_conscious", "S2_agent_artificial_lifelike",
    "S2_agent_rigid_elegant", "S2_agent_dislike_like",
    "S2_agent_unfriendly_friendly", "S2_agent_unkind_kind",
    "S2_agent_unpleasant_pleasant", "S2_agent_awful_nice",
)
REQUIRED_COLUMNS = {"session_id", *UES_COLUMNS, *UEQ_COLUMNS, *AGENT_COLUMNS,
                    "S2_reuse_intention", *[column for pair in DOMAIN_COLUMNS.values() for column in pair]}


class SurveyDataError(ValueError):
    """The configured survey workbook cannot safely be analyzed."""


def workbook_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _numeric(value, minimum: float, maximum: float) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if minimum <= value <= maximum else None


def _clean_row(headers: tuple, values: tuple) -> tuple[dict, int]:
    row, invalid = {}, 0
    for column, value in zip(headers, values):
        if column in PII_COLUMNS or column is None:
            continue
        if column in UES_COLUMNS or column in AGENT_COLUMNS or column == "S2_reuse_intention":
            cleaned = _numeric(value, 1, 5)
        elif column in UEQ_COLUMNS:
            cleaned = _numeric(value, 1, 7)
        elif column in {item for pair in DOMAIN_COLUMNS.values() for item in pair}:
            cleaned = _numeric(value, 0, 100)
        else:
            cleaned = value
        if value not in (None, "") and cleaned is None and column.startswith("S"):
            invalid += 1
        row[column] = cleaned
    row["session_id"] = str(row.get("session_id") or "").strip()
    return row, invalid


def load_survey_rows(path: str | Path, sheet_name: str = SHEET_NAME) -> tuple[list[dict], dict]:
    """Read the joined sheet and return safe rows plus transparent diagnostics."""
    path = Path(path)
    if not path.exists():
        raise SurveyDataError(f"Survey workbook is missing: {path}")
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet_name not in workbook.sheetnames:
            raise SurveyDataError(f"Survey sheet is missing: {sheet_name}")
        iterator = workbook[sheet_name].iter_rows(values_only=True)
        headers = tuple(next(iterator, ()))
        missing = REQUIRED_COLUMNS - set(headers)
        if missing:
            raise SurveyDataError(f"Survey sheet is missing required columns: {', '.join(sorted(missing))}")
        rows, invalid_cells, duplicates, seen = [], 0, 0, set()
        for values in iterator:
            if not any(value is not None for value in values):
                continue
            row, invalid = _clean_row(headers, values)
            invalid_cells += invalid
            session_id = row["session_id"]
            if not session_id:
                continue
            if session_id in seen:
                duplicates += 1
                continue
            seen.add(session_id)
            rows.append(row)
    finally:
        workbook.close()
    paired = sum(all(row.get(column) is not None for pair in DOMAIN_COLUMNS.values() for column in pair) for row in rows)
    return rows, {
        "rows_read": len(rows), "valid_session_ids": len(rows), "duplicate_session_ids": duplicates,
        "invalid_cells": invalid_cells, "paired_complete": paired, "source_hash": workbook_hash(path),
    }
