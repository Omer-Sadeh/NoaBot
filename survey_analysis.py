"""Pure scoring, joining, and small-sample summaries for survey outcomes."""

from __future__ import annotations

from collections import defaultdict
from statistics import median

import numpy as np
from scipy import stats

from survey_data import DOMAIN_COLUMNS


UES_SCALES = {
    "Focused attention": ("S2_UES_lost_myself", "S2_UES_time_slipped_away", "S2_UES_absorbed"),
    "Perceived usability": ("S2_UES_frustrated_raw", "S2_UES_confusing_raw", "S2_UES_taxing_raw"),
    "Reward": ("S2_UES_worthwhile", "S2_UES_rewarding", "S2_UES_interested"),
}
UEQ_SCALES = {
    "Pragmatic quality": ("S2_UEQ_obstructive_supportive", "S2_UEQ_complicated_easy", "S2_UEQ_inefficient_efficient", "S2_UEQ_confusing_clear"),
    "Hedonic quality": ("S2_UEQ_boring_exciting", "S2_UEQ_not_interesting_interesting", "S2_UEQ_conventional_inventive", "S2_UEQ_usual_leading_edge"),
}
AGENT_SCALES = {
    "Anthropomorphism": ("S2_agent_fake_natural", "S2_agent_machinelike_humanlike", "S2_agent_unconscious_conscious", "S2_agent_artificial_lifelike", "S2_agent_rigid_elegant"),
    "Likeability": ("S2_agent_dislike_like", "S2_agent_unfriendly_friendly", "S2_agent_unkind_kind", "S2_agent_unpleasant_pleasant", "S2_agent_awful_nice"),
}


def mean_complete(row: dict, columns: tuple[str, ...], transform=lambda value: value) -> float | None:
    values = [transform(row[column]) for column in columns if row.get(column) is not None]
    return round(float(np.mean(values)), 3) if len(values) == len(columns) else None


def score_instruments(row: dict) -> dict:
    scores = {
        name: mean_complete(row, columns, lambda value: 6 - value if name == "Perceived usability" else value)
        for name, columns in UES_SCALES.items()
    }
    scores.update({name: mean_complete(row, columns, lambda value: value - 4) for name, columns in UEQ_SCALES.items()})
    scores["Overall UX"] = mean_complete(row, tuple(item for scale in UEQ_SCALES.values() for item in scale), lambda value: value - 4)
    scores.update({name: mean_complete(row, columns) for name, columns in AGENT_SCALES.items()})
    scores["Reuse intention"] = row.get("S2_reuse_intention")
    return scores


def cronbach_alpha(rows: list[dict], columns: tuple[str, ...], transform=lambda value: value) -> float | None:
    """Return alpha only for complete, non-degenerate scale responses."""
    matrix = [[transform(row[column]) for column in columns]
              for row in rows if all(row.get(column) is not None for column in columns)]
    if len(matrix) < 3:
        return None
    values = np.asarray(matrix, dtype=float)
    item_variance = values.var(axis=0, ddof=1).sum()
    total_variance = values.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return None
    return round(float(len(columns) / (len(columns) - 1) * (1 - item_variance / total_variance)), 3)


def scale_reliability(cases: list[dict]) -> dict[str, float | None]:
    rows = [case["survey"] for case in cases]
    output = {
        name: cronbach_alpha(rows, columns, lambda value: 6 - value if name == "Perceived usability" else value)
        for name, columns in UES_SCALES.items()
    }
    output.update({name: cronbach_alpha(rows, columns, lambda value: value - 4) for name, columns in UEQ_SCALES.items()})
    output.update({name: cronbach_alpha(rows, columns) for name, columns in AGENT_SCALES.items()})
    return output


def domain_scores(row: dict) -> dict:
    return {
        domain: {
            "pre": row.get(columns[0]), "post": row.get(columns[1]),
            "change": round(row[columns[1]] - row[columns[0]], 3)
            if row.get(columns[0]) is not None and row.get(columns[1]) is not None else None,
        }
        for domain, columns in DOMAIN_COLUMNS.items()
    }


def select_attempts(attempts: list[dict]) -> tuple[dict[str, dict], dict[str, int]]:
    grouped = defaultdict(list)
    for attempt in attempts:
        if attempt.get("mode") == "open" and attempt.get("session_id"):
            grouped[attempt["session_id"]].append(attempt)
    def rank(attempt):
        timestamp = attempt.get("timestamp")
        return (attempt.get("status") == "completed", timestamp is not None, str(timestamp or ""), str(attempt.get("doc_id") or ""))
    return ({session_id: max(items, key=rank) for session_id, items in grouped.items()},
            {session_id: len(items) for session_id, items in grouped.items()})


def joined_cases(survey_rows: list[dict], attempts: list[dict], deterministic: dict, semantic: dict) -> tuple[list[dict], dict]:
    selected, candidates = select_attempts(attempts)
    cases, unmatched = [], 0
    for row in survey_rows:
        attempt = selected.get(row["session_id"])
        if attempt is None:
            unmatched += 1
            continue
        key = f"{attempt.get('session_id')}/{attempt.get('doc_id')}"
        cases.append({
            "survey": row, "attempt": attempt, "metrics": deterministic.get(key, {}),
            "semantic": semantic.get(key), "instruments": score_instruments(row),
            "domains": domain_scores(row), "candidate_count": candidates[row["session_id"]],
        })
    return cases, {"matched": len(cases), "survey_without_current_attempt": unmatched,
                   "multi_attempt_sessions": sum(count > 1 for count in candidates.values())}


def observed_value(case: dict, key: str) -> float | None:
    metrics, semantic = case["metrics"], case["semantic"] or {}
    if key.startswith("domain:"):
        return semantic.get("domain_reference_coverage", {}).get(key.split(":", 1)[1], {}).get("coverage")
    values = {
        "duration_minutes": (metrics.get("length", {}).get("duration_seconds") or 0) / 60,
        "trainee_turns": metrics.get("length", {}).get("user", {}).get("turn_count"),
        "tips_shown": case["attempt"].get("tips_shown"),
        "guideline_completion": _completion(metrics),
        "reference_coverage": _reference_coverage(semantic),
        "question_rate": metrics.get("therapeutic_register", {}).get("question_rate"),
        "reflection_markers": metrics.get("therapeutic_register", {}).get("reflection_marker_rate"),
        "style_alignment": metrics.get("style_alignment", {}).get("user_to_noa_mean_alignment"),
    }
    return values.get(key)


def _completion(metrics: dict) -> float | None:
    completion = metrics.get("completion", {})
    if not completion.get("guidelines_total"):
        return None
    return completion["guidelines_cleared"] / completion["guidelines_total"]


def _reference_coverage(semantic: dict) -> float | None:
    values = semantic.get("reference_trajectory", {}).get("user", {}).get("reference_move_coverage", [])
    values = [value for value in values if value is not None]
    return round(float(np.mean(values)), 3) if values else None


def paired_summary(cases: list[dict], domain: str) -> dict:
    values = [(case["domains"][domain]["pre"], case["domains"][domain]["post"]) for case in cases
              if case["domains"][domain]["change"] is not None]
    if not values:
        return {"n": 0}
    pre, post = map(np.array, zip(*values))
    changes = post - pre
    result = stats.wilcoxon(post, pre, method="auto") if len(changes) >= 5 and np.any(changes) else None
    rng = np.random.default_rng(20260904)
    samples = [np.median(rng.choice(changes, size=len(changes), replace=True)) for _ in range(2000)]
    return {"n": len(changes), "pre_median": round(float(np.median(pre)), 2), "post_median": round(float(np.median(post)), 2),
            "change_median": round(float(np.median(changes)), 2), "ci_low": round(float(np.percentile(samples, 2.5)), 2),
            "ci_high": round(float(np.percentile(samples, 97.5)), 2), "p_value": round(float(result.pvalue), 4) if result else None}


def spearman_summary(cases: list[dict], left, right) -> dict:
    pairs = []
    for case in cases:
        x = left(case)
        y = right(case)
        if x is not None and y is not None:
            pairs.append((x, y))
    if len(pairs) < 3:
        return {"n": len(pairs), "rho": None}
    x, y = map(np.array, zip(*pairs))
    rho = stats.spearmanr(x, y).statistic
    if len(pairs) < 10:
        return {"n": len(pairs), "rho": round(float(rho), 3), "p_value": None}
    rng = np.random.default_rng(20260904)
    bootstrap = [stats.spearmanr(*(np.array(pairs)[rng.integers(0, len(pairs), len(pairs))].T)).statistic for _ in range(1000)]
    return {"n": len(pairs), "rho": round(float(rho), 3), "p_value": round(float(stats.spearmanr(x, y).pvalue), 4),
            "ci_low": round(float(np.nanpercentile(bootstrap, 2.5)), 3), "ci_high": round(float(np.nanpercentile(bootstrap, 97.5)), 3)}
