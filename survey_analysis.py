"""Pure scoring, joining, and small-sample summaries for survey outcomes."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy import stats

from conversation_csv import classify_closed_option, parse_closed_stage_results
from script_loader import load_closed_script
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

# Declared mapping: closed stages ↔ survey domains. guiding_questions is cross-cutting.
STAGE_SKILL_MAP = {
    1: {
        "therapist_move": "normalize_and_explore",
        "therapist_label": "Normalize conflict and explore typical coping",
        "noa_behavior": "recognize_conflict_pattern",
        "noa_label": "Recognize the conflict pattern and how she usually reacts",
        "primary_domain": "calm_deescalation",
        "survey_domains": ("calm_deescalation", "viewpoint_reflection", "guiding_questions"),
    },
    2: {
        "therapist_move": "clarify_goal",
        "therapist_label": "Clarify goals and desired message",
        "noa_behavior": "articulate_goals_feelings",
        "noa_label": "Articulate feelings, goals, and what she wants to say",
        "primary_domain": "viewpoint_reflection",
        "survey_domains": ("viewpoint_reflection", "guiding_questions"),
    },
    3: {
        "therapist_move": "practice_assertiveness",
        "therapist_label": "Prompt assertive practice",
        "noa_behavior": "formulate_assertive_message",
        "noa_label": "Formulate an assertive message for Dana",
        "primary_domain": "constructive_next_step",
        "survey_domains": ("constructive_next_step", "guiding_questions"),
    },
    4: {
        "therapist_move": "rehearse_repair",
        "therapist_label": "Rehearse adaptive repair after pushback",
        "noa_behavior": "respond_adaptively",
        "noa_label": "Respond adaptively to Dana's counter-response",
        "primary_domain": "constructive_next_step",
        "survey_domains": ("constructive_next_step",),
    },
    5: {
        "therapist_move": "reinforce_compromise",
        "therapist_label": "Reinforce compromise and negotiation",
        "noa_behavior": "negotiate_resolution",
        "noa_label": "Offer a workable compromise for next time",
        "primary_domain": "constructive_next_step",
        "survey_domains": ("constructive_next_step",),
    },
}

DISTRACTOR_FAILURE_MODES = {
    (1, "incorrect_1"): "interest_without_skill_focus",
    (1, "incorrect_2"): "perspective_taking_too_early",
    (2, "incorrect_1"): "pattern_reflection_without_skill",
    (2, "incorrect_2"): "causes_instead_of_skill",
    (3, "incorrect_1"): "seek_external_support",
    (3, "incorrect_2"): "avoidance",
    (4, "incorrect_1"): "relationship_history_detour",
    (4, "incorrect_2"): "curiosity_without_practice",
    (5, "incorrect_1"): "relinquish_agency",
    (5, "incorrect_2"): "catastrophize_next_failure",
}

FAILURE_MODE_LABELS = {
    "interest_without_skill_focus": "Shows interest but skips conflict-skill focus",
    "perspective_taking_too_early": "Shifts to Dana's perspective too early",
    "pattern_reflection_without_skill": "Reflects patterns without building skill",
    "causes_instead_of_skill": "Explores causes instead of conflict management",
    "seek_external_support": "Diverts to external support",
    "avoidance": "Encourages avoidance",
    "relationship_history_detour": "Detours into relationship history",
    "curiosity_without_practice": "Expresses curiosity without practice",
    "relinquish_agency": "Hands control to Dana",
    "catastrophize_next_failure": "Focuses on failure without compromise",
    "unknown": "Unrecognized choice",
}

BOOTSTRAP_SEED = 20260904



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
    duration_seconds = metrics.get("length", {}).get("duration_seconds")
    values = {
        "duration_minutes": duration_seconds / 60
        if duration_seconds is not None
        else None,
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
    if completion.get("guidelines_cleared") is None or not completion.get("guidelines_total"):
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
    rng = np.random.default_rng(BOOTSTRAP_SEED)
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
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"n": len(pairs), "rho": None, "p_value": None}
    rho = stats.spearmanr(x, y).statistic
    if np.isnan(rho):
        return {"n": len(pairs), "rho": None, "p_value": None}
    if len(pairs) < 10:
        return {"n": len(pairs), "rho": round(float(rho), 3), "p_value": None}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = [stats.spearmanr(*(np.array(pairs)[rng.integers(0, len(pairs), len(pairs))].T)).statistic for _ in range(1000)]
    return {"n": len(pairs), "rho": round(float(rho), 3), "p_value": round(float(stats.spearmanr(x, y).pvalue), 4),
            "ci_low": round(float(np.nanpercentile(bootstrap, 2.5)), 3), "ci_high": round(float(np.nanpercentile(bootstrap, 97.5)), 3)}


def ground_truth_pathway() -> list[dict]:
    """Return the authored therapist/Noa skill pathway for stages 1–5."""
    return [
        {
            "stage": stage,
            **skill,
            "is_cross_cutting": "guiding_questions" in skill["survey_domains"]
            and skill["primary_domain"] != "guiding_questions",
        }
        for stage, skill in STAGE_SKILL_MAP.items()
    ]


def annotate_closed_stages(attempt: dict) -> list[dict]:
    """Attach option type and pedagogical failure mode to closed stage results."""
    language = attempt.get("session_language") or "en"
    if language not in {"en", "he"}:
        language = "en"
    script = [entry for entry in load_closed_script(language) if entry.get("correct_answer")]
    stages = attempt.get("closed_stage_results")
    if not stages:
        stages = parse_closed_stage_results(attempt.get("data") or "")
    annotated = []
    for stage_result in stages:
        stage = stage_result.get("stage")
        script_entry = script[stage - 1] if isinstance(stage, int) and 1 <= stage <= len(script) else {}
        option_type = classify_closed_option(stage_result.get("selected_answer"), script_entry)
        if stage_result.get("is_correct") and option_type == "unknown":
            option_type = "correct"
        failure_mode = None if option_type == "correct" else DISTRACTOR_FAILURE_MODES.get(
            (stage, option_type), "unknown"
        )
        annotated.append(
            {
                **stage_result,
                "option_type": option_type,
                "failure_mode": failure_mode,
                "failure_mode_label": FAILURE_MODE_LABELS.get(failure_mode) if failure_mode else None,
                "language": language,
            }
        )
    return annotated


def select_closed_attempts(attempts: list[dict]) -> list[dict]:
    """Prefer one completed closed attempt per session (latest completed)."""
    grouped = defaultdict(list)
    for attempt in attempts:
        if attempt.get("mode") != "closed" or not attempt.get("session_id"):
            continue
        if attempt.get("status") == "ongoing":
            continue
        grouped[attempt["session_id"]].append(attempt)

    def rank(attempt):
        timestamp = attempt.get("timestamp")
        return (
            attempt.get("status") == "completed",
            timestamp is not None,
            str(timestamp or ""),
            str(attempt.get("doc_id") or ""),
        )

    return [max(items, key=rank) for items in grouped.values()]


def closed_accuracy_rate(attempt: dict) -> float | None:
    total = attempt.get("total_questions")
    correct = attempt.get("correct_answers")
    if total in (None, 0) or correct is None:
        stages = annotate_closed_stages(attempt)
        if not stages:
            return None
        return round(sum(1 for stage in stages if stage.get("is_correct")) / len(stages), 4)
    return round(correct / total, 4)


def _proportion_ci(successes: int, n: int) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.binomial(n, successes / n, size=2000) / n
    return round(float(np.percentile(samples, 2.5)), 3), round(float(np.percentile(samples, 97.5)), 3)


def closed_cohort_summary(closed_attempts: list[dict]) -> dict:
    selected = select_closed_attempts(closed_attempts)
    rates = []
    perfect = 0
    with_stages = 0
    languages = Counter()
    for attempt in selected:
        rate = closed_accuracy_rate(attempt)
        if rate is None:
            continue
        rates.append(rate)
        languages[attempt.get("session_language") or "unknown"] += 1
        if rate >= 1.0:
            perfect += 1
        if annotate_closed_stages(attempt):
            with_stages += 1
    if not rates:
        return {
            "n": 0,
            "with_stage_detail": 0,
            "median_accuracy": None,
            "mean_accuracy": None,
            "perfect_rate": None,
            "languages": {},
        }
    return {
        "n": len(rates),
        "with_stage_detail": with_stages,
        "median_accuracy": round(float(np.median(rates)), 3),
        "mean_accuracy": round(float(np.mean(rates)), 3),
        "perfect_rate": round(perfect / len(rates), 3),
        "accuracy_values": rates,
        "languages": dict(languages),
    }


def closed_stage_difficulty(closed_attempts: list[dict]) -> list[dict]:
    selected = select_closed_attempts(closed_attempts)
    by_stage = defaultdict(list)
    for attempt in selected:
        for stage in annotate_closed_stages(attempt):
            if stage.get("is_correct") is None:
                continue
            by_stage[stage["stage"]].append(bool(stage["is_correct"]))
    rows = []
    for stage in sorted(by_stage):
        outcomes = by_stage[stage]
        successes = sum(outcomes)
        n = len(outcomes)
        rate = successes / n if n else None
        ci_low, ci_high = _proportion_ci(successes, n)
        skill = STAGE_SKILL_MAP.get(stage, {})
        rows.append(
            {
                "stage": stage,
                "n": n,
                "correct_rate": round(rate, 3) if rate is not None else None,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "therapist_label": skill.get("therapist_label"),
                "primary_domain": skill.get("primary_domain"),
            }
        )
    return rows


def closed_distractor_patterns(closed_attempts: list[dict]) -> list[dict]:
    selected = select_closed_attempts(closed_attempts)
    counts = Counter()
    stage_totals = Counter()
    for attempt in selected:
        for stage in annotate_closed_stages(attempt):
            if stage.get("is_correct"):
                continue
            stage_number = stage.get("stage")
            stage_totals[stage_number] += 1
            mode = stage.get("failure_mode") or "unknown"
            counts[(stage_number, mode)] += 1
    rows = []
    for (stage, mode), count in sorted(counts.items()):
        total = stage_totals[stage]
        rows.append(
            {
                "stage": stage,
                "failure_mode": mode,
                "failure_mode_label": FAILURE_MODE_LABELS.get(mode, mode),
                "count": count,
                "stage_error_n": total,
                "share_of_stage_errors": round(count / total, 3) if total else None,
                "primary_domain": STAGE_SKILL_MAP.get(stage, {}).get("primary_domain"),
            }
        )
    return rows


def open_stage_coverage(cases: list[dict]) -> dict:
    """Aggregate open semantic proximity to each authored closed-script stage."""
    by_stage = defaultdict(list)
    sequence_scores = []
    for case in cases:
        semantic = case.get("semantic") or {}
        trajectory = semantic.get("reference_trajectory", {}).get("user", {})
        coverage = trajectory.get("reference_move_coverage") or []
        for index, value in enumerate(coverage):
            if value is None:
                continue
            by_stage[index + 1].append(float(value))
        mean_similarity = trajectory.get("exploratory_monotonic_alignment", {}).get("mean_similarity")
        mean_coverage = _reference_coverage(semantic)
        if mean_similarity is not None and mean_coverage is not None:
            sequence_scores.append(
                {"coverage": mean_coverage, "sequence_fidelity": mean_similarity}
            )
    rows = []
    for stage in sorted(by_stage):
        values = by_stage[stage]
        skill = STAGE_SKILL_MAP.get(stage, {})
        rows.append(
            {
                "stage": stage,
                "n": len(values),
                "median_coverage": round(float(np.median(values)), 3),
                "mean_coverage": round(float(np.mean(values)), 3),
                "therapist_label": skill.get("therapist_label"),
                "primary_domain": skill.get("primary_domain"),
            }
        )
    return {
        "stages": rows,
        "sequence_pairs": sequence_scores,
        "n_sessions": len(cases),
        "n_with_coverage": sum(
            1 for case in cases if _reference_coverage(case.get("semantic") or {}) is not None
        ),
        "languages": dict(
            Counter(
                case.get("attempt", {}).get("session_language") or "unknown"
                for case in cases
            )
        ),
    }


def concordance_quadrants(cases: list[dict], domain: str, proxy_key: str | None = None) -> dict:
    """Median-split post confidence vs observed proxy; labels are concordance, not competence."""
    proxy_key = proxy_key or f"domain:{domain}"
    rows = []
    for case in cases:
        post = case["domains"][domain]["post"]
        proxy = observed_value(case, proxy_key)
        if post is None or proxy is None:
            continue
        rows.append({"post": float(post), "proxy": float(proxy), "change": case["domains"][domain]["change"]})
    if len(rows) < 2:
        return {"n": len(rows), "quadrants": {}, "post_median": None, "proxy_median": None}
    post_median = float(np.median([row["post"] for row in rows]))
    proxy_median = float(np.median([row["proxy"] for row in rows]))
    labels = []
    for row in rows:
        high_post = row["post"] >= post_median
        high_proxy = row["proxy"] >= proxy_median
        if high_post and high_proxy:
            labels.append("high_high")
        elif high_post and not high_proxy:
            labels.append("high_low")
        elif not high_post and high_proxy:
            labels.append("low_high")
        else:
            labels.append("low_low")
    counts = Counter(labels)
    n = len(labels)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_counts = {key: [] for key in ("high_high", "high_low", "low_high", "low_low")}
    for _ in range(1000):
        sample = rng.choice(labels, size=n, replace=True)
        sample_counts = Counter(sample)
        for key in bootstrap_counts:
            bootstrap_counts[key].append(sample_counts.get(key, 0) / n)
    quadrants = {}
    for key in ("high_high", "high_low", "low_high", "low_low"):
        count = counts.get(key, 0)
        proportion = count / n
        samples = bootstrap_counts[key]
        quadrants[key] = {
            "count": count,
            "proportion": round(proportion, 3),
            "ci_low": round(float(np.percentile(samples, 2.5)), 3),
            "ci_high": round(float(np.percentile(samples, 97.5)), 3),
        }
    change_vs_proxy = spearman_summary(
        cases,
        lambda case, d=domain: case["domains"][d]["change"],
        lambda case, key=proxy_key: observed_value(case, key),
    )
    post_vs_proxy = spearman_summary(
        cases,
        lambda case, d=domain: case["domains"][d]["post"],
        lambda case, key=proxy_key: observed_value(case, key),
    )
    return {
        "n": n,
        "domain": domain,
        "proxy_key": proxy_key,
        "post_median": round(post_median, 2),
        "proxy_median": round(proxy_median, 3),
        "quadrants": quadrants,
        "post_vs_proxy": post_vs_proxy,
        "change_vs_proxy": change_vs_proxy,
    }


def triangulation_profiles(
    closed_difficulty: list[dict],
    open_coverage: dict,
    survey_cases: list[dict],
) -> list[dict]:
    """Independent-cohort profiles aligned by stage/primary domain (not paired people)."""
    closed_by_stage = {row["stage"]: row for row in closed_difficulty}
    open_by_stage = {row["stage"]: row for row in open_coverage.get("stages", [])}
    rows = []
    for stage, skill in STAGE_SKILL_MAP.items():
        domain = skill["primary_domain"]
        survey = paired_summary(survey_cases, domain) if survey_cases else {"n": 0}
        closed = closed_by_stage.get(stage, {})
        open_row = open_by_stage.get(stage, {})
        rows.append(
            {
                "stage": stage,
                "primary_domain": domain,
                "therapist_label": skill["therapist_label"],
                "noa_label": skill["noa_label"],
                "closed_n": closed.get("n", 0),
                "closed_correct_rate": closed.get("correct_rate"),
                "closed_ci_low": closed.get("ci_low"),
                "closed_ci_high": closed.get("ci_high"),
                "open_n": open_row.get("n", 0),
                "open_median_coverage": open_row.get("median_coverage"),
                "survey_n": survey.get("n", 0),
                "survey_change_median": survey.get("change_median"),
                "survey_ci_low": survey.get("ci_low"),
                "survey_ci_high": survey.get("ci_high"),
            }
        )
    return rows


def insight_diagnostics(
    join_info: dict,
    closed_summary: dict,
    open_coverage: dict,
    cases: list[dict],
) -> dict:
    return {
        "survey_matched": join_info.get("matched", 0),
        "survey_unmatched": join_info.get("survey_without_current_attempt", 0),
        "multi_attempt_sessions": join_info.get("multi_attempt_sessions", 0),
        "closed_n": closed_summary.get("n", 0),
        "closed_with_stage_detail": closed_summary.get("with_stage_detail", 0),
        "open_semantic_n": open_coverage.get("n_with_coverage", 0),
        "open_case_n": open_coverage.get("n_sessions", len(cases)),
        "closed_languages": closed_summary.get("languages", {}),
        "open_languages": open_coverage.get("languages", {}),
    }