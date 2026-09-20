from survey_analysis import (
    annotate_closed_stages,
    closed_cohort_summary,
    closed_distractor_patterns,
    closed_stage_difficulty,
    concordance_quadrants,
    cronbach_alpha,
    domain_scores,
    ground_truth_pathway,
    insight_diagnostics,
    observed_value,
    open_stage_coverage,
    score_instruments,
    select_attempts,
    spearman_summary,
    triangulation_profiles,
)


def survey_row(**overrides):
    row = {
        "session_id": "s1",
        "S1_pre_calm_tense_conversation": 60, "S2_post_calm_tense_conversation": 70,
        "S1_pre_reflect_viewpoint": 60, "S2_post_reflect_viewpoint": 70,
        "S1_pre_constructive_next_step": 60, "S2_post_constructive_next_step": 70,
        "S1_pre_guiding_questions": 60, "S2_post_guiding_questions": 70,
        **{name: 3 for name in (
            "S2_UES_lost_myself", "S2_UES_time_slipped_away", "S2_UES_absorbed",
            "S2_UES_frustrated_raw", "S2_UES_confusing_raw", "S2_UES_taxing_raw",
            "S2_UES_worthwhile", "S2_UES_rewarding", "S2_UES_interested",
            "S2_UEQ_obstructive_supportive", "S2_UEQ_complicated_easy",
            "S2_UEQ_inefficient_efficient", "S2_UEQ_confusing_clear",
            "S2_UEQ_boring_exciting", "S2_UEQ_not_interesting_interesting",
            "S2_UEQ_conventional_inventive", "S2_UEQ_usual_leading_edge",
            "S2_agent_fake_natural", "S2_agent_machinelike_humanlike",
            "S2_agent_unconscious_conscious", "S2_agent_artificial_lifelike",
            "S2_agent_rigid_elegant", "S2_agent_dislike_like", "S2_agent_unfriendly_friendly",
            "S2_agent_unkind_kind", "S2_agent_unpleasant_pleasant", "S2_agent_awful_nice",
        )},
        "S2_reuse_intention": 4,
    }
    row.update(overrides)
    return row


def test_scores_reverse_usability_and_does_not_create_ues_total():
    scores = score_instruments(survey_row())

    assert scores["Perceived usability"] == 3
    assert scores["Overall UX"] == -1
    assert "UES total" not in scores


def test_domain_scores_return_paired_change():
    assert domain_scores(survey_row())["calm_deescalation"]["change"] == 10


def test_select_attempts_prefers_completed_then_latest_attempt():
    selected, counts = select_attempts([
        {"session_id": "a", "mode": "open", "status": "ongoing", "timestamp": "2026-01-02", "doc_id": "old"},
        {"session_id": "a", "mode": "open", "status": "completed", "timestamp": "2026-01-01", "doc_id": "complete"},
    ])

    assert selected["a"]["doc_id"] == "complete"
    assert counts["a"] == 2


def test_cronbach_alpha_returns_none_for_degenerate_scale():
    rows = [{"one": 3, "two": 3}, {"one": 3, "two": 3}, {"one": 3, "two": 3}]

    assert cronbach_alpha(rows, ("one", "two")) is None


def test_observed_value_preserves_missing_duration_and_completion():
    case = {
        "metrics": {
            "length": {"duration_seconds": None},
            "completion": {"guidelines_cleared": None, "guidelines_total": 5},
        },
        "semantic": None,
        "attempt": {},
    }

    assert observed_value(case, "duration_minutes") is None
    assert observed_value(case, "guideline_completion") is None


def test_spearman_summary_reports_constant_input_as_unavailable():
    cases = [{"x": 1, "y": value} for value in (1, 2, 3)]

    result = spearman_summary(
        cases, lambda case: case["x"], lambda case: case["y"]
    )

    assert result == {"n": 3, "rho": None, "p_value": None}


def test_ground_truth_pathway_covers_five_stages():
    pathway = ground_truth_pathway()

    assert [stage["stage"] for stage in pathway] == [1, 2, 3, 4, 5]
    assert pathway[0]["primary_domain"] == "calm_deescalation"
    assert "guiding_questions" in pathway[0]["survey_domains"]


def closed_attempt(session_id, *, correct_answers=3, stages=None, language="en"):
    return {
        "session_id": session_id,
        "doc_id": f"final_{session_id}",
        "mode": "closed",
        "status": "completed",
        "session_language": language,
        "total_questions": 5,
        "correct_answers": correct_answers,
        "closed_stage_results": stages or [],
        "data": "",
    }


def test_annotate_closed_stages_classifies_distractors():
    attempt = closed_attempt(
        "c1",
        stages=[
            {
                "stage": 1,
                "selected_answer": (
                    "Sounds rough, especially at the last minute... What did you end up doing?"
                ),
                "is_correct": False,
            },
            {
                "stage": 2,
                "selected_answer": (
                    "Suppose you did say something to her, what would you want to say? "
                    "What's your goal in talking to her?"
                ),
                "is_correct": True,
            },
        ],
    )

    annotated = annotate_closed_stages(attempt)

    assert annotated[0]["option_type"] == "incorrect_1"
    assert annotated[0]["failure_mode"] == "interest_without_skill_focus"
    assert annotated[1]["option_type"] == "correct"
    assert annotated[1]["failure_mode"] is None


def test_closed_cohort_summary_and_stage_difficulty():
    attempts = [
        closed_attempt(
            "a",
            correct_answers=5,
            stages=[
                {"stage": 1, "selected_answer": "x", "is_correct": True},
                {"stage": 2, "selected_answer": "y", "is_correct": True},
            ],
        ),
        closed_attempt(
            "b",
            correct_answers=2,
            stages=[
                {"stage": 1, "selected_answer": "x", "is_correct": False},
                {"stage": 2, "selected_answer": "y", "is_correct": True},
            ],
        ),
        closed_attempt("ongoing", correct_answers=1, stages=[]),
    ]
    attempts[-1]["status"] = "ongoing"

    summary = closed_cohort_summary(attempts)
    difficulty = closed_stage_difficulty(attempts)

    assert summary["n"] == 2
    assert summary["perfect_rate"] == 0.5
    assert summary["median_accuracy"] == 0.7
    assert difficulty[0]["stage"] == 1
    assert difficulty[0]["n"] == 2
    assert difficulty[0]["correct_rate"] == 0.5


def test_closed_distractor_patterns_group_errors():
    attempts = [
        closed_attempt(
            "a",
            stages=[
                {
                    "stage": 1,
                    "selected_answer": (
                        "Sounds rough, especially at the last minute... What did you end up doing?"
                    ),
                    "is_correct": False,
                },
                {
                    "stage": 1,
                    "selected_answer": (
                        "Let's try to see this situation from Dana's perspective. "
                        "Maybe she didn't mean to hurt you?"
                    ),
                    "is_correct": False,
                },
            ],
        )
    ]
    # Force two sessions with stage-1 errors of different types.
    attempts.append(
        closed_attempt(
            "b",
            stages=[
                {
                    "stage": 1,
                    "selected_answer": (
                        "Sounds rough, especially at the last minute... What did you end up doing?"
                    ),
                    "is_correct": False,
                }
            ],
        )
    )

    patterns = closed_distractor_patterns(attempts)

    assert patterns
    interest = next(
        row for row in patterns if row["failure_mode"] == "interest_without_skill_focus"
    )
    assert interest["count"] == 2
    assert interest["stage"] == 1


def make_case(session_id, *, post=70, coverage=0.4, stage_coverage=None, sequence=0.5):
    domains = domain_scores(
        survey_row(
            session_id=session_id,
            S2_post_calm_tense_conversation=post,
            S2_post_reflect_viewpoint=post,
            S2_post_constructive_next_step=post,
            S2_post_guiding_questions=post,
        )
    )
    stage_coverage = stage_coverage or [0.2, 0.3, 0.4, 0.5, 0.6]
    return {
        "survey": survey_row(session_id=session_id),
        "attempt": {
            "session_id": session_id,
            "doc_id": "final",
            "mode": "open",
            "session_language": "en",
        },
        "metrics": {},
        "semantic": {
            "reference_trajectory": {
                "user": {
                    "reference_move_coverage": stage_coverage,
                    "exploratory_monotonic_alignment": {"mean_similarity": sequence},
                }
            },
            "domain_reference_coverage": {
                domain: {"coverage": coverage} for domain in domains
            },
        },
        "instruments": score_instruments(survey_row()),
        "domains": domains,
        "candidate_count": 1,
    }


def test_open_stage_coverage_aggregates_by_stage():
    cases = [
        make_case("a", stage_coverage=[0.1, 0.2, 0.3, 0.4, 0.5], sequence=0.4),
        make_case("b", stage_coverage=[0.3, 0.4, 0.5, 0.6, 0.7], sequence=0.8),
    ]

    result = open_stage_coverage(cases)

    assert result["n_with_coverage"] == 2
    assert result["stages"][0]["stage"] == 1
    assert result["stages"][0]["median_coverage"] == 0.2
    assert len(result["sequence_pairs"]) == 2


def test_concordance_quadrants_split_on_medians():
    cases = [
        make_case("a", post=40, coverage=0.2),
        make_case("b", post=40, coverage=0.8),
        make_case("c", post=90, coverage=0.2),
        make_case("d", post=90, coverage=0.8),
    ]

    result = concordance_quadrants(cases, "calm_deescalation")

    assert result["n"] == 4
    assert result["quadrants"]["high_high"]["count"] == 1
    assert result["quadrants"]["high_low"]["count"] == 1
    assert result["quadrants"]["low_high"]["count"] == 1
    assert result["quadrants"]["low_low"]["count"] == 1


def test_concordance_quadrants_handles_sparse_data():
    assert concordance_quadrants([make_case("only")], "calm_deescalation")["n"] == 1


def test_triangulation_profiles_keep_independent_samples():
    cases = [make_case("a"), make_case("b")]
    closed = [
        closed_attempt(
            "closed-a",
            correct_answers=4,
            stages=[{"stage": 1, "selected_answer": "x", "is_correct": True}],
        )
    ]
    coverage = open_stage_coverage(cases)
    difficulty = closed_stage_difficulty(closed)
    profiles = triangulation_profiles(difficulty, coverage, cases)

    assert profiles[0]["stage"] == 1
    assert profiles[0]["closed_n"] == 1
    assert profiles[0]["open_n"] == 2
    assert profiles[0]["survey_n"] == 2
    assert profiles[0]["closed_correct_rate"] == 1.0


def test_insight_diagnostics_report_sample_gaps():
    diagnostics = insight_diagnostics(
        {"matched": 3, "survey_without_current_attempt": 1, "multi_attempt_sessions": 2},
        {"n": 4, "with_stage_detail": 3, "languages": {"en": 4}},
        {"n_with_coverage": 2, "n_sessions": 3, "languages": {"en": 3}},
        [],
    )

    assert diagnostics["survey_matched"] == 3
    assert diagnostics["closed_n"] == 4
    assert diagnostics["open_semantic_n"] == 2
