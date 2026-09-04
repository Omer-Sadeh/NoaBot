from survey_analysis import cronbach_alpha, domain_scores, score_instruments, select_attempts


def survey_row():
    return {
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
