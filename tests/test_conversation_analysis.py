from conversation_analysis import (
    cache_key,
    deterministic_metrics,
    language_reference,
    semantic_metrics,
)


def sample_attempt():
    return {
        "doc_id": "attempt",
        "session_language": "en",
        "session_duration_seconds": 30,
        "section_durations_seconds": [12],
        "section_time_semantics": "per_section_v2",
        "section_user_turns": [1],
        "section_transition_events": [
            {"section": 1, "completed": True, "elapsed_seconds": 12, "user_turns": 1}
        ],
        "completed_guidelines": 1,
        "completed_criteria_total": 6,
        "is_successful": False,
        "turns": [
            {"role": "noa", "content": "I feel upset.", "elapsed_seconds": None},
            {
                "role": "user",
                "content": "It sounds upsetting. How did you feel?",
                "elapsed_seconds": 4,
            },
            {"role": "noa", "content": "I avoided saying anything.", "elapsed_seconds": 8},
            {
                "role": "user",
                "content": "What would you want to say to Dana?",
                "elapsed_seconds": 12,
            },
            {"role": "noa", "content": "Maybe I could explain.", "elapsed_seconds": 16},
            {
                "role": "user",
                "content": "How could you say that directly?",
                "elapsed_seconds": 20,
            },
        ],
    }


def test_deterministic_metrics_reports_length_timing_and_style():
    metrics = deterministic_metrics(sample_attempt())

    assert metrics["length"]["user"]["turn_count"] == 3
    assert metrics["timing"]["completed_sections"] == 1
    assert metrics["style_alignment"]["available"] is True
    assert metrics["completion"]["guidelines_total"] == 6


def test_semantic_metrics_scores_identical_reference_moves_as_full_coverage():
    reference = language_reference("en")
    attempt = {
        "doc_id": "reference",
        "session_language": "en",
        "turns": [
            *({"role": "noa", "content": text} for text in reference["noa"]),
            *({"role": "user", "content": text} for text in reference["user"]),
        ],
    }
    all_texts = reference["noa"] + reference["user"]
    embeddings = {
        text: [float(index + 1), 1.0] for index, text in enumerate(all_texts)
    }

    metrics = semantic_metrics(attempt, embeddings)

    assert metrics["reference_trajectory"]["user"]["reference_move_coverage"] == [
        1.0
    ] * len(reference["user"])
    assert metrics["reference_trajectory"]["noa"]["median_nearest_reference_distance"] == 0.0


def test_cache_key_changes_when_model_or_source_changes():
    first = cache_key("source-a", "en", "model-a")

    assert first != cache_key("source-b", "en", "model-a")
    assert first != cache_key("source-a", "en", "model-b")


def test_semantic_metrics_excludes_fixed_initial_noa_prompt_from_generated_turns():
    attempt = {
        "doc_id": "initial",
        "session_language": "en",
        "turns": [
            {"role": "noa", "content": "Fixed opener"},
            {"role": "noa", "content": "Generated reply"},
        ],
    }
    reference = language_reference("en")
    embeddings = {
        "Fixed opener": [1.0, 0.0],
        "Generated reply": [0.0, 1.0],
        **{text: [0.0, 1.0] for text in reference["noa"] + reference["user"]},
    }

    metrics = semantic_metrics(attempt, embeddings)

    assert metrics["reference_trajectory"]["noa"]["turn_count"] == 1


def test_semantic_metrics_reports_domain_reference_coverage():
    attempt = sample_attempt()
    reference = language_reference("en")
    from conversation_analysis import DOMAIN_ANCHORS, semantic_texts

    texts = semantic_texts(attempt)
    embeddings = {text: [1.0, 0.0] for text in texts}
    metrics = semantic_metrics(attempt, embeddings)

    assert set(metrics["domain_reference_coverage"]) == set(DOMAIN_ANCHORS["en"])
    assert metrics["domain_reference_coverage"]["calm_deescalation"]["available"] is True
