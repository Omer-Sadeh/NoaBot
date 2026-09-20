"""Descriptive analysis dashboard for open-mode conversation attempts."""

from __future__ import annotations

from datetime import datetime, timezone

import altair as alt
import streamlit as st
from openai import OpenAI

import config
from analysis_export import build_analysis_export
from conversation_analysis import (
    ANALYZER_VERSION,
    cache_key,
    deterministic_metrics,
    semantic_metrics,
    semantic_texts,
)
from conversation_data import (
    clear_attempt_cache,
    filter_attempts,
    load_attempts,
    render_date_range_filter,
    setup_firestore,
)
from script_analysis_explanations import render_measure_explanation
from survey_data import SurveyDataError
from survey_screen import (
    clear_survey_cache,
    load_configured_surveys,
    render_survey_outcomes,
)


MIN_COMPARISON_SAMPLE = 10
ANALYSIS_STATUSES = ("success", "no success")


@st.cache_data(ttl=300, show_spinner=False)
def cached_analysis_export(
    attempts: list[dict],
    metrics_by_attempt: dict[str, dict],
    semantic_by_attempt: dict[str, dict | None],
    closed_attempts: list[dict],
    survey_rows: list[dict],
    filters: dict,
) -> bytes:
    return build_analysis_export(
        attempts=attempts,
        metrics_by_attempt=metrics_by_attempt,
        semantic_by_attempt=semantic_by_attempt,
        closed_attempts=closed_attempts,
        survey_rows=survey_rows,
        filters=filters,
        analyzer_version=ANALYZER_VERSION,
        embedding_model=config.EMBEDDING_MODEL,
    )


def analysis_attempt_key(attempt: dict) -> str:
    return f"{attempt.get('session_id')}/{attempt.get('doc_id')}"


def cache_document(database, key: str):
    return database.collection("analysis_cache").document(key)


def semantic_cache_key(attempt: dict, deterministic: dict) -> str:
    return cache_key(
        deterministic["source_hash"],
        attempt.get("session_language", "en"),
        config.EMBEDDING_MODEL,
    )


def cached_semantic_result(database, key: str) -> dict | None:
    document = cache_document(database, key).get()
    return document.to_dict().get("analysis") if document.exists else None


def cache_semantic_result(
    database, key: str, attempt: dict, deterministic: dict, analysis: dict
) -> None:
    cache_document(database, key).set(
        {
            "analysis": analysis,
            "source_hash": deterministic["source_hash"],
            "analysis_version": ANALYZER_VERSION,
            "embedding_model": config.EMBEDDING_MODEL,
            "language": attempt.get("session_language", "en"),
            "attempt": {
                "collection": config.get_variant(
                    st.session_state.get("variant")
                )["collection"],
                "session_id": attempt.get("session_id"),
                "doc_id": attempt.get("doc_id"),
            },
            "updated_at": datetime.now(timezone.utc),
        }
    )


def embed_texts(texts: list[str]) -> dict[str, list[float]]:
    client = OpenAI(api_key=st.secrets["openai_key"])
    response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return {
        texts[item.index]: item.embedding
        for item in response.data
        if item.index < len(texts)
    }


def analyze_missing(
    database, attempts: list[dict], metrics_by_attempt: dict[str, dict]
) -> int:
    missing = []
    for attempt in attempts:
        attempt_key = analysis_attempt_key(attempt)
        key = semantic_cache_key(attempt, metrics_by_attempt[attempt_key])
        if cached_semantic_result(database, key) is None:
            missing.append((attempt, key))
    if not missing:
        return 0
    texts = list(
        dict.fromkeys(
            text
            for attempt, _ in missing
            for text in semantic_texts(attempt)
        )
    )
    embeddings = embed_texts(texts)
    for attempt, key in missing:
        result = semantic_metrics(attempt, embeddings)
        cache_semantic_result(
            database, key, attempt, metrics_by_attempt[analysis_attempt_key(attempt)], result
        )
    return len(missing)


def distribution_rows(points: list[tuple[dict, str, float | int | None]]) -> list[dict]:
    return [
        {"series": series, "value": value}
        for _, series, value in points
        if value is not None
    ]


def render_distribution_chart(
    points: list[tuple[dict, str, float | int | None]], metric: str, unit: str
) -> None:
    rows = distribution_rows(points)
    if not rows:
        st.info(f"No recorded values are available for {metric.lower()}.")
        return
    has_multiple_series = len({row["series"] for row in rows}) > 1
    chart = alt.Chart(alt.Data(values=rows)).mark_bar().encode(
        x=alt.X(
            "value:Q",
            bin=alt.Bin(maxbins=10),
            title=f"{metric} ({unit})",
        ),
        y=alt.Y("count():Q", title="Sessions"),
        tooltip=[
            alt.Tooltip("series:N", title="Metric"),
            alt.Tooltip("count():Q", title="Sessions"),
        ],
    )
    if has_multiple_series:
        chart = chart.facet(
            column=alt.Column("series:N", title=None),
            columns=2,
        )
    st.altair_chart(chart, use_container_width=True)


def render_section_distribution(
    points: list[tuple[dict, str, float | int | None]], metric: str, unit: str
) -> None:
    rows = distribution_rows(points)
    if not rows:
        st.info("No section timing is available in this filtered sample.")
        return
    chart = alt.Chart(alt.Data(values=rows)).mark_bar().encode(
        x=alt.X("value:Q", bin=alt.Bin(maxbins=10), title=f"{metric} ({unit})"),
        y=alt.Y("count():Q", title="Sessions"),
        tooltip=[
            alt.Tooltip("series:N", title="Goal section"),
            alt.Tooltip("count():Q", title="Sessions"),
        ],
    ).facet(column=alt.Column("series:N", title=None), columns=3)
    st.altair_chart(chart, use_container_width=True)


def analysis_filters(date_range) -> dict:
    return {
        "mode": "open",
        "statuses": ANALYSIS_STATUSES,
        "date_range": date_range,
        "session_id": "",
        "language": "All",
    }


def render_overview(attempts: list[dict], metrics_by_attempt: dict[str, dict]) -> None:
    st.subheader("Overview")
    session_count = len({attempt["session_id"] for attempt in attempts})
    completed_count = sum(attempt.get("status") == "completed" for attempt in attempts)
    cleared_count = sum(
        bool(
            metrics_by_attempt[analysis_attempt_key(attempt)]["completion"][
                "is_llm_judged_success"
            ]
        )
        for attempt in attempts
    )
    columns = st.columns(4)
    columns[0].metric("Filtered attempts", len(attempts))
    columns[1].metric("Session IDs", session_count)
    columns[2].metric("Completed attempts", completed_count)
    columns[3].metric("LLM-judged guidelines cleared", cleared_count)
    render_measure_explanation("sample_overview")
    if len(attempts) < MIN_COMPARISON_SAMPLE:
        st.info(
            f"This filtered sample has fewer than {MIN_COMPARISON_SAMPLE} attempts. "
            "Use it for individual inspection, not aggregate comparison."
        )
    instrument_versions = {
        attempt.get("instrument", {}).get("app_version", "historical/unknown")
        for attempt in attempts
    }
    st.caption(
        "Measurement versions in this view: "
        + ", ".join(sorted(instrument_versions))
    )

    by_language = {}
    for attempt in attempts:
        language = attempt.get("session_language", "unknown")
        by_language[language] = by_language.get(language, 0) + 1
    st.bar_chart(
        [{"language": language, "attempts": count} for language, count in by_language.items()],
        x="language",
        y="attempts",
        use_container_width=True,
    )

def render_time_and_length(
    attempts: list[dict], metrics_by_attempt: dict[str, dict]
) -> None:
    st.subheader("Time and length")
    st.markdown("Conversation duration")
    render_measure_explanation("conversation_duration")
    render_distribution_chart(
        [
            (
                attempt,
                "Conversation duration",
                metrics_by_attempt[analysis_attempt_key(attempt)]["length"][
                    "duration_seconds"
                ]
                / 60
                if metrics_by_attempt[analysis_attempt_key(attempt)]["length"][
                    "duration_seconds"
                ]
                is not None
                else None,
            )
            for attempt in attempts
        ],
        "Conversation duration",
        "minutes",
    )

    st.markdown("Trainee turns")
    render_measure_explanation("trainee_turns")
    render_distribution_chart(
        [
            (
                attempt,
                "Trainee turns",
                metrics_by_attempt[analysis_attempt_key(attempt)]["length"]["user"][
                    "turn_count"
                ],
            )
            for attempt in attempts
        ],
        "Trainee turns",
        "turns",
    )

    section_time_points = []
    for attempt in attempts:
        timing = metrics_by_attempt[analysis_attempt_key(attempt)]["timing"]
        for section, seconds in enumerate(
            timing.get("section_durations_seconds") or [], start=1
        ):
            if seconds is not None:
                section_time_points.append(
                    (attempt, f"Goal section {section}", seconds)
                )
    st.markdown("Goal-section timing")
    render_measure_explanation("section_timing")
    render_section_distribution(
        section_time_points, "Time before advancing", "seconds"
    )


def render_alignment_and_complexity(
    attempts: list[dict],
    metrics_by_attempt: dict[str, dict],
    semantic_by_attempt: dict[str, dict | None],
) -> None:
    st.subheader("Reference, alignment, and complexity")
    st.caption(
        "How to read these charts: the horizontal axis groups similar values into "
        "ranges, and the vertical axis is the number of sessions in each range. "
        "They describe the filtered sample, not an individual trainee. Language is "
        "retained in the research export, but these charts pool languages when the "
        "filter includes more than one language."
    )

    coverage_points = []
    for attempt in attempts:
        semantic = semantic_by_attempt.get(analysis_attempt_key(attempt))
        if not semantic:
            continue
        for role, values in semantic["reference_trajectory"].items():
            coverage = [
                score
                for score in values["reference_move_coverage"]
                if score is not None
            ]
            if coverage:
                coverage_points.append(
                    (
                        attempt,
                        "Trainee" if role == "user" else "Noa",
                        sum(coverage) / len(coverage),
                    )
                )
    st.markdown("Reference-move coverage")
    render_measure_explanation("reference_coverage")
    if coverage_points:
        render_distribution_chart(
            coverage_points, "Reference-move coverage", "0–1"
        )
    else:
        st.info("Run semantic analysis to populate this trend.")

    question_rate_points = []
    reflection_rate_points = []
    for attempt in attempts:
        register = metrics_by_attempt[analysis_attempt_key(attempt)][
            "therapeutic_register"
        ]
        question_rate_points.append(
            (attempt, "Questions per trainee turn", register["question_rate"])
        )
        reflection_rate_points.append(
            (
                attempt,
                "Reflection markers",
                register["reflection_marker_rate"] * 100
                if register["reflection_marker_rate"] is not None
                else None,
            )
        )
    st.markdown("Therapeutic-register fidelity")
    render_measure_explanation("question_rate")
    render_measure_explanation("reflection_marker_rate")
    render_distribution_chart(
        question_rate_points,
        "Question rate",
        "questions per trainee turn",
    )
    render_distribution_chart(
        reflection_rate_points,
        "Reflection-marker rate",
        "markers per 100 words",
    )

    lexical_points = []
    sentence_length_points = []
    for attempt in attempts:
        complexity = metrics_by_attempt[analysis_attempt_key(attempt)]["complexity"]
        lexical_points.extend(
            [
                (attempt, "Lexical diversity", complexity["mattr_50"]),
                (attempt, "Repeated-token share", complexity["repetition_rate"]),
            ]
        )
        sentence_length_points.append(
            (attempt, "Words per sentence", complexity["median_words_per_sentence"])
        )
    st.markdown("Conversation complexity")
    render_measure_explanation("lexical_diversity")
    render_measure_explanation("repetition_rate")
    render_measure_explanation("sentence_length")
    render_distribution_chart(
        lexical_points, "Lexical signal", "proportion (0–1)"
    )
    render_distribution_chart(
        sentence_length_points,
        "Sentence length",
        "words per sentence",
    )

    alignment_points = []
    for attempt in attempts:
        alignment = metrics_by_attempt[analysis_attempt_key(attempt)][
            "style_alignment"
        ]
        if alignment["available"]:
            alignment_points.extend(
                [
                    (
                        attempt,
                        "Trainee to Noa",
                        alignment["user_to_noa_mean_alignment"],
                    ),
                    (
                        attempt,
                        "Noa to trainee",
                        alignment["noa_to_user_mean_alignment"],
                    ),
                    (
                        attempt,
                        "Shuffled within-attempt baseline",
                        alignment["within_attempt_shuffled_null"],
                    ),
                ]
            )
    st.markdown("Exploratory user-to-Noa style alignment")
    render_measure_explanation("style_alignment")
    render_distribution_chart(
        alignment_points, "Style alignment", "0–1"
    )


def render_analysis_screen() -> None:
    st.title("Open Conversation Analysis")
    collection_name = config.get_variant(st.session_state.get("variant"))["collection"]
    if st.sidebar.button("Refresh saved attempts", key="analysis_refresh"):
        clear_attempt_cache()
        clear_survey_cache()
        st.rerun()
    attempts = load_attempts(collection_name)
    st.sidebar.header("Filters")
    filters = analysis_filters(
        render_date_range_filter(attempts, key_prefix="analysis")
    )
    filtered = filter_attempts(attempts, filters)
    if not filtered:
        st.info("No open conversation attempts match the current filters.")
        if st.button("Back to Menu", key="analysis_back_empty"):
            st.session_state.pre_done = False
            st.rerun()
        return

    metrics_by_attempt = {
        analysis_attempt_key(attempt): deterministic_metrics(attempt) for attempt in filtered
    }
    database = setup_firestore()
    semantic_by_attempt = {
        analysis_attempt_key(attempt): cached_semantic_result(
            database,
            semantic_cache_key(
                attempt, metrics_by_attempt[analysis_attempt_key(attempt)]
            ),
        )
        for attempt in filtered
    }
    analyzed_count = sum(result is not None for result in semantic_by_attempt.values())
    st.sidebar.caption(
        f"Semantic analysis cached for {analyzed_count}/{len(filtered)} filtered attempts."
    )
    if st.sidebar.button("Analyze missing or changed sessions", type="primary"):
        try:
            with st.spinner("Creating cached reference metrics..."):
                analyzed = analyze_missing(database, filtered, metrics_by_attempt)
            st.success(f"Analyzed {analyzed} attempt(s).")
            st.rerun()
        except Exception as error:
            st.error(f"Semantic analysis could not complete: {error}")

    closed_filters = {**filters, "mode": "closed"}
    closed_filtered = filter_attempts(attempts, closed_filters)
    try:
        survey_rows, _ = load_configured_surveys()
    except SurveyDataError:
        survey_rows = []

    st.warning(
        "The research export contains internal session identifiers and raw conversation "
        "text. Treat the downloaded ZIP as identifiable, sensitive research data."
    )
    export_bytes = cached_analysis_export(
        filtered,
        metrics_by_attempt,
        semantic_by_attempt,
        closed_filtered,
        survey_rows,
        filters,
    )
    st.download_button(
        "Download research CSV bundle",
        data=export_bytes,
        file_name=f"noabot_analysis_{datetime.now(timezone.utc):%Y%m%d}.zip",
        mime="application/zip",
        help="Exports the current filtered sample. It uses cached semantic results only.",
    )

    overview_tab, timing_tab, alignment_tab, survey_tab = st.tabs(
        ("Overview", "Time and length", "Alignment and complexity", "Survey outcomes")
    )
    with overview_tab:
        render_overview(filtered, metrics_by_attempt)
    with timing_tab:
        render_time_and_length(filtered, metrics_by_attempt)
    with alignment_tab:
        render_alignment_and_complexity(
            filtered, metrics_by_attempt, semantic_by_attempt
        )
    with survey_tab:
        render_survey_outcomes(
            filtered,
            metrics_by_attempt,
            semantic_by_attempt,
            closed_filtered,
        )

    if st.button("Back to Menu", key="analysis_back"):
        st.session_state.pre_done = False
        st.rerun()
