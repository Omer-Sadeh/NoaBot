"""Streamlit presentation for survey outcomes and observed session measures."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import streamlit as st

import config
from script_analysis_explanations import render_measure_explanation
from survey_analysis import (
    closed_cohort_summary,
    closed_distractor_patterns,
    closed_stage_difficulty,
    concordance_quadrants,
    ground_truth_pathway,
    insight_diagnostics,
    joined_cases,
    observed_value,
    open_stage_coverage,
    paired_summary,
    scale_reliability,
    spearman_summary,
    triangulation_profiles,
)
from survey_data import DOMAIN_COLUMNS, SurveyDataError, load_survey_rows, workbook_hash


QUADRANT_LABELS = {
    "high_high": "High confidence · high proxy",
    "high_low": "High confidence · low proxy",
    "low_high": "Low confidence · high proxy",
    "low_low": "Low confidence · low proxy",
}


@st.cache_data(ttl=300, show_spinner=False)
def _cached_rows(path: str, source_hash: str):
    del source_hash
    return load_survey_rows(path)


def clear_survey_cache() -> None:
    _cached_rows.clear()


def load_configured_surveys() -> tuple[list[dict], dict]:
    path = Path(config.SURVEY_WORKBOOK_PATH)
    if not path.exists():
        raise SurveyDataError(f"Survey workbook is missing: {path}")
    return _cached_rows(str(path), workbook_hash(path))


def _chart(rows: list[dict], x: str, y: str, title: str) -> None:
    if not rows:
        st.info("No usable matched values are available for this comparison.")
        return
    st.altair_chart(
        alt.Chart(alt.Data(values=rows)).mark_circle(size=70, opacity=0.7).encode(
            x=alt.X(f"{x}:Q", title=x.replace("_", " ").title()),
            y=alt.Y(f"{y}:Q", title=y.replace("_", " ").title()),
            tooltip=[alt.Tooltip(f"{x}:Q", format=".2f"), alt.Tooltip(f"{y}:Q", format=".2f")],
        ).properties(title=title),
        use_container_width=True,
    )


def _distribution(cases: list[dict], score_name: str) -> None:
    values = [{"value": case["instruments"].get(score_name)} for case in cases if case["instruments"].get(score_name) is not None]
    if values:
        st.altair_chart(alt.Chart(alt.Data(values=values)).mark_bar().encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=8), title=score_name), y=alt.Y("count():Q", title="Sessions")
        ), use_container_width=True)


def _paired_rows(cases: list[dict], domain: str) -> list[dict]:
    return [{"pre": case["domains"][domain]["pre"], "post": case["domains"][domain]["post"]}
            for case in cases if case["domains"][domain]["change"] is not None]


def _change_plot(rows: list[dict], title: str) -> alt.Chart:
    """Show each within-person difference directly, without participant identifiers."""
    points = [
        {
            "change": row["post"] - row["pre"],
            "jitter": ((index * 17) % 11 - 5) / 10,
            "direction": "Increased" if row["post"] > row["pre"]
            else "Decreased" if row["post"] < row["pre"] else "Unchanged",
        }
        for index, row in enumerate(rows)
    ]
    data = alt.Data(values=points)
    zero = alt.Chart(alt.Data(values=[{"change": 0}])).mark_rule(
        color="#8c8c8c", strokeDash=[4, 4]
    ).encode(x="change:Q")
    dots = alt.Chart(data).mark_circle(size=85, opacity=0.82).encode(
        x=alt.X("change:Q", title="Change in confidence (post − pre)"),
        y=alt.Y("jitter:Q", axis=None, scale=alt.Scale(domain=[-0.65, 0.65])),
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(
                domain=["Decreased", "Unchanged", "Increased"],
                range=["#d95f02", "#757575", "#1b9e77"],
            ),
            title=None,
        ),
        tooltip=[
            alt.Tooltip("change:Q", title="Change", format="+.1f"),
            alt.Tooltip("direction:N", title="Direction"),
        ],
    )
    return (zero + dots).properties(title=title, height=130)


def _layer_banner() -> None:
    st.caption(
        "Evidence layers: **authored ground truth** · **closed-cohort recognition** · "
        "**open-session semantic proximity** · **matched self-report**. "
        "Open and closed cohorts are independent; scales are not interchangeable."
    )


def _render_self_report(cases: list[dict]) -> None:
    st.subheader("Self-reported change")
    render_measure_explanation("self_report_change")
    for domain in DOMAIN_COLUMNS:
        rows = _paired_rows(cases, domain)
        summary = paired_summary(cases, domain)
        st.markdown(f"**{domain.replace('_', ' ').title()}** — paired n={summary['n']}")
        if rows:
            st.altair_chart(
                _change_plot(rows, "Individual confidence changes"),
                use_container_width=True,
            )
        if summary["n"]:
            st.caption(f"Median change: {summary['change_median']:+.1f}; 95% bootstrap CI {summary['ci_low']:+.1f} to {summary['ci_high']:+.1f}" + (f"; Wilcoxon p={summary['p_value']}" if summary["p_value"] is not None else ""))


def _render_domain_comparisons(cases: list[dict]) -> None:
    st.subheader("Self-report versus observed session measures")
    render_measure_explanation("domain_reference_coverage")
    render_measure_explanation("spearman_association")
    for domain in DOMAIN_COLUMNS:
        rows = []
        for case in cases:
            post = case["domains"][domain]["post"]
            coverage = observed_value(case, f"domain:{domain}")
            if post is not None and coverage is not None:
                rows.append({"post_confidence": post, "domain_reference_coverage": coverage})
        result = spearman_summary(cases, lambda case, d=domain: case["domains"][d]["post"], lambda case, d=domain: observed_value(case, f"domain:{d}"))
        st.markdown(f"**{domain.replace('_', ' ').title()}** — n={result['n']}, Spearman ρ={result['rho']}")
        _chart(rows, "post_confidence", "domain_reference_coverage", "Post confidence and observed proxy")
        if rows:
            midpoint_x = float(sorted(row["post_confidence"] for row in rows)[len(rows) // 2])
            midpoint_y = float(sorted(row["domain_reference_coverage"] for row in rows)[len(rows) // 2])
            st.caption(f"Relative concordance uses sample medians: self-report {midpoint_x:.1f}; observed proxy {midpoint_y:.2f}.")


def _render_experience(cases: list[dict]) -> None:
    st.subheader("Experience versus observed behavior")
    render_measure_explanation("experience_scales")
    for score in ("Focused attention", "Perceived usability", "Reward", "Pragmatic quality", "Hedonic quality", "Overall UX", "Anthropomorphism", "Likeability", "Reuse intention"):
        st.markdown(f"**{score}**")
        _distribution(cases, score)
    reliability = scale_reliability(cases)
    render_measure_explanation("scale_reliability")
    st.caption("Internal consistency (Cronbach’s α; complete cases only; exploratory at this sample size): " + ", ".join(
        f"{name}={value if value is not None else 'unavailable'}" for name, value in reliability.items()
    ))
    definitions = (
        ("Focused attention", "duration_minutes"), ("Focused attention", "trainee_turns"),
        ("Perceived usability", "tips_shown"), ("Reward", "guideline_completion"),
        ("Reuse intention", "guideline_completion"), ("Anthropomorphism", "style_alignment"),
    )
    st.markdown("**Curated associations**")
    render_measure_explanation("spearman_association")
    for score, observed in definitions:
        result = spearman_summary(cases, lambda case, s=score: case["instruments"].get(s), lambda case, o=observed: observed_value(case, o))
        st.write(f"{score} × {observed.replace('_', ' ')}: n={result['n']}, ρ={result['rho']}" + (f", 95% CI {result.get('ci_low')} to {result.get('ci_high')}" if result.get("ci_low") is not None else ""))


def _render_ground_truth() -> None:
    st.subheader("Ground-truth pathway")
    st.caption(
        "When answered correctly, the closed script is the authored ground truth for how the "
        "therapist should act and how Noa should respond. Survey domain `guiding_questions` is "
        "cross-cutting and not a one-to-one stage mapping."
    )
    for stage in ground_truth_pathway():
        domains = ", ".join(domain.replace("_", " ") for domain in stage["survey_domains"])
        st.markdown(
            f"**Stage {stage['stage']}** — therapist: {stage['therapist_label']}; "
            f"Noa: {stage['noa_label']}; primary survey domain: "
            f"`{stage['primary_domain'].replace('_', ' ')}` (also: {domains})."
        )


def _render_closed_cohort(closed_attempts: list[dict]) -> None:
    st.subheader("Closed cohort recognition")
    render_measure_explanation("closed_accuracy")
    summary = closed_cohort_summary(closed_attempts)
    if not summary["n"]:
        st.info("No completed closed-script attempts match the current filters.")
        return
    columns = st.columns(4)
    columns[0].metric("Closed sessions", summary["n"])
    columns[1].metric("Median accuracy", f"{summary['median_accuracy']:.0%}" if summary["median_accuracy"] is not None else "—")
    columns[2].metric("Mean accuracy", f"{summary['mean_accuracy']:.0%}" if summary["mean_accuracy"] is not None else "—")
    columns[3].metric("Perfect score rate", f"{summary['perfect_rate']:.0%}" if summary["perfect_rate"] is not None else "—")
    if summary.get("accuracy_values"):
        st.altair_chart(
            alt.Chart(alt.Data(values=[{"accuracy": value} for value in summary["accuracy_values"]]))
            .mark_bar()
            .encode(
                x=alt.X("accuracy:Q", bin=alt.Bin(maxbins=6), title="Closed accuracy"),
                y=alt.Y("count():Q", title="Sessions"),
            )
            .properties(title="Closed-script accuracy distribution"),
            use_container_width=True,
        )
    difficulty = closed_stage_difficulty(closed_attempts)
    if difficulty:
        st.markdown("**Stage difficulty**")
        render_measure_explanation("closed_stage_difficulty")
        chart_rows = [
            {
                "stage": f"Stage {row['stage']}",
                "correct_rate": row["correct_rate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "n": row["n"],
                "label": row.get("therapist_label") or "",
            }
            for row in difficulty
            if row.get("correct_rate") is not None
        ]
        if chart_rows:
            base = alt.Chart(alt.Data(values=chart_rows))
            points = base.mark_circle(size=90).encode(
                x=alt.X("correct_rate:Q", title="P(correct)", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("stage:N", title=None, sort=[row["stage"] for row in chart_rows]),
                tooltip=["stage:N", "label:N", "n:Q", alt.Tooltip("correct_rate:Q", format=".0%")],
            )
            error = base.mark_rule().encode(
                x="ci_low:Q",
                x2="ci_high:Q",
                y=alt.Y("stage:N", sort=[row["stage"] for row in chart_rows]),
            )
            st.altair_chart((error + points).properties(height=180), use_container_width=True)
            for row in difficulty:
                st.caption(
                    f"Stage {row['stage']} ({row.get('therapist_label')}): "
                    f"n={row['n']}, P(correct)={row['correct_rate']}"
                    + (
                        f", 95% CI {row['ci_low']}–{row['ci_high']}"
                        if row.get("ci_low") is not None
                        else ""
                    )
                )
    patterns = closed_distractor_patterns(closed_attempts)
    if patterns:
        st.markdown("**Common distractor patterns among incorrect choices**")
        render_measure_explanation("closed_distractor_pattern")
        for row in patterns[:8]:
            st.write(
                f"Stage {row['stage']}: {row['failure_mode_label']} — "
                f"{row['count']}/{row['stage_error_n']} errors "
                f"({row['share_of_stage_errors']:.0%})"
            )
    elif summary["with_stage_detail"] == 0:
        st.info("Stage-level Correct: lines were not recoverable for these closed attempts.")


def _render_open_vs_target(cases: list[dict]) -> None:
    st.subheader("Open sessions versus authored target")
    render_measure_explanation("reference_coverage")
    coverage = open_stage_coverage(cases)
    if not coverage["stages"]:
        st.info("No cached semantic reference coverage is available for matched open sessions.")
        return
    st.caption(
        f"Sessions with coverage: {coverage['n_with_coverage']}/{coverage['n_sessions']}."
    )
    chart_rows = [
        {
            "stage": f"Stage {row['stage']}",
            "median_coverage": row["median_coverage"],
            "label": row.get("therapist_label") or "",
            "n": row["n"],
        }
        for row in coverage["stages"]
    ]
    st.altair_chart(
        alt.Chart(alt.Data(values=chart_rows))
        .mark_bar()
        .encode(
            x=alt.X("median_coverage:Q", title="Median reference coverage", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("stage:N", title=None, sort=[row["stage"] for row in chart_rows]),
            tooltip=["stage:N", "label:N", "n:Q", alt.Tooltip("median_coverage:Q", format=".2f")],
        )
        .properties(title="Open proximity to authored therapist moves", height=180),
        use_container_width=True,
    )
    if coverage["sequence_pairs"]:
        st.markdown("**Coverage versus sequence fidelity**")
        render_measure_explanation("sequence_fidelity")
        _chart(
            coverage["sequence_pairs"],
            "coverage",
            "sequence_fidelity",
            "Mean coverage and sequence fidelity",
        )


def _render_survey_calibration(cases: list[dict]) -> None:
    st.subheader("Survey calibration against open proxies")
    render_measure_explanation("calibration_quadrants")
    for domain in DOMAIN_COLUMNS:
        result = concordance_quadrants(cases, domain)
        st.markdown(
            f"**{domain.replace('_', ' ').title()}** — n={result['n']}; "
            f"post median={result['post_median']}; proxy median={result['proxy_median']}"
        )
        if result["n"] < 2:
            st.info("Not enough matched values for concordance quadrants.")
            continue
        post_rho = result["post_vs_proxy"]
        change_rho = result["change_vs_proxy"]
        st.write(
            f"Post × proxy: ρ={post_rho.get('rho')} (n={post_rho.get('n')}); "
            f"change × proxy: ρ={change_rho.get('rho')} (n={change_rho.get('n')})"
        )
        quad_rows = [
            {
                "quadrant": QUADRANT_LABELS[key],
                "count": values["count"],
                "proportion": values["proportion"],
            }
            for key, values in result["quadrants"].items()
        ]
        st.altair_chart(
            alt.Chart(alt.Data(values=quad_rows))
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Sessions"),
                y=alt.Y("quadrant:N", title=None, sort=list(QUADRANT_LABELS.values())),
                tooltip=["quadrant:N", "count:Q", alt.Tooltip("proportion:Q", format=".0%")],
            )
            .properties(height=140),
            use_container_width=True,
        )
        for key, values in result["quadrants"].items():
            st.caption(
                f"{QUADRANT_LABELS[key]}: {values['count']} "
                f"({values['proportion']:.0%}; 95% CI {values['ci_low']:.0%}–{values['ci_high']:.0%})"
            )


def _render_triangulation(cases: list[dict], closed_attempts: list[dict]) -> None:
    st.subheader("Cross-source triangulation")
    render_measure_explanation("triangulation")
    st.warning(
        "These panels align constructs by stage/domain mapping only. "
        "Closed and open/survey cohorts are independent samples with different scales. "
        "Do not read this as person-level transfer."
    )
    closed_difficulty = closed_stage_difficulty(closed_attempts)
    coverage = open_stage_coverage(cases)
    profiles = triangulation_profiles(closed_difficulty, coverage, cases)
    if not any(row["closed_n"] or row["open_n"] or row["survey_n"] for row in profiles):
        st.info("Not enough closed, open, or survey data to triangulate.")
        return
    for row in profiles:
        st.markdown(f"**Stage {row['stage']} · {row['therapist_label']}**")
        st.caption(f"Intended Noa response: {row['noa_label']}")
        columns = st.columns(3)
        closed_text = (
            f"{row['closed_correct_rate']:.0%} (n={row['closed_n']})"
            if row["closed_correct_rate"] is not None
            else f"unavailable (n={row['closed_n']})"
        )
        open_text = (
            f"{row['open_median_coverage']:.2f} (n={row['open_n']})"
            if row["open_median_coverage"] is not None
            else f"unavailable (n={row['open_n']})"
        )
        survey_text = (
            f"{row['survey_change_median']:+.1f} (n={row['survey_n']})"
            if row["survey_change_median"] is not None
            else f"unavailable (n={row['survey_n']})"
        )
        columns[0].metric("Closed P(correct)", closed_text)
        columns[1].metric("Open median coverage", open_text)
        columns[2].metric(
            f"Survey Δ {row['primary_domain'].replace('_', ' ')}",
            survey_text,
        )


def _render_script_insights(cases: list[dict], closed_attempts: list[dict], diagnostics: dict) -> None:
    _layer_banner()
    st.caption(
        f"Samples — survey matched: {diagnostics['survey_matched']}; "
        f"closed: {diagnostics['closed_n']}; "
        f"open semantic: {diagnostics['open_semantic_n']}/{diagnostics['open_case_n']}."
    )
    pathway_tab, closed_tab, open_tab, calibration_tab, triangle_tab = st.tabs(
        (
            "Ground-truth pathway",
            "Closed cohort",
            "Open vs target",
            "Survey calibration",
            "Triangulation",
        )
    )
    with pathway_tab:
        _render_ground_truth()
    with closed_tab:
        _render_closed_cohort(closed_attempts)
    with open_tab:
        _render_open_vs_target(cases)
    with calibration_tab:
        _render_survey_calibration(cases)
    with triangle_tab:
        _render_triangulation(cases, closed_attempts)


def _render_methods(diagnostics: dict) -> None:
    st.markdown(
        "Scores use complete cases only. UES usability items are reverse-scored; UEQ-S values "
        "are converted from 1–7 to −3–3. Correlations are Spearman rank associations with "
        "deterministic bootstrap intervals when n≥10. "
        "Closed stage difficulty uses bootstrap proportion intervals. "
        "No result establishes causality or objective therapeutic competence."
    )
    st.markdown(
        f"- Survey matched to open attempts: **{diagnostics['survey_matched']}** "
        f"(unmatched surveys: {diagnostics['survey_unmatched']}; "
        f"multi-attempt open sessions: {diagnostics['multi_attempt_sessions']}).\n"
        f"- Closed completed sessions: **{diagnostics['closed_n']}** "
        f"(with recoverable stage detail: {diagnostics['closed_with_stage_detail']}).\n"
        f"- Open sessions with semantic coverage: **{diagnostics['open_semantic_n']}** / "
        f"{diagnostics['open_case_n']}.\n"
        f"- Closed languages: {diagnostics['closed_languages'] or 'none'}; "
        f"open languages: {diagnostics['open_languages'] or 'none'}."
    )
    st.markdown(
        "**Limitations:** closed multiple-choice recognition ≠ open therapeutic skill; "
        "embedding proximity and LLM guideline completion are not ground truth; "
        "survey change is not causal evidence; open and closed cohorts are not person-linked."
    )


def render_survey_outcomes(
    attempts: list[dict],
    deterministic: dict,
    semantic: dict,
    closed_attempts: list[dict] | None = None,
) -> None:
    """Render safely: workbook problems must not affect the existing analysis tabs."""
    try:
        surveys, _ = load_configured_surveys()
    except SurveyDataError as error:
        st.error(f"Survey outcomes unavailable: {error}")
        return
    cases, join_info = joined_cases(surveys, attempts, deterministic, semantic)
    closed_attempts = closed_attempts or []
    closed_summary = closed_cohort_summary(closed_attempts)
    coverage = open_stage_coverage(cases)
    diagnostics = insight_diagnostics(join_info, closed_summary, coverage, cases)

    change_tab, concordance_tab, experience_tab, script_tab, methods_tab = st.tabs(
        ("Self-reported change", "Self-report vs session", "Experience vs behavior", "Script insights", "Methods")
    )
    with change_tab:
        _render_self_report(cases)
    with concordance_tab:
        _render_domain_comparisons(cases)
    with experience_tab:
        _render_experience(cases)
    with script_tab:
        _render_script_insights(cases, closed_attempts, diagnostics)
    with methods_tab:
        _render_methods(diagnostics)
