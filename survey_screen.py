"""Streamlit presentation for survey outcomes and observed session measures."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import streamlit as st

import config
from survey_analysis import (
    joined_cases,
    observed_value,
    paired_summary,
    scale_reliability,
    spearman_summary,
)
from survey_data import DOMAIN_COLUMNS, SurveyDataError, load_survey_rows, workbook_hash


@st.cache_data(ttl=300, show_spinner=False)
def _cached_rows(path: str, source_hash: str):
    del source_hash
    return load_survey_rows(path)


def clear_survey_cache() -> None:
    _cached_rows.clear()


def load_configured_surveys() -> tuple[list[dict], dict]:
    path = Path(config.SURVEY_WORKBOOK_PATH)
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


def _render_self_report(cases: list[dict]) -> None:
    st.subheader("Self-reported change")
    st.caption("Pre/post values are self-reported confidence (0–100). Changes occurred following the session; this design does not establish causality.")
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
    st.caption("Observed domain-reference coverage is semantic similarity to language-specific rubric anchors, not a direct measure of therapeutic competence.")
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
    st.caption("UES Perceived Usability is reverse scored. No total UES-SF score is shown because Aesthetic Appeal items are unavailable.")
    for score in ("Focused attention", "Perceived usability", "Reward", "Pragmatic quality", "Hedonic quality", "Overall UX", "Anthropomorphism", "Likeability", "Reuse intention"):
        st.markdown(f"**{score}**")
        _distribution(cases, score)
    reliability = scale_reliability(cases)
    st.caption("Internal consistency (Cronbach’s α; complete cases only; exploratory at this sample size): " + ", ".join(
        f"{name}={value if value is not None else 'unavailable'}" for name, value in reliability.items()
    ))
    definitions = (
        ("Focused attention", "duration_minutes"), ("Focused attention", "trainee_turns"),
        ("Perceived usability", "tips_shown"), ("Reward", "guideline_completion"),
        ("Reuse intention", "guideline_completion"), ("Anthropomorphism", "style_alignment"),
    )
    st.markdown("**Curated associations**")
    for score, observed in definitions:
        result = spearman_summary(cases, lambda case, s=score: case["instruments"].get(s), lambda case, o=observed: observed_value(case, o))
        st.write(f"{score} × {observed.replace('_', ' ')}: n={result['n']}, ρ={result['rho']}" + (f", 95% CI {result.get('ci_low')} to {result.get('ci_high')}" if result.get("ci_low") is not None else ""))


def render_survey_outcomes(attempts: list[dict], deterministic: dict, semantic: dict) -> None:
    """Render safely: workbook problems must not affect the existing analysis tabs."""
    try:
        surveys, _ = load_configured_surveys()
    except SurveyDataError as error:
        st.error(f"Survey outcomes unavailable: {error}")
        return
    cases, _ = joined_cases(surveys, attempts, deterministic, semantic)
    change_tab, concordance_tab, experience_tab, script_tab, methods_tab = st.tabs(
        ("Self-reported change", "Self-report vs session", "Experience vs behavior", "Script vs learning", "Methods")
    )
    with change_tab:
        _render_self_report(cases)
    with concordance_tab:
        _render_domain_comparisons(cases)
    with experience_tab:
        _render_experience(cases)
    with script_tab:
        st.caption("Reference-move coverage measures similarity to the closed-script examples. It is exploratory and not a quality score.")
        for domain in DOMAIN_COLUMNS:
            result = spearman_summary(cases, lambda case: observed_value(case, "reference_coverage"), lambda case, d=domain: case["domains"][d]["change"])
            st.write(f"Reference coverage × {domain.replace('_', ' ')} change: n={result['n']}, ρ={result['rho']}")
    with methods_tab:
        st.markdown("Scores use complete cases only. UES usability items are reverse-scored; UEQ-S values are converted from 1–7 to −3–3. Correlations are Spearman rank associations with deterministic bootstrap intervals when n≥10. No result establishes causality or objective therapeutic competence.")
