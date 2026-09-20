"""Shared research definitions for analysis UI copy and export metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MeasureDefinition:
    title: str
    summary: str
    operationalization: str
    interpretation: str
    limitations: str
    unit: str
    missingness: str
    source: str


MEASURE_DEFINITIONS = {
    "sample_overview": MeasureDefinition(
        "Analysis sample",
        "Counts the saved attempts and unique session IDs included by the active filters.",
        "Attempts are saved conversation documents; one session ID may contain multiple attempts. "
        "The analysis includes open attempts with a completed success/no-success outcome.",
        "Use attempt counts for document-level analyses and session counts only when the analysis "
        "explicitly selects one attempt per session.",
        "Attempts are not independent when several share a session ID. LLM-judged guideline "
        "completion is not externally validated performance.",
        "count",
        "Rows outside the active mode, outcome, and date filters are excluded.",
        "Saved conversation metadata",
    ),
    "conversation_duration": MeasureDefinition(
        "Conversation duration",
        "Measures elapsed time during an open conversation attempt.",
        "Saved elapsed seconds from the first trainee turn through the end of the attempt, converted "
        "to minutes for display.",
        "Higher values indicate longer elapsed sessions, not necessarily more active engagement.",
        "Includes user idle time and system latency. Historical records may use different timing "
        "instrumentation.",
        "minutes",
        "Missing when no reliable saved duration is available.",
        "Open-session timing telemetry",
    ),
    "trainee_turns": MeasureDefinition(
        "Trainee turns",
        "Counts trainee-authored messages in an attempt.",
        "Number of structured conversation turns whose role is `user`.",
        "Describes interaction volume; it is not a measure of response quality.",
        "Legacy records without recoverable structured turns may be incomplete.",
        "turns",
        "Zero means no recoverable trainee turns; blank indicates unavailable source data.",
        "Structured conversation turns",
    ),
    "section_timing": MeasureDefinition(
        "Goal-section timing",
        "Measures elapsed time before the conversation advanced through each goal section.",
        "Uses per-section durations for current records and derives section intervals from saved "
        "cumulative offsets for compatible legacy records.",
        "Compare sections only when instrumentation versions and timing semantics are compatible.",
        "Includes idle time and latency. A transition reflects the application state, not an "
        "independent judgment of therapeutic mastery.",
        "seconds",
        "Missing for attempts without recoverable section timing.",
        "Section transition telemetry",
    ),
    "reference_coverage": MeasureDefinition(
        "Reference-move coverage",
        "Estimates semantic proximity between observed conversation turns and authored script moves.",
        "For each language-matched reference move, take the maximum embedding cosine similarity "
        "across observed turns by the matching speaker; displayed attempt values average available moves.",
        "Scores nearer 1 indicate greater semantic similarity to the authored example.",
        "Similarity is not correctness, therapeutic quality, or competence. Results depend on the "
        "embedding model, authored reference, and language.",
        "cosine similarity, nominally 0–1",
        "Missing until semantic analysis is cached or when no usable text/embedding pair exists.",
        "OpenAI embeddings and authored closed-script references",
    ),
    "domain_reference_coverage": MeasureDefinition(
        "Domain-reference coverage",
        "Estimates semantic proximity between trainee language and authored domain rubric anchors.",
        "For each anchor in a domain, take the highest embedding cosine similarity among trainee "
        "turns, then average the available anchor scores.",
        "Higher values mean the session used language more similar to the domain anchors.",
        "This is an exploratory language proxy, not a direct assessment of therapeutic competence.",
        "cosine similarity, nominally 0–1",
        "Missing until semantic analysis is cached or when no usable trainee text is available.",
        "OpenAI embeddings and language-specific domain anchors",
    ),
    "sequence_fidelity": MeasureDefinition(
        "Exploratory sequence fidelity",
        "Estimates whether semantically similar trainee moves occurred in the authored order.",
        "Greedily aligns each observed trainee turn to an equal-or-later reference move and averages "
        "the matched cosine similarities.",
        "Higher values indicate stronger ordered semantic resemblance to the reference pathway.",
        "Greedy alignment is exploratory, can skip moves, and does not establish process quality.",
        "mean cosine similarity",
        "Missing when semantic embeddings cannot produce ordered matches.",
        "OpenAI embeddings and authored closed-script order",
    ),
    "question_rate": MeasureDefinition(
        "Question rate",
        "Describes how often trainees use question punctuation.",
        "Count question marks in trainee messages and divide by the number of trainee turns.",
        "Higher values indicate more questions per message, not better guiding questions.",
        "Punctuation is a surface feature and can miss indirect questions or count multiple marks.",
        "questions per trainee turn",
        "Missing when there are no trainee turns.",
        "Trainee message text",
    ),
    "reflection_marker_rate": MeasureDefinition(
        "Reflection-marker rate",
        "Describes use of a small language-specific list of reflective phrases.",
        "Count configured English/Hebrew reflection markers in trainee text and divide by trainee "
        "word count; the UI reports the result per 100 words.",
        "Higher values indicate more listed phrases, not necessarily more accurate reflection.",
        "Dictionary matching misses paraphrases and context and may count formulaic use.",
        "markers per 100 words",
        "Missing when no trainee words are recoverable.",
        "Trainee text and configured reflection-marker dictionary",
    ),
    "lexical_diversity": MeasureDefinition(
        "Lexical diversity (MATTR-50)",
        "Describes variation in trainee vocabulary while controlling for text length.",
        "Average type-token ratio over every moving 50-token window in trainee text.",
        "Higher values indicate more varied vocabulary within this sample.",
        "Vocabulary diversity is not conversational quality and is language/tokenization dependent.",
        "proportion, 0–1",
        "Missing for attempts with fewer than 50 trainee tokens.",
        "Tokenized trainee text",
    ),
    "repetition_rate": MeasureDefinition(
        "Repeated-token share",
        "Describes how much trainee vocabulary is reused.",
        "One minus unique token count divided by total trainee token count.",
        "Higher values indicate more token reuse.",
        "Common function words and language morphology affect the score; repetition may be clinically appropriate.",
        "proportion, 0–1",
        "Missing when no trainee tokens are recoverable.",
        "Tokenized trainee text",
    ),
    "sentence_length": MeasureDefinition(
        "Median sentence length",
        "Describes the typical length of trainee sentences.",
        "Split trainee text on supported sentence punctuation, count tokens, and take the median.",
        "Higher values indicate longer detected sentences.",
        "Punctuation and language-specific sentence boundaries affect segmentation; this is not difficulty.",
        "words per sentence",
        "Missing when no sentence can be detected.",
        "Trainee message text",
    ),
    "style_alignment": MeasureDefinition(
        "Exploratory style alignment",
        "Compares function-word profiles in adjacent trainee and Noa messages.",
        "Compute one minus mean absolute distance between language-specific function-word rates for "
        "adjacent turns; report medians only with at least three trainee-to-Noa pairs and compare "
        "with reversed within-attempt pairings.",
        "Higher values indicate more similar surface style; higher is not inherently better.",
        "The small dictionary and shuffled baseline are exploratory and do not isolate interpersonal adaptation.",
        "alignment score, nominally 0–1",
        "Missing when fewer than three usable adjacent turn pairs exist.",
        "Adjacent conversation turns and function-word dictionaries",
    ),
    "self_report_change": MeasureDefinition(
        "Self-reported confidence change",
        "Measures within-person change in domain confidence from pre- to post-session.",
        "Subtract pre-session confidence from post-session confidence; summarize with the median, a "
        "deterministic bootstrap 95% interval, and Wilcoxon signed-rank p-value when eligible.",
        "Positive values indicate higher self-reported confidence after the session.",
        "Without a control group this is not causal evidence of learning or effectiveness.",
        "confidence points, 0–100 scale",
        "Complete-case analysis; pairs missing either pre or post are excluded.",
        "Matched pre/post survey responses",
    ),
    "spearman_association": MeasureDefinition(
        "Spearman association",
        "Measures monotonic association between two observed variables.",
        "Compute Spearman rank correlation on complete pairs; report bootstrap intervals and p-values "
        "only when at least 10 pairs are available.",
        "ρ ranges from -1 to 1; sign gives direction and magnitude gives rank association strength.",
        "Association is not causation. Small samples and multiple exploratory comparisons increase uncertainty.",
        "Spearman ρ",
        "Missing with fewer than three complete, non-degenerate pairs; intervals/p-values omitted below n=10.",
        "Matched survey and session-derived measures",
    ),
    "scale_reliability": MeasureDefinition(
        "Scale internal consistency",
        "Estimates consistency among items assigned to the same survey scale.",
        "Cronbach's α on complete cases after configured reverse scoring and scale transformations.",
        "Higher α indicates greater item covariance, not validity or unidimensionality.",
        "Unstable in small samples and undefined for degenerate responses; thresholds should not be applied mechanically.",
        "Cronbach's α",
        "Missing with fewer than three complete cases or zero total-score variance.",
        "Survey item responses",
    ),
    "experience_scales": MeasureDefinition(
        "Experience and agent-perception scales",
        "Summarizes complete survey item sets for engagement, user experience, agent perception, and reuse intention.",
        "Average the declared items for each UES-SF, UEQ-S, and agent-perception subscale. "
        "Perceived Usability items are reverse-scored; UEQ-S items are shifted from 1–7 to -3–3. "
        "Reuse intention is a single 1–5 item.",
        "Interpret each named subscale separately; no UES total is calculated because Aesthetic Appeal items are absent.",
        "Scale means do not establish effectiveness. Agent-perception ratings describe the simulated agent experience.",
        "UES/agent means 1–5; UEQ-S means -3–3; reuse intention 1–5",
        "A scale is missing unless every required item for that scale is present.",
        "Post-session survey items",
    ),
    "closed_accuracy": MeasureDefinition(
        "Closed-script recognition accuracy",
        "Measures recognition of authored therapist moves in the multiple-choice script.",
        "Correct choices divided by answered stages, summarized across one selected completed attempt per session.",
        "Higher values indicate closer recognition of the authored answer key.",
        "Multiple-choice recognition is not equivalent to open-session therapeutic skill.",
        "proportion correct, 0–1",
        "Missing when neither totals nor recoverable stage-level answers are available.",
        "Closed-script responses and authored answer key",
    ),
    "closed_stage_difficulty": MeasureDefinition(
        "Closed-stage correct rate",
        "Describes the proportion of selected closed attempts answered correctly at each stage.",
        "Correct stage responses divided by usable responses, with deterministic bootstrap 95% intervals.",
        "Lower correct rates indicate stages that were harder for this cohort to recognize.",
        "Difficulty is sample- and item-specific and does not estimate a latent ability scale.",
        "proportion correct, 0–1",
        "Missing for stages without recoverable correctness.",
        "Closed-script stage responses",
    ),
    "closed_distractor_pattern": MeasureDefinition(
        "Closed-script distractor pattern",
        "Describes which authored error pattern appeared among incorrect stage choices.",
        "Classify each selected incorrect option against the authored option set, then divide each "
        "pattern count by all recoverable errors at that stage.",
        "Higher shares indicate more common selected distractors within that stage's errors.",
        "Categories are authored pedagogical labels, not diagnoses of participant reasoning.",
        "count and share of stage errors",
        "Missing when selected answers cannot be recovered or classified.",
        "Closed-script choices and authored distractor mapping",
    ),
    "calibration_quadrants": MeasureDefinition(
        "Median-split concordance quadrants",
        "Compares relative self-reported confidence with relative semantic-proxy values.",
        "Split post confidence and the domain proxy at their sample medians and count the four combinations.",
        "Quadrants describe within-sample concordance only.",
        "They are not classifications of overconfidence, underconfidence, or competence; cut points change with the sample.",
        "count and proportion",
        "Requires at least two complete matched values.",
        "Matched surveys and domain-reference coverage",
    ),
    "triangulation": MeasureDefinition(
        "Cross-source triangulation",
        "Places closed recognition, open semantic proximity, and survey change beside the same authored stage/domain.",
        "Map each authored stage to its primary survey domain and report each source's independently computed summary.",
        "Use the profile to compare patterns across evidence layers, not individual participants.",
        "Closed and open/survey cohorts are independent and the scales are not interchangeable.",
        "source-specific units",
        "Each source remains missing independently when its sample is unavailable.",
        "Authored mapping, closed responses, open semantics, and matched surveys",
    ),
}


def measure_definition(key: str) -> MeasureDefinition:
    return MEASURE_DEFINITIONS[key]


def measure_codebook_rows() -> list[dict]:
    return [{"measure_id": key, **asdict(value)} for key, value in MEASURE_DEFINITIONS.items()]


def render_measure_explanation(key: str) -> None:
    import streamlit as st

    definition = measure_definition(key)
    st.caption(definition.summary)
    with st.expander(f"How {definition.title.lower()} is measured and interpreted"):
        st.markdown(f"**Operationalization:** {definition.operationalization}")
        st.markdown(f"**Interpretation:** {definition.interpretation}")
        st.markdown(f"**Unit:** {definition.unit}")
        st.markdown(f"**Missing values:** {definition.missingness}")
        st.markdown(f"**Limitations:** {definition.limitations}")
