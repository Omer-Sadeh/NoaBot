"""Pure metrics for descriptive open-conversation analysis."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from statistics import median

from script_loader import load_closed_script


ANALYZER_VERSION = "1"
MIN_STYLE_TURNS = 3
WORD_PATTERN = re.compile(r"[\w\u0590-\u05ff]+", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?。؟])\s+")

FUNCTION_WORDS = {
    "en": {
        "i", "you", "he", "she", "we", "they", "it", "a", "an", "the", "and",
        "or", "but", "if", "because", "to", "of", "in", "on", "for", "with",
        "is", "are", "was", "were", "do", "does", "not",
    },
    "he": {
        "אני", "את", "אתה", "הוא", "היא", "אנחנו", "הם", "הן", "של", "עם",
        "על", "אל", "אם", "אבל", "כי", "גם", "לא", "כן", "זה", "זאת", "היה",
        "הייתה", "יהיה", "יש", "אין",
    },
}

REFLECTION_MARKERS = {
    "en": ("it sounds", "i hear", "you feel", "that sounds", "i can see"),
    "he": ("נשמע", "אני שומע", "אני מבינה", "את מרגישה", "זה נשמע"),
}


def canonical_hash(value) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def cache_key(source_hash: str, language: str, model: str) -> str:
    return canonical_hash(
        {
            "analysis_version": ANALYZER_VERSION,
            "source_hash": source_hash,
            "language": language,
            "model": model,
        }
    )


def words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower())


def sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_PATTERN.split(text) if sentence.strip()]


def role_turns(attempt: dict, role: str) -> list[dict]:
    return [turn for turn in attempt.get("turns", []) if turn.get("role") == role]


def turn_contents(turns: list[dict]) -> list[str]:
    return [turn["content"] for turn in turns if turn.get("content")]


def generated_role_texts(attempt: dict, role: str) -> list[str]:
    texts = turn_contents(role_turns(attempt, role))
    return texts[1:] if role == "noa" else texts


def count_questions(texts: list[str]) -> int:
    return sum(text.count("?") + text.count("؟") for text in texts)


def moving_type_token_ratio(tokens: list[str], window: int = 50) -> float | None:
    if len(tokens) < window:
        return None
    ratios = [
        len(set(tokens[index : index + window])) / window
        for index in range(len(tokens) - window + 1)
    ]
    return sum(ratios) / len(ratios)


def median_or_none(values: list[float]) -> float | None:
    return round(median(values), 3) if values else None


def ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


def language_reference(language: str) -> dict[str, list[str]]:
    script = load_closed_script(language)
    return {
        "noa": [entry["Noa"] for entry in script if entry.get("Noa")],
        "user": [
            entry["correct_answer"]
            for entry in script
            if entry.get("correct_answer")
        ],
    }


def profile_texts(texts: list[str]) -> dict:
    token_lists = [words(text) for text in texts]
    tokens = [token for token_list in token_lists for token in token_list]
    sentence_list = [sentence for text in texts for sentence in sentences(text)]
    return {
        "turn_count": len(texts),
        "word_count": len(tokens),
        "character_count": sum(len(text) for text in texts),
        "median_words_per_turn": median_or_none([len(value) for value in token_lists]),
        "median_characters_per_turn": median_or_none([len(text) for text in texts]),
        "median_words_per_sentence": median_or_none(
            [len(words(sentence)) for sentence in sentence_list]
        ),
        "question_rate": ratio(count_questions(texts), len(texts)),
        "mattr_50": moving_type_token_ratio(tokens),
        "repetition_rate": repetition_rate(tokens),
    }


def repetition_rate(tokens: list[str]) -> float | None:
    if not tokens:
        return None
    return round(1 - len(set(tokens)) / len(tokens), 3)


def function_word_profile(text: str, language: str) -> dict[str, float]:
    tokens = words(text)
    function_words = FUNCTION_WORDS.get(language, set())
    if not tokens:
        return {}
    counts = Counter(token for token in tokens if token in function_words)
    return {word: count / len(tokens) for word, count in counts.items()}


def style_distance(left: str, right: str, language: str) -> float | None:
    left_profile = function_word_profile(left, language)
    right_profile = function_word_profile(right, language)
    keys = set(left_profile) | set(right_profile)
    if not keys:
        return None
    return sum(abs(left_profile.get(key, 0) - right_profile.get(key, 0)) for key in keys) / len(keys)


def style_alignment(attempt: dict, language: str) -> dict:
    turns = attempt.get("turns", [])
    user_to_noa = []
    noa_to_user = []
    user_turns = turn_contents(role_turns(attempt, "user"))
    noa_turns = generated_role_texts(attempt, "noa")
    for previous, current in zip(turns, turns[1:]):
        if previous.get("role") == "noa" and current.get("role") == "user":
            distance = style_distance(current.get("content", ""), previous.get("content", ""), language)
            if distance is not None:
                user_to_noa.append(1 - distance)
        if previous.get("role") == "user" and current.get("role") == "noa":
            distance = style_distance(current.get("content", ""), previous.get("content", ""), language)
            if distance is not None:
                noa_to_user.append(1 - distance)
    enough_data = len(user_to_noa) >= MIN_STYLE_TURNS
    null_alignments = []
    for user_turn, shuffled_noa_turn in zip(user_turns, reversed(noa_turns)):
        distance = style_distance(user_turn, shuffled_noa_turn, language)
        if distance is not None:
            null_alignments.append(1 - distance)
    return {
        "available": enough_data,
        "user_to_noa_mean_alignment": median_or_none(user_to_noa) if enough_data else None,
        "noa_to_user_mean_alignment": median_or_none(noa_to_user) if enough_data else None,
        "within_attempt_shuffled_null": median_or_none(null_alignments)
        if enough_data
        else None,
        "user_to_noa_pairs": len(user_to_noa),
        "note": "Exploratory only. Higher alignment is not better."
        if enough_data
        else "Insufficient turn pairs for exploratory style alignment.",
    }


def therapeutic_register(attempt: dict, language: str) -> dict:
    user_texts = turn_contents(role_turns(attempt, "user"))
    full_text = " ".join(user_texts).lower()
    token_count = len(words(full_text))
    markers = REFLECTION_MARKERS.get(language, ())
    return {
        "question_rate": ratio(count_questions(user_texts), len(user_texts)),
        "reflection_marker_rate": ratio(
            sum(full_text.count(marker) for marker in markers), token_count
        ),
        "second_person_rate": ratio(
            sum(token in {"you", "your", "את", "אתה", "שלך"} for token in words(full_text)),
            token_count,
        ),
        "first_person_rate": ratio(
            sum(token in {"i", "me", "my", "אני", "שלי"} for token in words(full_text)),
            token_count,
        ),
    }


def timing_profile(attempt: dict) -> dict:
    durations = attempt.get("section_durations_seconds")
    transitions = attempt.get("section_transition_events", [])
    return {
        "time_semantics": attempt.get("section_time_semantics"),
        "section_durations_seconds": durations,
        "section_user_turns": attempt.get("section_user_turns"),
        "completed_sections": sum(
            event.get("completed", False) for event in transitions
        ),
        "transition_events": transitions,
    }


def deterministic_metrics(attempt: dict) -> dict:
    language = attempt.get("session_language", "en")
    user_texts = turn_contents(role_turns(attempt, "user"))
    noa_texts = turn_contents(role_turns(attempt, "noa"))
    reference = language_reference(language)
    return {
        "attempt_id": attempt.get("attempt_id") or attempt.get("doc_id"),
        "source_hash": canonical_hash(
            {
                "turns": attempt.get("turns", []),
                "status": attempt.get("status"),
                "section_durations_seconds": attempt.get("section_durations_seconds"),
            }
        ),
        "language": language,
        "length": {
            "duration_seconds": attempt.get("session_duration_seconds"),
            "all": profile_texts(user_texts + noa_texts),
            "user": profile_texts(user_texts),
            "noa": profile_texts(noa_texts),
        },
        "timing": timing_profile(attempt),
        "complexity": profile_texts(user_texts),
        "reference_user_profile": profile_texts(reference["user"]),
        "therapeutic_register": therapeutic_register(attempt, language),
        "style_alignment": style_alignment(attempt, language),
        "completion": {
            "guidelines_cleared": attempt.get("completed_guidelines"),
            "guidelines_total": attempt.get("completed_criteria_total")
            or attempt.get("guidelines_total"),
            "is_llm_judged_success": attempt.get("is_successful"),
        },
    }


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def semantic_texts(attempt: dict) -> list[str]:
    language = attempt.get("session_language", "en")
    reference = language_reference(language)
    texts = []
    for role in ("noa", "user"):
        texts.extend(generated_role_texts(attempt, role))
        texts.extend(reference[role])
    return list(dict.fromkeys(text for text in texts if text.strip()))


def semantic_metrics(attempt: dict, embeddings: dict[str, list[float]]) -> dict:
    language = attempt.get("session_language", "en")
    reference = language_reference(language)
    roles = {}
    for role in ("noa", "user"):
        observed = generated_role_texts(attempt, role)
        reference_moves = reference[role]
        coverage = []
        nearest_distances = []
        for reference_move in reference_moves:
            similarities = [
                cosine_similarity(embeddings[text], embeddings[reference_move])
                for text in observed
                if text in embeddings and reference_move in embeddings
            ]
            coverage.append(round(max(similarities), 6) if similarities else None)
        for observed_turn in observed:
            similarities = [
                cosine_similarity(embeddings[observed_turn], embeddings[reference_move])
                for reference_move in reference_moves
                if observed_turn in embeddings and reference_move in embeddings
            ]
            if similarities:
                nearest_distances.append(round(1 - max(similarities), 6))
        roles[role] = {
            "reference_move_coverage": coverage,
            "median_nearest_reference_distance": median_or_none(nearest_distances),
            "turn_count": len(observed),
            "exploratory_monotonic_alignment": monotonic_alignment(
                observed, reference_moves, embeddings
            ),
        }
    return {"analysis_version": ANALYZER_VERSION, "reference_trajectory": roles}


def monotonic_alignment(
    observed: list[str],
    reference_moves: list[str],
    embeddings: dict[str, list[float]],
) -> dict:
    """Greedily align each turn to an equal-or-later reference move."""
    last_reference_index = 0
    matches = []
    for turn in observed:
        candidates = [
            (
                index,
                cosine_similarity(embeddings[turn], embeddings[reference_move]),
            )
            for index, reference_move in enumerate(reference_moves)
            if index >= last_reference_index
            and turn in embeddings
            and reference_move in embeddings
        ]
        if not candidates:
            continue
        index, similarity = max(candidates, key=lambda item: item[1])
        last_reference_index = index
        matches.append({"reference_index": index, "similarity": round(similarity, 6)})
    return {
        "matches": matches,
        "mean_similarity": round(
            sum(match["similarity"] for match in matches) / len(matches), 6
        )
        if matches
        else None,
    }
