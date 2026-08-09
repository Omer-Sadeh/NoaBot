"""Shared Firestore loading and filtering for saved conversation attempts."""

from __future__ import annotations

import datetime
import json
from typing import Any

import firebase_admin
import streamlit as st
from firebase_admin import firestore
from google.oauth2 import service_account

from conversation_csv import normalize_conversation


STATUS_OPTIONS = ("ongoing", "success", "no success", "unknown")
CACHE_TTL_SECONDS = 300


def setup_firestore():
    if not firebase_admin._apps:
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(st.secrets["firestore_creds"])
        )
        firebase_admin.initialize_app(credentials, {"projectId": "noabotprompts"})
    return firestore.client()


def detect_language_from_data(data: str) -> str:
    if not data:
        return "unknown"
    hebrew_characters = sum("\u0590" <= character <= "\u05ff" for character in data)
    if hebrew_characters > 10:
        return "he"
    english_markers = ("Session Duration:", "Conversation Transcript:", "-- Noa:")
    hebrew_markers = ("משך השיחה", "תמלול השיחה", "-- נועה:")
    english_count = sum(marker in data for marker in english_markers)
    hebrew_count = sum(marker in data for marker in hebrew_markers)
    if hebrew_count > english_count:
        return "he"
    if english_count > hebrew_count:
        return "en"
    return "unknown"


def _serialize_value(value: Any) -> Any:
    if isinstance(value, datetime.datetime):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, datetime.date):
        return {"__date__": value.isoformat()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value


def _deserialize_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_deserialize_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "__datetime__" in value:
        return datetime.datetime.fromisoformat(value["__datetime__"])
    if "__date__" in value:
        return datetime.date.fromisoformat(value["__date__"])
    return {key: _deserialize_value(item) for key, item in value.items()}


def _load_attempts(collection_name: str) -> list[dict]:
    database = setup_firestore()
    attempts = []
    for conversation in database.collection_group("conversations").stream():
        path = conversation.reference.path.split("/")
        if len(path) != 4 or path[0] != collection_name:
            continue
        conversation_data = conversation.to_dict()
        language = conversation_data.get("language", "unknown")
        if language == "unknown":
            language = detect_language_from_data(conversation_data.get("data", ""))
        attempts.append(
            normalize_conversation(
                {
                    **conversation_data,
                    "session_id": path[1],
                    "doc_id": conversation.id,
                    "mode": conversation_data.get("mode", "open"),
                    "session_language": language,
                }
            )
        )
    return attempts


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_attempts(collection_name: str) -> list[dict]:
    """Load normalized attempt dictionaries for one configured collection."""
    return _deserialize_value(_serialize_value(_load_attempts(collection_name)))


def clear_attempt_cache() -> None:
    load_attempts.clear()


def is_successful(conversation: dict) -> bool:
    if conversation.get("is_successful") is not None:
        return bool(conversation["is_successful"])
    if conversation.get("mode") == "open":
        return "Completed: True" in conversation.get("data", "")
    data = conversation.get("data", "")
    return "Closed Script Completed: True" in data


def effective_status(conversation: dict) -> str:
    if conversation.get("status") == "ongoing":
        return "ongoing"
    if conversation.get("status") == "completed":
        return "success" if is_successful(conversation) else "no success"
    return "unknown"


def attempt_date(conversation: dict):
    timestamp = conversation.get("timestamp")
    return timestamp.date() if hasattr(timestamp, "date") else timestamp


def available_date_range(conversations: list[dict]) -> tuple | tuple[None, None]:
    dates = [attempt_date(conversation) for conversation in conversations]
    dates = [date for date in dates if date is not None]
    return (min(dates), max(dates)) if dates else (None, None)


def render_filters(
    conversations: list[dict],
    *,
    mode: str | None = None,
    key_prefix: str = "conversations",
) -> dict:
    """Render filters that both admin screens use."""
    minimum, maximum = available_date_range(conversations)
    st.sidebar.header("Filters")
    if mode is None:
        selected_mode = st.sidebar.selectbox(
            "Mode",
            options=("All", "open", "closed"),
            key=f"{key_prefix}_mode",
        )
    else:
        selected_mode = mode
        st.sidebar.caption(f"Mode: {mode}")
    date_value = (
        (minimum, maximum) if minimum is not None and maximum is not None else None
    )
    return {
        "mode": selected_mode,
        "statuses": st.sidebar.multiselect(
            "Status",
            options=STATUS_OPTIONS,
            default=STATUS_OPTIONS,
            key=f"{key_prefix}_status",
        ),
        "date_range": st.sidebar.date_input(
            "Date range",
            value=date_value,
            key=f"{key_prefix}_date",
        ),
        "session_id": st.sidebar.text_input(
            "Session ID contains",
            key=f"{key_prefix}_session_id",
        ),
        "language": st.sidebar.selectbox(
            "Language",
            options=("All", "en", "he", "unknown"),
            key=f"{key_prefix}_language",
        ),
    }


def filter_attempts(conversations: list[dict], filters: dict) -> list[dict]:
    filtered = []
    for conversation in conversations:
        if filters["mode"] != "All" and conversation.get("mode") != filters["mode"]:
            continue
        if filters["statuses"] and effective_status(conversation) not in filters["statuses"]:
            continue
        selected_date_range = filters["date_range"]
        date = attempt_date(conversation)
        if (
            isinstance(selected_date_range, tuple)
            and len(selected_date_range) == 2
            and all(selected_date_range)
            and date is not None
            and not (selected_date_range[0] <= date <= selected_date_range[1])
        ):
            continue
        if (
            filters["session_id"]
            and filters["session_id"] not in conversation.get("session_id", "")
        ):
            continue
        if (
            filters["language"] != "All"
            and conversation.get("session_language") != filters["language"]
        ):
            continue
        filtered.append(conversation)
    return sorted(filtered, key=lambda item: item.get("timestamp") or 0, reverse=True)
