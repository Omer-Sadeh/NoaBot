import streamlit as st
import json
import time
from google.oauth2 import service_account
import firebase_admin
from firebase_admin import firestore
import datetime

def setup_firestore():
    if not firebase_admin._apps:
        cred = service_account.Credentials.from_service_account_info(json.loads(st.secrets["firestore_creds"]))
        firebase_admin.initialize_app(cred, {'projectId': 'noabotprompts',})
    return firestore.client()

def backfill_sessions(db):
    # Get all session IDs from conversations subcollections
    sessions_ref = db.collection("sessions")
    conv_session_ids = set()
    for session_ref in sessions_ref.list_documents():
        conv_session_ids.add(session_ref.id)
    # For each session_id, check if parent exists
    missing = []
    for session_id in conv_session_ids:
        doc_ref = db.collection("sessions").document(session_id)
        if not doc_ref.get().exists:
            missing.append(session_id)
            doc_ref.set({"created": firestore.SERVER_TIMESTAMP}, merge=True)
    return missing

def render_database_screen():
    st.title("Saved Conversations Database")
    db = setup_firestore()
    # Backfill missing parent session documents automatically
    backfill_sessions(db)
    conversations = []
    sessions = list(db.collection("sessions").stream())
    for i, session in enumerate(sessions):
        session_id = session.id
        convs = list(db.collection("sessions").document(session_id).collection("conversations").order_by("timestamp", direction=firestore.Query.DESCENDING).stream())
        for j, conv in enumerate(convs):
            data = conv.to_dict()
            conversations.append({
                "timestamp": data.get("timestamp"),
                "data": data.get("data", ""),
                "session_id": session_id,
                "doc_id": conv.id,
                "mode": data.get("mode", "open")
            })
    # --- FILTERS ---
    st.sidebar.header("Filters")
    mode_filter = st.sidebar.selectbox("Mode", options=["All", "open", "closed"], index=0)
    # Date range filter
    min_date = None
    max_date = None
    timestamps = [c["timestamp"] for c in conversations if c["timestamp"]]
    if timestamps:
        min_date = min([t.date() if hasattr(t, 'date') else t for t in timestamps])
        max_date = max([t.date() if hasattr(t, 'date') else t for t in timestamps])
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date) if min_date and max_date else (None, None))
    completed_filter = st.sidebar.selectbox("Completed", options=["All", "Completed", "Not Completed"], index=0)
    session_id_filter = st.sidebar.text_input("Session ID contains")
    # --- FILTERING LOGIC ---
    def is_completed(conv):
        data = conv["data"]
        if conv["mode"] == "open":
            if "Completed: True" in data:
                return True
            if "Completed: False" in data:
                return False
        elif conv["mode"] == "closed":
            if "Closed Script Completed: True" in data:
                return True
            if "Closed Script Completed: False" in data:
                return False
        return False  # Default if not found
    filtered = []
    for conv in conversations:
        # Mode filter
        if mode_filter != "All" and conv["mode"] != mode_filter:
            continue
        # Date range filter
        ts = conv["timestamp"]
        if ts and hasattr(ts, 'date'):
            ts_date = ts.date()
            if date_range and isinstance(date_range, tuple) and all(date_range):
                if ts_date < date_range[0] or ts_date > date_range[1]:
                    continue
        # Completed filter
        if completed_filter == "Completed" and not is_completed(conv):
            continue
        if completed_filter == "Not Completed" and is_completed(conv):
            continue
        # Session ID filter
        if session_id_filter and session_id_filter not in conv["session_id"]:
            continue
        filtered.append(conv)
    filtered.sort(key=lambda x: x["timestamp"] or 0, reverse=True)
    for conv in filtered:
        ts = conv["timestamp"]
        if ts:
            try:
                ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                ts_str = str(ts)
        else:
            ts_str = "No timestamp"
        label = f"Session: `{conv['session_id']}` | Time: {ts_str} | Mode: {conv['mode']}"
        with st.expander(label):
            st.code(conv["data"], language="text")
        st.markdown("---")
    if st.button("Back to Menu"):
        st.session_state.pre_done = False
        st.rerun() 