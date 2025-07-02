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

def get_session_status(conv):
    """Determine session status from conversation data"""
    # Check new format first
    if "status" in conv:
        return conv["status"]
    
    # Fallback to analyzing data content for backward compatibility
    data = conv.get("data", "")
    if "Status: " in data:
        for line in data.split("\n"):
            if line.startswith("Status: "):
                return line.replace("Status: ", "").strip()
    
    # Check completion status for legacy format
    if conv.get("mode") == "open":
        if "Completed: True" in data:
            return "completed"
        elif "Completed: False" in data:
            return "ongoing"
    elif conv.get("mode") == "closed":
        if "Closed Script Completed: True" in data:
            return "completed"
        elif "Closed Script Completed: False" in data:
            return "ongoing"
    
    # Default fallback
    return "unknown"

def render_database_screen():
    st.title("Saved Conversations Database")
    db = setup_firestore()
    # Backfill missing parent session documents automatically
    backfill_sessions(db)
    conversations = []
    sessions = list(db.collection("sessions").stream())
    
    for i, session in enumerate(sessions):
        session_id = session.id
        session_data = session.to_dict()
        convs = list(db.collection("sessions").document(session_id).collection("conversations").order_by("timestamp", direction=firestore.Query.DESCENDING).stream())
        for j, conv in enumerate(convs):
            data = conv.to_dict()
            conversations.append({
                "timestamp": data.get("timestamp"),
                "data": data.get("data", ""),
                "session_id": session_id,
                "doc_id": conv.id,
                "mode": data.get("mode", "open"),
                "status": get_session_status(data),
                "is_completed": data.get("is_completed", None),
                "session_created": session_data.get("created"),
                "session_language": session_data.get("language", "unknown")
            })
    
    # --- FILTERS ---
    st.sidebar.header("Filters")
    mode_filter = st.sidebar.selectbox("Mode", options=["All", "open", "closed"], index=0)
    
    # Status filter (new)
    status_filter = st.sidebar.selectbox("Status", options=["All", "ongoing", "completed", "unknown"], index=0)
    
    # Date range filter
    min_date = None
    max_date = None
    timestamps = [c["timestamp"] for c in conversations if c["timestamp"]]
    if timestamps:
        min_date = min([t.date() if hasattr(t, 'date') else t for t in timestamps])
        max_date = max([t.date() if hasattr(t, 'date') else t for t in timestamps])
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date) if min_date and max_date else (None, None))
    
    # Keep backward compatible completion filter for legacy data
    completed_filter = st.sidebar.selectbox("Completed (Legacy)", options=["All", "Completed", "Not Completed"], index=0)
    
    session_id_filter = st.sidebar.text_input("Session ID contains")
    
    # Language filter (new)
    language_filter = st.sidebar.selectbox("Language", options=["All", "en", "he", "unknown"], index=0)
    
    # --- FILTERING LOGIC ---
    def is_completed(conv):
        # Use new status field if available
        if conv.get("is_completed") is not None:
            return conv["is_completed"]
        
        # Fallback to legacy logic
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
        
        # Status filter
        if status_filter != "All" and conv["status"] != status_filter:
            continue
        
        # Date range filter
        ts = conv["timestamp"]
        if ts and hasattr(ts, 'date'):
            ts_date = ts.date()
            if date_range and isinstance(date_range, tuple) and all(date_range):
                if ts_date < date_range[0] or ts_date > date_range[1]:
                    continue
        
        # Legacy completed filter (for backward compatibility)
        if completed_filter == "Completed" and not is_completed(conv):
            continue
        if completed_filter == "Not Completed" and is_completed(conv):
            continue
        
        # Session ID filter
        if session_id_filter and session_id_filter not in conv["session_id"]:
            continue
            
        # Language filter
        if language_filter != "All" and conv["session_language"] != language_filter:
            continue
        
        filtered.append(conv)
    
    filtered.sort(key=lambda x: x["timestamp"] or 0, reverse=True)
    
    # Display session count
    st.write(f"Found {len(filtered)} sessions")
    
    for conv in filtered:
        ts = conv["timestamp"]
        if ts:
            try:
                ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                ts_str = str(ts)
        else:
            ts_str = "No timestamp"
        
        # Enhanced label with status information
        status_emoji = {"ongoing": "🔄", "completed": "✅", "unknown": "❓"}.get(conv["status"], "❓")
        label = f"{status_emoji} Session: `{conv['session_id']}` | Time: {ts_str} | Mode: {conv['mode']} | Status: {conv['status']} | Lang: {conv['session_language']}"
        
        # Use different colors for ongoing vs completed sessions
        if conv["status"] == "ongoing":
            with st.expander(label, expanded=False):
                st.warning("⚠️ This is an ongoing session (may be incomplete)")
                st.code(conv["data"], language="text")
        else:
            with st.expander(label):
                st.code(conv["data"], language="text")
        st.markdown("---")
    
    if st.button("Back to Menu"):
        st.session_state.pre_done = False
        st.rerun() 