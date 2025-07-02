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
    
    # For legacy sessions, if they exist in the database, they were completed
    # (since old versions only saved at the end)
    # Check for any indication this is a legacy completed session
    if conv.get("mode") == "open":
        # Legacy open mode sessions - look for end screen indicators
        if any(indicator in data for indicator in [
            "Completed:", "Session Duration:", "Number of user messages:", 
            "Number of Completed Criteria:", "Thank you for participating"
        ]):
            return "completed"
    elif conv.get("mode") == "closed":
        # Legacy closed mode sessions - look for final results indicators  
        if any(indicator in data for indicator in [
            "Closed Script Completed:", "Number of questions:", 
            "Number of correct answers:", "--- Transcript ---"
        ]):
            return "completed"
    
    # All legacy sessions in database should be completed
    # (since incremental saving is new)
    return "completed"

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
                "is_successful": data.get("is_successful", None),
                "session_finished": data.get("session_finished", None),
                "session_created": session_data.get("created"),
                "session_language": session_data.get("language", "unknown")
            })
    
    # --- FILTERS ---
    st.sidebar.header("Filters")
    mode_filter = st.sidebar.selectbox("Mode", options=["All", "open", "closed"], index=0)
    
    # Combined status filter
    status_filter = st.sidebar.selectbox("Status", options=["All", "ongoing", "success", "no success", "unknown"], index=0)
    
    # Date range filter
    min_date = None
    max_date = None
    timestamps = [c["timestamp"] for c in conversations if c["timestamp"]]
    if timestamps:
        min_date = min([t.date() if hasattr(t, 'date') else t for t in timestamps])
        max_date = max([t.date() if hasattr(t, 'date') else t for t in timestamps])
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date) if min_date and max_date else (None, None))
    

    
    session_id_filter = st.sidebar.text_input("Session ID contains")
    
    # Language filter (new)
    language_filter = st.sidebar.selectbox("Language", options=["All", "en", "he", "unknown"], index=0)
    
    # --- FILTERING LOGIC ---
    def is_successful(conv):
        # Use new success field if available
        if conv.get("is_successful") is not None:
            return conv["is_successful"]
        
        # Fallback to legacy logic (check data content)
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
        
        # Combined status filter
        if status_filter != "All":
            conv_status = conv["status"]
            conv_successful = is_successful(conv)
            
            if status_filter == "ongoing" and conv_status != "ongoing":
                continue
            elif status_filter == "success" and (conv_status != "completed" or not conv_successful):
                continue
            elif status_filter == "no success" and (conv_status != "completed" or conv_successful):
                continue
            elif status_filter == "unknown" and conv_status != "unknown":
                continue
        
        # Date range filter
        ts = conv["timestamp"]
        if ts and hasattr(ts, 'date'):
            ts_date = ts.date()
            if date_range and isinstance(date_range, tuple) and all(date_range):
                if ts_date < date_range[0] or ts_date > date_range[1]:
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
        
        # Determine if session is finished (for success emoji logic)
        session_is_finished = conv["status"] == "completed"
        
        # Success icon logic (only one icon needed)
        if is_successful(conv):
            success_icon = "✅"  # Successful
        elif session_is_finished:
            success_icon = "❌"  # Finished but unsuccessful
        else:
            success_icon = "⏳"  # Still ongoing or unknown
        
        label = f"{success_icon} Session: `{conv['session_id']}` | Time: {ts_str} | Mode: {conv['mode']} | Status: {conv['status']} | Lang: {conv['session_language']}"
        
        # Use different display for ongoing vs completed sessions
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