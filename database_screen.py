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

def detect_language_from_data(data):
    """Detect language from conversation data for legacy sessions"""
    if not data:
        return "unknown"
    
    # Hebrew language indicators
    hebrew_indicators = [
        # Open session Hebrew indicators
        "דקות",  # minutes in Hebrew
        "שניות",  # seconds in Hebrew  
        "הושלמו",  # completed in Hebrew
        "משך השיחה",  # conversation duration in Hebrew
        "מספר הודעות משתמש",  # number of user messages in Hebrew
        "מספר קריטריונים",  # number of criteria in Hebrew
        "טיפים",  # tips in Hebrew
        "תודה על השתתפותך",  # thank you for participating in Hebrew
        "זמן בכל שלב",  # time on each step in Hebrew
        "שלבים שהושלמו",  # completed steps in Hebrew
        "עברית",  # Hebrew language name
        "-- נועה:",  # Noa in Hebrew transcript format
        "נועה:",  # Noa in Hebrew
        
        # Closed session Hebrew indicators (conversation content)
        "דנה",  # Dana (common name in Hebrew sessions)
        "ברגע האחרון",  # last minute
        "חברות",  # friends/friendship
        "מתבאסתי",  # was bummed/upset
        "קבענו",  # we arranged/planned
        "היפגש",  # to meet
        "מצטערת",  # sorry
        "מבאס",  # bummer/annoying
        "קונפליקט",  # conflict
        "במצבים כאלה",  # in such situations
        "ענית נכון על",  # answered correctly (from closed_stats_message)
        "מתוך",  # out of (from closed_stats_message)
        "ההצעה שלך",  # your suggestion
        "פשרה טובה",  # good compromise
        "חשובה לי",  # important to me
        "בלתי נמנע",  # inevitable
        "התבאסתי",  # I was upset
        "לקבוע תוכניות",  # to make plans
        "לצאת מהבית",  # to leave the house
        "לא יכולה",  # can't/couldn't
        "כבר לא יודעת",  # don't know anymore
    ]
    
    # English language indicators  
    english_indicators = [
        # Open session English indicators
        "minutes",
        "seconds", 
        "Session Duration:",
        "Number of user messages:",
        "Number of Completed Criteria:",
        "Time on Each Step",
        "tips shown:",
        "Thank you for participating",
        "Conversation Duration:",
        "-- Noa:",  # Noa in English transcript format
        
        # Closed session English indicators
        "Number of questions:",
        "Number of correct answers:",
        "Closed Script Completed:",
        "--- Transcript ---",
        "You answered",
        "out of",
        "correctly",
    ]
    
    # Check for Hebrew characters (Unicode range for Hebrew)
    hebrew_char_count = sum(1 for char in data if '\u0590' <= char <= '\u05FF')
    
    # Count occurrences of language indicators
    hebrew_count = sum(1 for indicator in hebrew_indicators if indicator in data)
    english_count = sum(1 for indicator in english_indicators if indicator in data)
    
    # Add weight for Hebrew characters
    if hebrew_char_count > 10:  # If there are Hebrew characters, add weight
        hebrew_count += 5
    
    # Determine language based on which indicators appear more
    if hebrew_count > english_count:
        return "he"
    elif english_count > hebrew_count:
        return "en"
    else:
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
            # Get language from session data, fallback to detecting from conversation data for legacy sessions
            session_language = session_data.get("language", "unknown")
            if session_language == "unknown":
                session_language = detect_language_from_data(data.get("data", ""))
            
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
                "session_language": session_language
            })
    
    # --- FILTERS ---
    st.sidebar.header("Filters")
    mode_filter = st.sidebar.selectbox("Mode", options=["All", "open", "closed"], index=0)
    
    # Multi-select status filter
    status_options = ["ongoing", "success", "no success", "unknown"]
    status_filter = st.sidebar.multiselect("Status", options=status_options, default=status_options)
    
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
            # For closed mode, check if ALL answers were correct
            try:
                total_questions = None
                correct_answers = None
                
                for line in data.split('\n'):
                    if line.startswith("Number of questions:"):
                        total_questions = int(line.split(':')[1].strip())
                    elif line.startswith("Number of correct answers:"):
                        correct_answers = int(line.split(':')[1].strip())
                
                # Success only if all answers were correct
                if total_questions is not None and correct_answers is not None:
                    return correct_answers == total_questions
                
                # Fallback to legacy field if numbers not found
                if "Closed Script Completed: True" in data:
                    return True
                if "Closed Script Completed: False" in data:
                    return False
            except (ValueError, IndexError):
                # If parsing fails, fallback to legacy logic
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
        
        # Multi-select status filter
        if status_filter:  # Only filter if something is selected
            conv_status = conv["status"]
            conv_successful = is_successful(conv)
            
            # Determine effective status
            if conv_status == "ongoing":
                effective_status = "ongoing"
            elif conv_status == "completed" and conv_successful:
                effective_status = "success"
            elif conv_status == "completed" and not conv_successful:
                effective_status = "no success"
            else:
                effective_status = "unknown"
            
            # Skip if effective status not in selected filters
            if effective_status not in status_filter:
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