import streamlit as st
import csv
import io
import time
from firebase_admin import firestore
import datetime
import config
from conversation_csv import CSV_FIELDNAMES
from conversation_data import (
    effective_status,
    filter_attempts,
    is_successful,
    load_attempts,
    render_filters,
    setup_firestore as shared_setup_firestore,
)
FORMULA_PREFIXES = ("=", "+", "-", "@")


def csv_cell(value):
    if value is None:
        return ""

    if isinstance(value, (datetime.datetime, datetime.date)):
        value = value.isoformat()
    else:
        value = str(value)

    if value.startswith(FORMULA_PREFIXES):
        return f"'{value}"

    return value


def conversations_to_csv(conversations):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {
            fieldname: csv_cell(conversation.get(fieldname))
            for fieldname in CSV_FIELDNAMES
        }
        for conversation in conversations
    )
    return output.getvalue().encode("utf-8-sig")


def backfill_sessions(db, collection_name):
    # Get all session IDs from conversations subcollections
    sessions_ref = db.collection(collection_name)
    conv_session_ids = set()
    for session_ref in sessions_ref.list_documents():
        conv_session_ids.add(session_ref.id)
    # For each session_id, check if parent exists
    missing = []
    for session_id in conv_session_ids:
        doc_ref = db.collection(collection_name).document(session_id)
        if not doc_ref.get().exists:
            missing.append(session_id)
            doc_ref.set({"created": firestore.SERVER_TIMESTAMP}, merge=True)
    return missing

def render_database_screen():
    st.title("Saved Conversations Database")
    db = shared_setup_firestore()
    collection_name = config.get_variant(st.session_state.get("variant"))["collection"]
    # Backfill missing parent session documents automatically
    backfill_sessions(db, collection_name)
    conversations = load_attempts(collection_name)
    
    st.download_button(
        "Download all data (CSV)",
        data=conversations_to_csv(conversations),
        file_name=f"conversations_{collection_name}_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help=f"Download all {len(conversations)} conversations in the active collection.",
    )

    filters = render_filters(conversations, key_prefix="database")
    filtered = filter_attempts(conversations, filters)
    
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
        
        label = f"{success_icon} Attempt: `{conv['session_id']}/{conv['doc_id']}` | Time: {ts_str} | Mode: {conv['mode']} | Status: {effective_status(conv)} | Lang: {conv['session_language']}"
        
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