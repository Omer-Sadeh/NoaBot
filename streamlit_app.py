import time
import streamlit as st
from google.oauth2 import service_account
from openai import OpenAI
import json
import firebase_admin
from firebase_admin import firestore

# Initialize Firebase
if not firebase_admin._apps:
    cred = service_account.Credentials.from_service_account_info(json.loads(st.secrets["firestore_creds"]))
    try:
        firebase_admin.initialize_app(cred, {'projectId': 'noabotprompts',})
    except ValueError:
        print("App already initialized")
        firebase_admin.initialize_app(cred, {'projectId': 'noabotprompts',}, name='noabotprompts')

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["openai_key"])
db = firestore.client()
prompts = db.collection("prompts")

def reset_session():
    st.session_state.messages = [
        {"role": "assistant", "content": "האמת אתמול בערב ממש התבאסתי על דנה, קבענו להיפגש לקפה אתמול אחרי מלא זמן שלא נפגשנו. ממש קשה לי גם ככה לקבוע תוכניות ולצאת מהבית בתקופה הזו. חצי שעה לפני הזמן שקבענו, אחרי שכבר התארגנתי ובאתי לצאת היא כתבה לי שהיא ממש מצטערת אבל היא לא יכולה, בעלה היה בעבודה או משהו והיא הייתה צריכה לשמור על הילדים. לא משנה, זאת כבר הפעם השלישית שהיא עושה לי את זה. אני כבר לא יודעת מה לחשוב.. "}
    ]

def get_system_prompt(name: str):
    for doc in prompts.stream():
        if doc.to_dict()["name"] == name:
            return doc.to_dict()["prompt"]
    raise ValueError(f"Prompt '{name}' not found in database.")

def sidebar():
    st.sidebar.title("בחירת מערכת תסריטים")

    if "prompt_names" not in st.session_state:
        st.session_state.prompt_names = [doc.to_dict()["name"] for doc in prompts.stream()]

    selected_prompt_name = st.sidebar.selectbox(
        "בחר תסריט", st.session_state.prompt_names, index=0
    )

    if "selected_prompt" not in st.session_state or st.session_state.selected_prompt != selected_prompt_name:
        st.session_state.selected_prompt = selected_prompt_name
        st.session_state.system_prompt = get_system_prompt(selected_prompt_name)
        reset_session()

    st.sidebar.text_area(
        "ערוך תסריט נוכחי",
        st.session_state.system_prompt, key="edited_prompt",
        height=200
    )

    if st.sidebar.button("עדכן תסריט פעיל (לא ישמר)"):
        st.session_state.system_prompt = st.session_state.edited_prompt
        reset_session()

    new_prompt_name = st.sidebar.text_input(
        "בחר שם לתסריט לשמירה",
        selected_prompt_name
    )

    if st.sidebar.button("שמור תסריט חדש", type="secondary", disabled=(new_prompt_name == "omer-base")):
        try:
            if not new_prompt_name or new_prompt_name in st.session_state.prompt_names:
                if not new_prompt_name:
                    new_prompt_name = selected_prompt_name
                prompt_docs = [doc for doc in prompts.stream() if doc.to_dict()["name"] == new_prompt_name]
                if prompt_docs:
                    prompt_doc = prompt_docs[0]
                    db.collection("prompts").document(prompt_doc.id).update({"prompt": st.session_state.edited_prompt})
                    st.sidebar.success(f"Prompt '{new_prompt_name}' updated successfully.")
            else:
                prompts.add({"name": new_prompt_name, "prompt": st.session_state.edited_prompt})
                st.session_state.prompt_names.append(new_prompt_name)
                st.sidebar.success(f"Prompt '{new_prompt_name}' saved successfully.")
        except Exception as e:
            st.sidebar.error(f"Error saving prompt: {e}")

        time.sleep(2)
        st.rerun()

    # Add a delete button with confirmation
    delete_confirmation = st.sidebar.checkbox("אישור מחיקה")
    if st.sidebar.button("מחק תסריט נבחר", type="primary", disabled=(selected_prompt_name == "omer-base")):
        if selected_prompt_name == "omer-base":
            st.sidebar.error("Cannot delete the base prompt.")
        else:
            if delete_confirmation:
                # Fetch the current prompt document
                prompt_docs = [doc for doc in prompts.stream() if doc.to_dict()["name"] == st.session_state.selected_prompt]
                if prompt_docs:
                    prompt_doc = prompt_docs[0]
                    db.collection("prompts").document(prompt_doc.id).delete()
                    st.sidebar.success(f"Prompt '{st.session_state.selected_prompt}' deleted successfully.")
                    # Remove from prompt names and reset session
                    st.session_state.prompt_names.remove(st.session_state.selected_prompt)
                    st.session_state.selected_prompt = ""
                    st.session_state.system_prompt = ""
                    reset_session()
                    time.sleep(2)
                    st.rerun()
            else:
                st.sidebar.warning("Please confirm to delete the prompt.")

def render_screen():
    st.markdown(
        """
        <style>
            .stChatMessage, .stHeading, .stMarkdown, .stChatInput, .stTextInput, .stSelectbox, .stElementContainer {
                text-align: right;
                direction: rtl;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    sidebar()

    st.title("שיחה עם נועה")
    st.write("את/ה המטפלת. נהלי שיחה עם נועה כדי לעזור לה בהתמודדויות שלה.")
    conversation_text = ""

    # Button to download the current conversation
    st.download_button(
        "Save Conversation",
        conversation_text,
        file_name="conversation.txt",
        mime="text/plain"
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        conversation_text += f"{message['role']}: {message['content']}\n"

    if prompt := st.chat_input("מה תרצי לענות?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": st.session_state.system_prompt}] + st.session_state.messages,
            stream=True,
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
        conversation_text += f"user: {prompt}\nassistant: {response}\n"

if __name__ == "__main__":
    render_screen()