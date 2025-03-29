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

BASE_PROMPT = "omer-base-eng"

INPUT_BLOCKED = False

hebrew_prompt = "האמת אתמול בערב ממש התבאסתי על דנה, קבענו להיפגש לקפה אתמול אחרי מלא זמן שלא נפגשנו. ממש קשה לי גם ככה לקבוע תוכניות ולצאת מהבית בתקופה הזו. חצי שעה לפני הזמן שקבענו, אחרי שכבר התארגנתי ובאתי לצאת היא כתבה לי שהיא ממש מצטערת אבל היא לא יכולה, בעלה היה בעבודה או משהו והיא הייתה צריכה לשמור על הילדים. לא משנה, זאת כבר הפעם השלישית שהיא עושה לי את זה. אני כבר לא יודעת מה לחשוב.. "
english_prompt = "Yesterday evening I was really upset with Dana, we had arranged to meet for coffee yesterday after a long time we hadn't met. It's really hard for me to make plans and leave the house during this period. Half an hour before the time we arranged, after I had already arranged and come to leave, she wrote to me that she was really sorry but she couldn't, her husband was at work or something and she had to take care of the kids. Never mind, this is already the third time she's done this to me. I don't know what to think anymore.. "

def is_hebrew():
    return "language" in st.session_state and st.session_state.language == "hebrew"

def reset_session():
    st.session_state.messages = [
        {"role": "assistant", "content": hebrew_prompt if is_hebrew() else english_prompt}
    ]
    st.session_state.current_stage = 0
    st.session_state.guidelines = [
        [
            "Identifying the conflict and initial frustration - Did the therapist recognize and validate Noa’s frustration regarding the conflict?"
        ],
        [
            "Identifying and framing the emotions involved in the conflict - Did the therapist help Noa articulate and frame her emotions instead of rushing to solutions?",
            "Recognizing Noa’s coping or behavioral patterns in conflicts - Did the therapist explore Noa’s typical ways of handling conflicts?",
            "Defining a goal for the conflict - Did the therapist help Noa define a goal for the conversation?"
        ],
        [
            "Practicing assertive communication - Did the therapist prompt Noa to practice formulating an assertive message for the conversation?",
            "Creating a compromise or summary that supports a healthy resolution - Did the therapist assist Noa in drawing conclusions and considering a compromise?"
        ]
    ]
    st.session_state.rounds_since_last_completion = 0

def evaluate_guidelines():
    system = f"You are an expert in human emotions and psychology. \
    I will show you a part of a therapy session with Noa, a patient who is struggling with a conflict. \
    Your goal is to evaluate the therapist's performance according to the following guidelines:"

    current_guidelines = st.session_state.guidelines[st.session_state.current_stage]

    for idx, guideline in enumerate(current_guidelines):
        system += f"\n\n{idx+1}. {guideline}"

    system += f"\n\nMark a guideline as completed if and only if the therapist has completed it, and it has absolutely \
    been a part of the conversation, not indirectly but very obviously. \
    A guideline is completed only if has been attempted by the therapist. \
    If the guideline was completed by Noa with no help from the therapist, \
    please do not mark it as completed."

    therapy_session = ""
    for message in st.session_state.messages:
        therapy_session += f"{'Noa' if message['role'] == 'assistant' else 'Therapist'}: {message['content']}\n\n"

    answer = client.chat.completions.create(
        model="o3-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": therapy_session}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "guidelines_evaluation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "completed_indexes": {
                            "type": "array",
                            "items": {
                                "type": "integer"
                            },
                            "description": "Indexes of the guidelines that were completed by the therapist and Noa."
                        }
                    },
                    "required": ["completed_indexes"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    ).choices[0].message.content

    completed_idxs = json.loads(answer)["completed_indexes"]
    updated_guidelines = []
    completed_guidelines = []

    for idx, guideline in enumerate(current_guidelines):
        if idx + 1 not in completed_idxs:
            updated_guidelines.append(guideline)
        else:
            completed_guidelines.append(guideline.split(" - ")[0])

    st.session_state.guidelines[st.session_state.current_stage] = updated_guidelines

    completed = False

    if len(updated_guidelines) == 0:
        st.session_state.current_stage += 1
        if st.session_state.current_stage >= len(st.session_state.guidelines):
            completed = True

    return completed_guidelines, completed

def get_director_tip():
    system = f"You are an expert in human emotions and psychology. \
    I will show you a part of a therapy session with Noa, a patient who is struggling with a conflict. \
    Your goal is to give the therapist the best tip for improving the therapy session. \
    The therapist has the following guidelines to accomplish:"

    current_guidelines = st.session_state.guidelines[st.session_state.current_stage]

    for idx, guideline in enumerate(current_guidelines):
        system += f"\n\n{idx+1}. {guideline}"

    system += f"\n\nIf the therapist has completely lost direction and is not behaving as a therapist should, \
    please give a tip to the therapist on how to get back on track. \
    Be gentle but direct, and keep it a one short sentence."

    therapy_session = ""
    for message in st.session_state.messages:
        therapy_session += f"{'Noa' if message['role'] == 'assistant' else 'Therapist'}: {message['content']}\n\n"

    answer = client.chat.completions.create(
        model="o3-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": therapy_session}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "tip_for_therapist",
                "schema": {
                    "type": "object",
                    "properties": {
                        "tip": {
                            "type": "string",
                            "description": "The tip for the therapist to improve the therapy session."
                        }
                    },
                    "required": ["tip"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    ).choices[0].message.content

    return json.loads(answer)["tip"]

def update_directions():
    if is_hebrew():
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
    else:
        st.markdown(
            """
            <style>
                .stChatMessage, .stHeading, .stMarkdown, .stChatInput, .stTextInput, .stSelectbox, .stElementContainer {
                    text-align: left;
                    direction: ltr;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

def get_system_prompt(name: str):
    for doc in prompts.stream():
        if doc.to_dict()["name"] == name:
            return doc.to_dict()["prompt"]
    raise ValueError(f"Prompt '{name}' not found in database.")

def sidebar():
    st.sidebar.title("בחירת מערכת תסריטים")

    if "prompt_names" not in st.session_state:
        st.session_state.prompt_names = [doc.to_dict()["name"] for doc in prompts.stream()]

    try:
        selected_prompt_name = st.sidebar.selectbox(
            "בחר תסריט", st.session_state.prompt_names, index=1
        )
    except IndexError:
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

    if st.sidebar.button("שמור תסריט חדש", type="secondary", disabled=(new_prompt_name == BASE_PROMPT)):
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
    if st.sidebar.button("מחק תסריט נבחר", type="primary", disabled=(selected_prompt_name == BASE_PROMPT)):
        if selected_prompt_name == BASE_PROMPT:
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
    global INPUT_BLOCKED

    update_directions()

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

    # button to change the prompt
    if st.button("החלף שפה"):
        if is_hebrew():
            st.session_state.language = "english"
        elif "language" not in st.session_state or st.session_state.language == "english":
            st.session_state.language = "hebrew"
        reset_session()
        update_directions()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        conversation_text += f"{message['role']}: {message['content']}\n"

    if not INPUT_BLOCKED:
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

            INPUT_BLOCKED = True

            completed_guidelines, done = evaluate_guidelines()

            if len(completed_guidelines) > 0:
                st.session_state.rounds_since_last_completion = 0
            else:
                st.session_state.rounds_since_last_completion += 1

            for guideline in completed_guidelines:
                st.success(f"Guideline '{guideline}' completed successfully.")

            if done:
                st.success("All guidelines completed successfully!")

            if st.session_state.rounds_since_last_completion > 0:
                tip = get_director_tip()
                st.warning(f"Tip: {tip}")
                INPUT_BLOCKED = False
            else:
                INPUT_BLOCKED = False

if __name__ == "__main__":
    render_screen()