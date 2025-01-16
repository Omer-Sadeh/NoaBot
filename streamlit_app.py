import streamlit as st
from google.oauth2 import service_account
from openai import OpenAI
from google.cloud import firestore
import json

client = OpenAI(api_key=st.secrets["openai_key"])
db = firestore.Client(credentials=service_account.Credentials.from_service_account_info(json.loads(st.secrets["firestore_creds"])))
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

def render_screen():
    st.markdown(
        """
        <style>
            .stChatMessage, .stHeading, .stMarkdown, .stChatInput {
                text-align: right;
                direction: rtl;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("שיחה עם נועה")
    st.write("את/ה המטפלת. נהלי שיחה עם נועה כדי לעזור לה בהתמודדויות שלה.")

    if "messages" not in st.session_state:
        reset_session()

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("מה תרצי לענות?"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": get_system_prompt("omer-base")}] +
            [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

        # Stream the response to the chat using `st.write_stream`, then store it in
        # session state.
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    render_screen()
