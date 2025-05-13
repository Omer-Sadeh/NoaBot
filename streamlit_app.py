import base64
from pathlib import Path

import streamlit as st
from openai import OpenAI
import json
import concurrent.futures
import time

client = OpenAI(api_key=st.secrets["openai_key"])
async_context = concurrent.futures.ThreadPoolExecutor()

def load_prompt(file_path: str, language: str = "en") -> str:
    full_path = f"prompts/{language}/{file_path}"
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_guidelines(language: str = "en") -> list:
    file_path = f"prompts/{language}/guidelines.json"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Guidelines file not found: {file_path}")
        return []
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from guidelines file: {file_path}")
        return []

def start_promise(function: callable, *args, **kwargs) -> concurrent.futures.Future:
    global async_context
    return async_context.submit(function, *args, **kwargs)

def reset_session():
    st.session_state.messages = [
        {"role": "assistant", "content": load_prompt("initial_message.txt", st.session_state.get("language", "en"))}
    ]
    st.session_state.current_stage = 0
    st.session_state.guidelines = load_guidelines(st.session_state.get("language", "en"))
    st.session_state.rounds_since_last_completion = 0
    st.session_state.system_prompt = load_prompt("system_prompt_noa.txt", st.session_state.get("language", "en"))
    st.session_state.done = False
    st.session_state.running = True

    st.session_state.start_time = None
    st.session_state.end_time = None
    st.session_state.step_times = []

    st.session_state.step_user_messages_amount = []

    st.session_state.completed_guidelines = 0

    st.session_state.tips_shown = 0

    st.session_state.voice = False
    if "language" not in st.session_state:
        st.session_state.language = "en"

def evaluate_guidelines(state: dict):
    prompt_template = load_prompt("evaluate_guidelines_system.txt", state.get("language", "en"))

    current_guidelines_list = state['guidelines'][state['current_stage']]
    guidelines_section_text = ""
    for idx, guideline_text in enumerate(current_guidelines_list):
        guidelines_section_text += f"\n\n{idx+1}. {guideline_text}"

    system = prompt_template.format(guidelines_section=guidelines_section_text)

    therapy_session = ""
    for message in state['messages']:
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

    for idx, guideline in enumerate(current_guidelines_list):
        if idx + 1 not in completed_idxs:
            updated_guidelines.append(guideline)
        else:
            completed_guidelines.append(guideline.split(" - ")[0])

    state['guidelines'][state['current_stage']] = updated_guidelines

    if len(updated_guidelines) == 0:
        state['current_stage'] += 1

        step_time = time.time() - state['start_time']
        for t in state['step_times']:
            step_time -= t
        state['step_times'].append(step_time)

        user_messages_amount = len([msg for msg in state['messages'] if msg['role'] == 'user'])
        for s in state['step_user_messages_amount']:
            user_messages_amount -= s
        state['step_user_messages_amount'].append(user_messages_amount)

        if state['current_stage'] >= len(state['guidelines']):
            state['done'] = True
            state['running'] = False
            state['end_time'] = time.time()

    state['completed_guidelines'] += len(completed_guidelines)

    return completed_guidelines, state

def get_director_tip(state: dict):
    prompt_template = load_prompt("get_director_tip_system.txt", state.get("language", "en"))

    current_guidelines_list = state['guidelines'][state['current_stage']]
    guidelines_section_text = ""
    for idx, guideline_text in enumerate(current_guidelines_list):
        guidelines_section_text += f"\n\n{idx+1}. {guideline_text}"

    system = prompt_template.format(guidelines_section=guidelines_section_text)

    therapy_session = ""
    for message in state['messages']:
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

def add_to_sidebar(text):
    sidebar = st.sidebar
    info_container = sidebar.container()

    with info_container:
        if text.startswith("Tip:"):
            st.warning(text)
            st.session_state.tips_shown += 1
        else:
            st.success(text)

def end_session():
    st.session_state.running = False
    st.session_state.end_time = time.time()
    if not st.session_state.done:
        user_messages_amount = len([msg for msg in st.session_state.messages if msg['role'] == 'user'])
        for s in st.session_state.step_user_messages_amount:
            user_messages_amount -= s
        st.session_state.step_user_messages_amount.append(user_messages_amount)

        step_time = st.session_state.end_time - st.session_state.start_time
        for t in st.session_state.step_times:
            step_time -= t
        st.session_state.step_times.append(step_time)

def text_to_speech(input_text):
    # Create a temporary directory if it doesn't exist
    temp_dir = Path("temp_audio")
    temp_dir.mkdir(exist_ok=True)

    # Generate a file path for the audio file
    output_path = temp_dir / "speech.mp3"

    try:
        with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="coral",
                input=input_text
        ) as response:
            response.stream_to_file(output_path)
        return output_path
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    audio_base64 = base64.b64encode(audio_bytes).decode()
    md_html = f"""
        <audio autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
    """
    st.markdown(md_html, unsafe_allow_html=True)

def render_screen():
    def set_language(lang):
        st.session_state.language = lang
        reset_session()
        st.rerun()

    st.sidebar.title("Tips & Progress")
    
    lang_options = {"English": "en", "עברית": "he"}
    selected_lang_key = st.sidebar.selectbox(
        "Language / שפה",
        options=list(lang_options.keys()),
        index=list(lang_options.values()).index(st.session_state.language)
    )

    if lang_options[selected_lang_key] != st.session_state.language:
        set_language(lang_options[selected_lang_key])

    st.title("Talk with Noa")
    st.write("You are the therapist, help Noa with her conflict.")
    st.checkbox("Sound voice", value=st.session_state.voice, on_change=lambda: st.session_state.update({"voice": not st.session_state.voice}))

    # Create a container for the conversation history
    conversation_container = st.container()

    # Create a container for inputs at the bottom
    input_container = st.container()

    # Handle inputs first (but they'll be rendered later)
    with input_container:
        col1, col2 = st.columns([3, 1])
        with col1:
            placeholder = "Type your message here..." if "audio_data" not in st.session_state or st.session_state["audio_data"] is None else "Delete recorded audio to enable typing"
            prompt = st.chat_input(placeholder, disabled=("audio_data" in st.session_state and st.session_state["audio_data"] is not None))
        with col2:
            audio_bytes = st.audio_input("Or record your voice...", key=f"audio_data")

    if len(st.session_state.messages) > 1:
        # Add End Conversation button right after the input container
        st.button("End Conversation", on_click=end_session)

    # Process any inputs
    if audio_bytes:
        with st.spinner("Processing your speech..."):
            try:
                prompt = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio_bytes,
                    language=st.session_state.language
                ).text
            except Exception as e:
                st.error(f"An error occurred during transcription: {str(e)}")

    # Process the conversation and display in the conversation container
    with conversation_container:
        # Show existing messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Process new input if available
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            st.session_state.start_time = time.time()

            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": st.session_state.system_prompt}] + st.session_state.messages,
                stream=True,
            )

            with st.chat_message("assistant"):
                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
            if st.session_state.voice:
                with st.spinner("Generating audio..."):
                    audio_file = text_to_speech(response)
                    if audio_file:
                        autoplay_audio(audio_file)

            guidelines_promise = start_promise(evaluate_guidelines, st.session_state.to_dict())
            tip_promise = start_promise(get_director_tip, st.session_state.to_dict())
            completed_guidelines, new_state = guidelines_promise.result()
            tip = tip_promise.result()

            for key, value in new_state.items():
                if key == 'reset_button':
                    continue
                if key != 'audio_data':
                    st.session_state[key] = value

            if len(completed_guidelines) > 0:
                st.session_state.rounds_since_last_completion = 0
            else:
                st.session_state.rounds_since_last_completion += 1

            for guideline in completed_guidelines:
                add_to_sidebar(f"Guideline '{guideline}' completed successfully.")

            if st.session_state.rounds_since_last_completion > 0:
                add_to_sidebar(f"Tip: {tip}")

def render_end_screen():
    st.title("Session Ended")
    st.write("The session has ended. Please refresh the page to start a new session.")

    if st.session_state.done:
        st.success("All guidelines completed successfully!")
    else:
        st.error("Not all guidelines were completed.")

    elapsed_time = st.session_state.end_time - st.session_state.start_time
    elapsed_time_text = f"{int(elapsed_time / 60)} minutes {int(elapsed_time % 60)} seconds"
    st.write(f"Conversation Duration: {elapsed_time_text}")

    user_msgs = len([msg for msg in st.session_state.messages if msg['role'] == 'user'])
    st.write(f"Number of user messages: {user_msgs}")

    st.write(f"Number of completed Steps: {st.session_state.current_stage}")

    st.write(f"Number of Completed Criteria: {st.session_state.completed_guidelines}")

    steps_times = [int(t / 60) for t in st.session_state.step_times]
    st.write(f"Time on Each Step (min): {steps_times}")

    st.write(f"Number of tips shown: {st.session_state.tips_shown}")

    st.write("Thank you for participating in this session!")

    conv_transcript = "\n\n".join([
        f"-- {'Noa' if message['role'] == 'assistant' else 'User'}: {message['content']}"
        for message in st.session_state.messages
    ])

    save_data = f"""\
Completed: {st.session_state.done}\n\
Session Duration: {elapsed_time_text}\n\
Number of user messages: {user_msgs}\n\
Number of completed Steps: {st.session_state.current_stage}\n\
Number of Completed Criteria: {st.session_state.completed_guidelines}\n\
Time on Each Step (min): {steps_times}\n\
Number of tips shown: {st.session_state.tips_shown}\n\
Conversation Transcript: \n\
--------------------------\n\n\
{conv_transcript}\n\n\
--------------------------
"""

    st.download_button(
        "Save Results + Transcript",
        save_data,
        file_name=f"transcript_results_{time.strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

if __name__ == "__main__":
    if "running" not in st.session_state:
        reset_session()

    if not st.session_state.running:
        render_end_screen()
    else:
        render_screen()
