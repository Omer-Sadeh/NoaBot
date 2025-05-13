import base64
from pathlib import Path

import streamlit as st
from openai import OpenAI
import json
import concurrent.futures
import time

def load_translations(language: str = "en") -> dict:
    """Load translations from a JSON file."""
    file_path = f"prompts/{language}/translations.json"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Translations file not found: {file_path}")
        # Fallback to English if the translations for the selected language are not found
        if language != "en":
            st.warning(f"Falling back to English translations.")
            file_path_en = f"prompts/en/translations.json"
            try:
                with open(file_path_en, "r", encoding="utf-8") as f:
                    return json.load(f)
            except FileNotFoundError:
                st.error(f"English fallback translations file also not found: {file_path_en}")
                return {} # Return empty dict if English fallback also fails
        return {} # Return empty dict if primary language file not found (and it was English)
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from translations file: {file_path}")
        return {}

def tr(key: str, lang: str = None, **kwargs) -> str:
    """Get translated string."""
    if lang is None:
        lang = st.session_state.get("language", "en")
    
    # Ensure translations are loaded for the current language
    if "translations" not in st.session_state or st.session_state.get("translations_lang") != lang:
        st.session_state.translations = load_translations(lang)
        st.session_state.translations_lang = lang

    return st.session_state.translations.get(key, key).format(**kwargs)

def set_page_direction(lang: str = None):
    if lang is None:
        lang = st.session_state.get("language", "en")
    if lang == "he":
        st.markdown(
            """
            <style>
                /* Basic RTL for the whole app */
                body, .main, .stApp, .stChatInputContainer, .stChatMessage, .stButton button, .stTextInput input, .stTextArea textarea {
                    direction: rtl !important;
                    text-align: right !important;
                }
                /* Ensure chat messages content aligns correctly */
                .stChatMessage > div > div {
                    text-align: right !important;
                }
                /* Fix for Streamlit's internal handling of text input, might need adjustments */
                input[type="text"], textarea {
                    direction: rtl !important;
                    text-align: right !important;
                }
                /* Ensure selectbox dropdown aligns correctly */
                .stSelectbox [data-baseweb="select"] > div {
                    direction: ltr !important; /* Keep dropdown arrow on the left for hebrew too, as it's more standard */
                }
                 .stSelectbox [data-baseweb="select"] input {
                    direction: rtl !important;
                    text-align: right !important;
                }
                /* Sidebar needs specific targeting */
                .css-1d391kg { /* This selector might change with streamlit versions, targets sidebar */
                     direction: rtl !important;
                }
                .css-1d391kg * {
                    text-align: right !important;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Explicitly set LTR if not Hebrew to override previous RTL if language is switched
        st.markdown(
            """
            <style>
                body, .main, .stApp, .stChatInputContainer, .stChatMessage, .stButton button, .stTextInput input, .stTextArea textarea {
                    direction: ltr !important;
                    text-align: left !important;
                }
                .stChatMessage > div > div {
                    text-align: left !important;
                }
                input[type="text"], textarea {
                    direction: ltr !important;
                    text-align: left !important;
                }
                 .stSelectbox [data-baseweb="select"] > div {
                    direction: ltr !important;
                }
                 .stSelectbox [data-baseweb="select"] input {
                    direction: ltr !important;
                    text-align: left !important;
                }
                .css-1d391kg {
                     direction: ltr !important;
                }
                .css-1d391kg * {
                    text-align: left !important;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

client = OpenAI(api_key=st.secrets["openai_key"])
async_context = concurrent.futures.ThreadPoolExecutor()

def load_prompt(file_path: str, language: str = "en") -> str:
    full_path = f"prompts/{language}/{file_path}"
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        st.error(f"Prompt file not found: {full_path} for language '{language}'. Trying English fallback.")
        # Fallback to English if the prompt for the selected language is not found
        if language != "en":
            full_path_en = f"prompts/en/{file_path}"
            try:
                with open(full_path_en, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except FileNotFoundError:
                st.error(f"English fallback prompt file also not found: {full_path_en}")
                return f"ERROR: Prompt file '{file_path}' not found for language '{language}' or English."
        return f"ERROR: Prompt file '{file_path}' not found for language '{language}'."

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
        {"role": "assistant", "content": load_prompt(tr("initial_message_prompt_file"), st.session_state.get("language", "en"))}
    ]
    st.session_state.current_stage = 0
    st.session_state.guidelines = load_guidelines(st.session_state.get("language", "en"))
    st.session_state.rounds_since_last_completion = 0
    st.session_state.system_prompt = load_prompt(tr("system_prompt_file"), st.session_state.get("language", "en"))
    st.session_state.done = False
    st.session_state.running = True

    st.session_state.start_time = None
    st.session_state.end_time = None
    st.session_state.step_times = []

    st.session_state.step_user_messages_amount = []

    st.session_state.completed_guidelines = 0

    st.session_state.tips_shown = 0
    st.session_state.sidebar_messages = [] # Initialize sidebar messages

    st.session_state.voice = False
    if "language" not in st.session_state:
        st.session_state.language = "en"

    # Ensure translations are loaded at the start of a session
    if "translations" not in st.session_state or st.session_state.get("translations_lang") != st.session_state.language:
        st.session_state.translations = load_translations(st.session_state.language)
        st.session_state.translations_lang = st.session_state.language

def evaluate_guidelines(state: dict):
    prompt_template = load_prompt(tr("evaluate_guidelines_prompt_file", state.get("language", "en")), state.get("language", "en"))

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
    prompt_template = load_prompt(tr("get_director_tip_prompt_file", state.get("language", "en")), state.get("language", "en"))

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

def add_to_sidebar(content: str, message_type: str):
    """Adds a message to the persistent sidebar_messages list in session_state."""
    if "sidebar_messages" not in st.session_state:
        st.session_state.sidebar_messages = []

    message_id = time.time_ns() # Generate a unique ID for the message
    st.session_state.sidebar_messages.append({'id': message_id, 'type': message_type, 'content': content})

    if message_type == 'tip':
        st.session_state.tips_shown += 1

def remove_sidebar_message(message_id_to_remove):
    """Removes a specific message from the sidebar_messages list by its ID."""
    if "sidebar_messages" in st.session_state:
        st.session_state.sidebar_messages = [
            msg for msg in st.session_state.sidebar_messages if msg.get('id') != message_id_to_remove
        ]
        st.rerun() # Rerun to reflect changes in the sidebar immediately

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
    temp_dir = Path("temp_audio")
    temp_dir.mkdir(exist_ok=True)
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
        st.error(tr("audio_generation_error", st.session_state.get("language", "en"), error=str(e)))
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
        # Reset session state that depends on language
        st.session_state.messages = [
            {"role": "assistant", "content": load_prompt(tr("initial_message_prompt_file"), lang)}
        ]
        st.session_state.guidelines = load_guidelines(lang)
        st.session_state.system_prompt = load_prompt(tr("system_prompt_file"), lang)
        st.session_state.sidebar_messages = [] # Reset sidebar messages on language change
        # Load translations for the new language
        st.session_state.translations = load_translations(lang)
        st.session_state.translations_lang = lang
        st.rerun()

    current_lang = st.session_state.get("language", "en")
    set_page_direction(current_lang) # Apply RTL/LTR styling

    st.sidebar.title(tr("tips_progress", current_lang))
    
    lang_options = {"English": "en", "עברית": "he"}
    # Get current language index for selectbox
    lang_keys = list(lang_options.keys())
    lang_values = list(lang_options.values())
    current_lang_display_name = lang_keys[lang_values.index(current_lang)]

    selected_lang_key = st.sidebar.selectbox(
        tr("language_label", current_lang), # This label is bilingual by design in the selectbox
        options=lang_keys,
        index=lang_keys.index(current_lang_display_name) # Ensure correct display name is selected
    )

    if lang_options[selected_lang_key] != current_lang:
        set_language(lang_options[selected_lang_key])

    # Container for all persistent messages
    message_display_container = st.sidebar.container()

    # Render persistent sidebar messages
    if "sidebar_messages" in st.session_state and st.session_state.sidebar_messages:
        with message_display_container:
            for message_info in st.session_state.sidebar_messages:
                if message_info['type'] == 'tip':
                    col1, col2 = st.columns([0.85, 0.15]) # Adjust column ratios as needed
                    with col1:
                        st.warning(f"{tr('tip_sidebar_prefix', current_lang)} {message_info['content']}")
                    with col2:
                        if st.button("X", key=f"remove_tip_{message_info['id']}", help="Remove this tip"):
                            remove_sidebar_message(message_info['id'])
                            # No need to explicitly call st.rerun() here as remove_sidebar_message does it
                elif message_info['type'] == 'guideline':
                    st.success(message_info['content'])

    st.title(tr("app_title", current_lang))
    st.write(tr("app_subtitle", current_lang))
    st.checkbox(tr("sound_voice_checkbox", current_lang), value=st.session_state.voice, on_change=lambda: st.session_state.update({"voice": not st.session_state.voice}))

    # Create a container for the conversation history
    conversation_container = st.container()

    # Create a container for inputs at the bottom
    input_container = st.container()

    # Handle inputs first (but they'll be rendered later)
    with input_container:
        col1, col2 = st.columns([3, 1])
        with col1:
            placeholder_key = "chat_placeholder_audio" if "audio_data" in st.session_state and st.session_state["audio_data"] is not None else "chat_placeholder_typing"
            prompt = st.chat_input(tr(placeholder_key, current_lang), disabled=("audio_data" in st.session_state and st.session_state["audio_data"] is not None))
        with col2:
            audio_bytes = st.audio_input(tr("audio_input_label", current_lang), key=f"audio_data_{current_lang}") # Ensure key changes with lang to reset if needed

    if len(st.session_state.messages) > 1:
        # Add End Conversation button right after the input container
        st.button(tr("end_conversation_button", current_lang), on_click=end_session)

    # Process any inputs
    if audio_bytes:
        with st.spinner(tr("processing_speech_spinner", current_lang)):
            try:
                # Ensure the audio file object is correctly named for the API
                audio_file_for_transcription = ("audio.wav", audio_bytes, "audio/wav") # Assuming wav, adjust if audio_input gives different format
                prompt = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe", # Consider if a specific model is better for Hebrew
                    file=audio_file_for_transcription,
                    language=current_lang # Pass current language to transcription
                ).text
            except Exception as e:
                st.error(tr("transcription_error", current_lang, error=str(e)))
                prompt = None # Ensure prompt is None if transcription fails

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
                with st.spinner(tr("generating_audio_spinner", current_lang)):
                    audio_file = text_to_speech(response)
                    if audio_file:
                        autoplay_audio(audio_file)
                    else: # Handle case where text_to_speech might return None
                        st.warning(tr("audio_generation_error", current_lang, error="Failed to generate audio file path."))

            guidelines_promise = start_promise(evaluate_guidelines, st.session_state.to_dict())
            tip_promise = start_promise(get_director_tip, st.session_state.to_dict())
            completed_guidelines, new_state = guidelines_promise.result()
            tip = tip_promise.result()

            for key, value in new_state.items():
                if key == 'reset_button':
                    continue
                if key != f"audio_data_{current_lang}":
                    st.session_state[key] = value

            if len(completed_guidelines) > 0:
                st.session_state.rounds_since_last_completion = 0
            else:
                st.session_state.rounds_since_last_completion += 1

            for guideline in completed_guidelines:
                add_to_sidebar(tr("guideline_completed_sidebar", current_lang, guideline=guideline), message_type='guideline')

            if st.session_state.rounds_since_last_completion > 0 and tip: # Ensure tip is not None
                add_to_sidebar(tip, message_type='tip') # Tip content itself is from LLM

            # Force a rerun to update the UI immediately with new chat messages and sidebar content
            st.rerun()

def render_end_screen():
    current_lang = st.session_state.get("language", "en")
    set_page_direction(current_lang) # Apply RTL/LTR styling

    st.title(tr("session_ended_title", current_lang))
    st.write(tr("session_ended_subtitle", current_lang))

    if st.session_state.done:
        st.success(tr("all_guidelines_completed_success", current_lang))
    else:
        st.error(tr("not_all_guidelines_completed_error", current_lang))

    elapsed_time = st.session_state.end_time - st.session_state.start_time
    minutes = int(elapsed_time / 60)
    seconds = int(elapsed_time % 60)
    duration_str = f"{minutes} {tr('minutes_unit', current_lang)} {seconds} {tr('seconds_unit', current_lang)}"
    st.write(tr("conversation_duration_label", current_lang, duration=duration_str))

    user_msgs = len([msg for msg in st.session_state.messages if msg['role'] == 'user'])
    st.write(tr("user_messages_label", current_lang, count=user_msgs))

    st.write(tr("completed_steps_label", current_lang, count=st.session_state.current_stage))

    st.write(tr("completed_criteria_label", current_lang, count=st.session_state.completed_guidelines))

    steps_times_minutes = [int(t / 60) for t in st.session_state.step_times] # Keep as numbers for format
    st.write(tr("time_on_each_step_label", current_lang, times=steps_times_minutes))

    st.write(tr("tips_shown_label", current_lang, count=st.session_state.tips_shown))

    st.write(tr("thank_you_message", current_lang))

    conv_transcript = "\n\n".join([
        f"-- {'Noa' if message['role'] == 'assistant' else 'User'}: {message['content']}"
        for message in st.session_state.messages
    ])

    save_data = f"""\
Completed: {st.session_state.done}\n\
Session Duration: {duration_str}\n\
Number of user messages: {user_msgs}\n\
Number of completed Steps: {st.session_state.current_stage}\n\
Number of Completed Criteria: {st.session_state.completed_guidelines}\n\
Time on Each Step (min): {steps_times_minutes}\n\
Number of tips shown: {st.session_state.tips_shown}\n\
Conversation Transcript: \n\
--------------------------\n\n\
{conv_transcript}\n\n\
--------------------------
"""

    st.download_button(
        tr("save_results_button", current_lang),
        save_data,
        file_name=f"transcript_results_{time.strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

if __name__ == "__main__":
    if "language" not in st.session_state: # Ensure language is set at the very beginning
        st.session_state.language = "en" # Default to English
    if "translations" not in st.session_state or st.session_state.get("translations_lang") != st.session_state.language: # Ensure translations are loaded initially
        st.session_state.translations = load_translations(st.session_state.language)
        st.session_state.translations_lang = st.session_state.language
    if "sidebar_messages" not in st.session_state: # Ensure sidebar_messages is initialized
        st.session_state.sidebar_messages = []

    if "running" not in st.session_state:
        reset_session() # This will now use the language from session_state

    if not st.session_state.running:
        render_end_screen()
    else:
        render_screen()
