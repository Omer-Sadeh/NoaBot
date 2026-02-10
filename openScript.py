import base64
from pathlib import Path

import streamlit as st
from openai import OpenAI
import json
import concurrent.futures
import time
import uuid
import config

from google.oauth2 import service_account
import json
import firebase_admin
from firebase_admin import firestore

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

if not firebase_admin._apps:
    cred = service_account.Credentials.from_service_account_info(json.loads(st.secrets["firestore_creds"]))
    firebase_admin.initialize_app(cred, {'projectId': 'noabotprompts',})
    
client = OpenAI(api_key=st.secrets["openai_key"])
async_context = concurrent.futures.ThreadPoolExecutor()
db = firestore.client()

def check_model_availability():
    """Check if all configured models are available in the OpenAI client"""
    required_models = {
        "BASIC_CHAT_MODEL": config.BASIC_CHAT_MODEL,
        "ADVANCED_REASONING_MODEL": config.ADVANCED_REASONING_MODEL,
        "TRANSCRIPTION_MODEL": config.TRANSCRIPTION_MODEL,
        "TTS_MODEL": config.TTS_MODEL
    }
    
    try:
        available_models = client.models.list()
        available_model_ids = [model.id for model in available_models.data]
        
        missing_models = {}
        for name, model_id in required_models.items():
            if model_id not in available_model_ids:
                missing_models[name] = model_id
        
        return missing_models, available_model_ids
    except Exception as e:
        st.error(f"Failed to fetch available models: {e}")
        return None, None

def render_model_error_screen(missing_models, available_models):
    """Render a screen showing missing models and available alternatives"""
    st.title("⚠️ Model Configuration Error")
    st.error("Some required models are not available in your OpenAI account.")
    
    st.subheader("Missing Models:")
    for config_name, model_id in missing_models.items():
        st.warning(f"**{config_name}**: `{model_id}`")
    
    st.divider()
    
    st.subheader("Available Models:")
    st.write("Here are all models currently available in your OpenAI account:")
    
    chat_models = []
    audio_models = []
    other_models = []
    
    for model_id in sorted(available_models):
        if "gpt" in model_id.lower() or "o1" in model_id or "o3" in model_id:
            if "transcribe" in model_id.lower() or "tts" in model_id.lower():
                audio_models.append(model_id)
            else:
                chat_models.append(model_id)
        else:
            other_models.append(model_id)
    
    if chat_models:
        with st.expander("💬 Chat/Text Models", expanded=True):
            for model in chat_models:
                st.code(model)
    
    if audio_models:
        with st.expander("🎤 Audio Models (Transcription/TTS)", expanded=True):
            for model in audio_models:
                st.code(model)
    
    if other_models:
        with st.expander("🔧 Other Models", expanded=False):
            for model in other_models:
                st.code(model)
    
    st.info("💡 Please update the models in `config.py` to use available models from the list above.")

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

    st.session_state.start_time = None  # Will be set on first user input
    st.session_state.end_time = None
    st.session_state.step_times = []
    st.session_state.step_start_time = None  # Track the start of each step

    st.session_state.step_user_messages_amount = []

    st.session_state.completed_guidelines = 0

    st.session_state.tips_shown = 0
    st.session_state.sidebar_messages = [] # Initialize sidebar messages

    st.session_state.voice = True
    if "language" not in st.session_state:
        st.session_state.language = "en"

    # Add a unique session_id if not already present
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Ensure translations are loaded at the start of a session
    if "translations" not in st.session_state or st.session_state.get("translations_lang") != st.session_state.language:
        st.session_state.translations = load_translations(st.session_state.language)
        st.session_state.translations_lang = st.session_state.language

    # Reset session_saved flag
    st.session_state.session_saved = False

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
        model=config.ADVANCED_REASONING_MODEL,
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

        now = time.time()
        # Use step_start_time to calculate step duration
        if 'step_start_time' in state and state['step_start_time'] is not None:
            step_time = now - state['step_start_time']
        else:
            step_time = 0
        if 'step_times' not in state:
            state['step_times'] = []
        state['step_times'].append(step_time)
        # Update step_start_time for the next stage
        state['step_start_time'] = now

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
        model=config.ADVANCED_REASONING_MODEL,
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

def _mark_tip_for_removal(tip_id: str):
    if "tip_ids_to_remove" not in st.session_state:
        st.session_state.tip_ids_to_remove = set()
    st.session_state.tip_ids_to_remove.add(tip_id)

def end_session():
    st.session_state.running = False
    st.session_state.end_time = time.time()
    if not st.session_state.done:
        user_messages_amount = len([msg for msg in st.session_state.messages if msg['role'] == 'user'])
        for s in st.session_state.step_user_messages_amount:
            user_messages_amount -= s
        st.session_state.step_user_messages_amount.append(user_messages_amount)

        # Calculate the last step time if session ended early
        if st.session_state.step_start_time is not None:
            step_time = time.time() - st.session_state.step_start_time
        else:
            step_time = 0
        st.session_state.step_times.append(step_time)
    
    # Save final session data when manually ended
    save_session_incrementally("completed")

def text_to_speech(input_text):
    temp_dir = Path("temp_audio")
    temp_dir.mkdir(exist_ok=True)
    
    # Clean up old audio files (older than 1 hour) to prevent disk space issues
    try:
        import time as time_module
        current_time = time_module.time()
        for old_file in temp_dir.glob("speech_*.mp3"):
            if current_time - old_file.stat().st_mtime > 3600:  # 1 hour
                old_file.unlink(missing_ok=True)
    except Exception:
        pass  # Cleanup is best-effort, don't fail if it doesn't work
    
    # Use unique filename to avoid race conditions
    output_path = temp_dir / f"speech_{uuid.uuid4()}.mp3"

    try:
        # Use with_options to set timeout for this specific request
        with client.with_options(timeout=30.0).audio.speech.with_streaming_response.create(
                model=config.TTS_MODEL,
                voice="coral",
                input=input_text
        ) as response:
            response.stream_to_file(output_path)
        
        # Verify the file was created and has content
        if not output_path.exists() or output_path.stat().st_size == 0:
            st.error(tr("audio_generation_error", st.session_state.get("language", "en"), error="Audio file was not created or is empty"))
            return None
            
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

# - Main functions -

def setup_env():
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "translations" not in st.session_state or st.session_state.get("translations_lang") != st.session_state.language:
        st.session_state.translations = load_translations(st.session_state.language)
        st.session_state.translations_lang = st.session_state.language
    if "sidebar_messages" not in st.session_state:
        st.session_state.sidebar_messages = []

    if "model_check_done" not in st.session_state:
        st.session_state.model_check_done = False
        st.session_state.models_available = True
        st.session_state.missing_models = {}
        st.session_state.available_models = []

    if not st.session_state.model_check_done:
        missing, available = check_model_availability()
        if missing is None:
            st.error("Unable to verify model availability. Please check your OpenAI API connection.")
            st.stop()
        
        st.session_state.model_check_done = True
        if missing:
            st.session_state.models_available = False
            st.session_state.missing_models = missing
            st.session_state.available_models = available
        else:
            st.session_state.models_available = True

    if not st.session_state.models_available:
        render_model_error_screen(st.session_state.missing_models, st.session_state.available_models)
        st.stop()

    if "running" not in st.session_state:
        reset_session()
    
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

    # Initialize audio iteration count if not present
    if "audio_iteration_count" not in st.session_state:
        st.session_state.audio_iteration_count = 0

    if "tip_ids_to_remove" in st.session_state and st.session_state.tip_ids_to_remove:
        ids_that_were_removed = st.session_state.tip_ids_to_remove.copy()
        st.session_state.sidebar_messages = [
            msg for msg in st.session_state.sidebar_messages 
            if not (msg.get('type') == 'tip' and msg.get('id') in ids_that_were_removed)
        ]
        st.session_state.tip_ids_to_remove.clear()
        st.rerun()

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

    any_tips_removed = False
    if "sidebar_messages" in st.session_state:
        messages_to_keep = []
        for message_info in st.session_state.sidebar_messages:
            if message_info['type'] == 'tip':
                button_key = f"remove_tip_{message_info['id']}"
                if button_key in st.session_state and st.session_state[button_key]:
                    any_tips_removed = True
                    st.session_state[button_key] = False
                    continue 
            messages_to_keep.append(message_info)
        
        if any_tips_removed:
            st.session_state.sidebar_messages = messages_to_keep
            st.rerun()

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
                        # Button click now calls _mark_tip_for_removal via on_click
                        st.button(
                            "X", 
                            key=f"remove_btn_{message_info['id']}", 
                            help="Remove this tip",
                            on_click=_mark_tip_for_removal,
                            args=(message_info['id'],)
                        )
                elif message_info['type'] == 'guideline':
                    st.success(message_info['content'])

    st.title(tr("app_title", current_lang))
    st.write(tr("app_subtitle_open", current_lang))
    st.checkbox(tr("sound_voice_checkbox", current_lang), value=st.session_state.voice, on_change=lambda: st.session_state.update({"voice": not st.session_state.voice}))

    # Create a container for the conversation history
    conversation_container = st.container()

    # Create a container for inputs at the bottom
    input_container = st.container()

    # Handle inputs first (but they'll be rendered later)
    with input_container:
        col1, col2 = st.columns([3, 1])
        with col1:
            audio_input_key = f"audio_data_{current_lang}_{st.session_state.audio_iteration_count}"
            input_locked = st.session_state.get("input_locked", False)
            chat_input_disabled = input_locked or ("pending_audio_transcript" in st.session_state)
            if input_locked:
                placeholder_text_key = "chat_placeholder_processing"
            elif "pending_audio_transcript" in st.session_state:
                placeholder_text_key = "chat_placeholder_audio"
            else:
                placeholder_text_key = "chat_placeholder_typing"
            chat_input_value = st.chat_input(
                tr(placeholder_text_key, current_lang),
                disabled=chat_input_disabled 
            )
        with col2:
            audio_bytes = st.audio_input(
                tr("audio_input_label", current_lang),
                key=audio_input_key, # Dynamic key
                disabled=input_locked # Lock audio input as well
            )

    if len(st.session_state.messages) > 1:
        # Add End Conversation button right after the input container
        st.button(tr("end_conversation_button", current_lang), on_click=end_session)

    # Process any inputs: first audio, then chat input if no audio was processed in this cycle.
    prompt_for_llm = None

    if audio_bytes: # If new audio was provided in this script run
        with st.spinner(tr("processing_speech_spinner", current_lang)):
            try:
                audio_file_for_transcription = ("audio.wav", audio_bytes, "audio/wav")
                transcribed_text = client.audio.transcriptions.create(
                    model=config.TRANSCRIPTION_MODEL, 
                    file=audio_file_for_transcription,
                    language=current_lang 
                ).text
                st.session_state.pending_audio_transcript = transcribed_text
                st.session_state.audio_iteration_count += 1
                st.rerun() # Rerun to use new audio key & process transcript
            except Exception as e:
                st.error(tr("transcription_error", current_lang, error=str(e)))
                # If transcription fails, we might still want to increment and rerun 
                # to clear the audio input, or handle it differently.
                # For now, let's ensure the input is cleared to prevent loops on error.
                st.session_state.audio_iteration_count += 1 
                st.rerun()

    # Check for a pending transcript from a previous audio processing cycle (after rerun)
    if "pending_audio_transcript" in st.session_state:
        prompt_for_llm = st.session_state.pending_audio_transcript
        del st.session_state.pending_audio_transcript
    elif chat_input_value: # If no audio was processed, use chat input value
        prompt_for_llm = chat_input_value

    # --- Two-phase input locking logic ---
    # Phase 1: User submits input, lock and rerun
    if not st.session_state.get("input_locked", False) and prompt_for_llm:
        st.session_state.input_locked = True
        st.session_state.pending_user_input = prompt_for_llm
        st.rerun()

    # Phase 2: If locked and pending input, process and unlock
    if st.session_state.get("input_locked", False) and st.session_state.get("pending_user_input"):
        prompt_for_llm = st.session_state.pending_user_input
        del st.session_state.pending_user_input
        # Process the conversation and display in the conversation container
        with conversation_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            st.session_state.messages.append({"role": "user", "content": prompt_for_llm})
            with st.chat_message("user"):
                st.markdown(prompt_for_llm)
            if st.session_state.start_time is None:
                st.session_state.start_time = time.time()
            if st.session_state.step_start_time is None:
                st.session_state.step_start_time = time.time()
            stream = client.chat.completions.create(
                model=config.BASIC_CHAT_MODEL,
                messages=[{"role": "system", "content": st.session_state.system_prompt}] + st.session_state.messages,
                stream=True,
            )
            with st.chat_message("assistant"):
                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
            guidelines_promise = start_promise(evaluate_guidelines, st.session_state.to_dict())
            tip_promise = start_promise(get_director_tip, st.session_state.to_dict())
            if st.session_state.voice:
                with st.spinner(tr("generating_audio_spinner", current_lang)):
                    audio_file = text_to_speech(response)
                    if audio_file:
                        autoplay_audio(audio_file)
                    else:
                        st.warning(tr("audio_generation_error", current_lang, error="Failed to generate audio file path."))
            completed_guidelines_list, new_state_dict_from_eval = guidelines_promise.result() 
            tip_text = tip_promise.result()
            logic_keys_managed_by_evaluate_guidelines = [
                "guidelines", "current_stage", "step_times", "step_user_messages_amount", "done", "running", "end_time", "completed_guidelines"
            ]
            for key_to_update in logic_keys_managed_by_evaluate_guidelines:
                if key_to_update in new_state_dict_from_eval:
                    st.session_state[key_to_update] = new_state_dict_from_eval[key_to_update]
            if completed_guidelines_list:
                st.session_state.rounds_since_last_completion = 0
            else:
                st.session_state.rounds_since_last_completion += 1
            for guideline in completed_guidelines_list:
                add_to_sidebar(tr("guideline_completed_sidebar", current_lang, guideline=guideline), message_type='guideline')
            if st.session_state.rounds_since_last_completion > 0 and tip_text:
                add_to_sidebar(tip_text, message_type='tip')
            
            # Save session data incrementally after each interaction
            save_session_incrementally("ongoing")
            
            st.session_state.input_locked = False
            st.rerun()

    # If not in phase 2, and not processing, show conversation as usual
    if not st.session_state.get("input_locked", False) and not st.session_state.get("pending_user_input"):
        with conversation_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

def render_end_screen():
    current_lang = st.session_state.get("language", "en")
    set_page_direction(current_lang) # Apply RTL/LTR styling

    st.title(tr("session_ended_title", current_lang))
    st.write(tr("session_ended_subtitle", current_lang))

    if st.session_state.done:
        st.success(tr("all_guidelines_completed_success", current_lang))
    else:
        st.error(tr("not_all_guidelines_completed_error", current_lang))

    elapsed_time = (st.session_state.end_time or 0) - (st.session_state.start_time or 0)
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    duration_str = f"{minutes} {tr('minutes_unit', current_lang)} {seconds} {tr('seconds_unit', current_lang)}"
    st.write(tr("conversation_duration_label", current_lang, duration=duration_str))

    user_msgs = len([msg for msg in st.session_state.messages if msg['role'] == 'user'])
    st.write(tr("user_messages_label", current_lang, count=user_msgs))

    total_criteria = 6
    completed_criteria_str = f"{st.session_state.completed_guidelines}/{total_criteria}"
    st.write(tr("completed_criteria_label", current_lang, count=completed_criteria_str))

    steps_times = st.session_state.step_times if hasattr(st.session_state, 'step_times') else []
    steps_times_str = [f"{int(t // 60)}:{int(t % 60):02d}" for t in steps_times]
    st.write(tr("time_on_each_step_label", current_lang, times=steps_times_str))

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
Number of Completed Criteria: {completed_criteria_str}\n\
Time on Each Step (min:sec): {steps_times_str}\n\
Number of tips shown: {st.session_state.tips_shown}\n\
Conversation Transcript: \n\
--------------------------\n\n\
{conv_transcript}\n\n\
--------------------------\n"""

    # --- Save to Firestore ---
    try:
        session_id = st.session_state.get("session_id")
        if session_id and not st.session_state.get("session_saved", False):
            with st.spinner(tr("autosave_spinner", current_lang)):
                # Use incremental save function for final save
                save_session_incrementally("completed")
                st.session_state.session_saved = True
    except Exception as e:
        st.warning(f"Failed to save session to Firestore: {e}")

    st.download_button(
        tr("save_results_button", current_lang),
        save_data,
        file_name=f"transcript_results_{time.strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

    # --- Add buttons for new conversation and questionnaire ---
    col1, col2 = st.columns(2)
    with col1:
        if st.button(tr("try_again_button", current_lang)):
            reset_session()
            st.rerun()
    with col2:
        st.link_button(
            tr("continue_to_questionnaire_button", current_lang),
            "https://www.surveymonkey.com/r/NP7M559"
        )

def save_session_incrementally(status="ongoing"):
    """Save current session data to Firestore incrementally"""
    try:
        session_id = st.session_state.get("session_id")
        current_lang = st.session_state.get("language", "en")
        
        if not session_id:
            return
            
        # Calculate current session stats
        elapsed_time = 0
        if st.session_state.get("start_time") and status == "completed":
            elapsed_time = (st.session_state.get("end_time") or time.time()) - st.session_state.start_time
        elif st.session_state.get("start_time"):
            elapsed_time = time.time() - st.session_state.start_time
            
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        duration_str = f"{minutes} {tr('minutes_unit', current_lang)} {seconds} {tr('seconds_unit', current_lang)}"
        
        user_msgs = len([msg for msg in st.session_state.messages if msg['role'] == 'user'])
        total_criteria = 6
        completed_criteria_str = f"{st.session_state.get('completed_guidelines', 0)}/{total_criteria}"
        
        steps_times = st.session_state.get('step_times', [])
        steps_times_str = [f"{int(t // 60)}:{int(t % 60):02d}" for t in steps_times]
        
        # Build conversation transcript
        conv_transcript = "\n\n".join([
            f"-- {'Noa' if message['role'] == 'assistant' else 'User'}: {message['content']}"
            for message in st.session_state.messages
        ])
        
        # Determine session completion status (ongoing vs finished)
        session_finished = status == "completed" or st.session_state.get("done", False) or not st.session_state.get("running", True)
        if session_finished:
            status = "completed"
        
        # Determine success status (guidelines met)
        is_successful = st.session_state.get("done", False)
        
        save_data = f"""\
Status: {status}\n\
Completed: {is_successful}\n\
Session Duration: {duration_str}\n\
Number of user messages: {user_msgs}\n\
Number of Completed Criteria: {completed_criteria_str}\n\
Time on Each Step (min:sec): {steps_times_str}\n\
Number of tips shown: {st.session_state.get('tips_shown', 0)}\n\
Current Stage: {st.session_state.get('current_stage', 0)}\n\
Conversation Transcript: \n\
--------------------------\n\n\
{conv_transcript}\n\n\
--------------------------\n"""

        # Ensure parent session document exists
        db.collection("sessions").document(session_id).set({
            "created": firestore.SERVER_TIMESTAMP,
            "last_updated": firestore.SERVER_TIMESTAMP,
            "mode": "open",
            "status": status,
            "language": current_lang
        }, merge=True)
        
        # Use a fixed document ID for ongoing sessions, create new for completed
        doc_id = "current" if status == "ongoing" else f"final_{int(time.time())}"
        
        db.collection("sessions").document(session_id).collection("conversations").document(doc_id).set({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "data": save_data,
            "mode": "open",
            "status": status,
            "is_successful": is_successful,
            "session_finished": session_finished,
            "user_message_count": user_msgs,
            "completed_guidelines": st.session_state.get('completed_guidelines', 0),
            "current_stage": st.session_state.get('current_stage', 0)
        }, merge=True)
        
        # Delete the ongoing session document when saving completed session
        if status == "completed":
            try:
                db.collection("sessions").document(session_id).collection("conversations").document("current").delete()
            except Exception:
                pass  # Ignore if current document doesn't exist
        
        return True
    except Exception as e:
        st.warning(f"Failed to save session incrementally: {e}")
        return False
