import random
import streamlit as st
import json
import os
from google.oauth2 import service_account
import firebase_admin
from firebase_admin import firestore
import time
import uuid
from streamlit_theme import st_theme
import base64
from pathlib import Path
from openai import OpenAI
import config

if not firebase_admin._apps:
    cred = service_account.Credentials.from_service_account_info(json.loads(st.secrets["firestore_creds"]))
    firebase_admin.initialize_app(cred, {'projectId': 'noabotprompts',})
client = OpenAI(api_key=st.secrets["openai_key"]) 

def load_closed_script(language: str = "en"):
    file_path = f"script/{language}.json"
    if not os.path.exists(file_path):
        file_path = "script/en.json"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def tr(key: str, lang: str = None, **kwargs) -> str:
    if lang is None:
        lang = st.session_state.get("language", "en")
    if "translations" not in st.session_state or st.session_state.get("translations_lang") != lang:
        try:
            with open(f"prompts/{lang}/translations.json", "r", encoding="utf-8") as f:
                st.session_state.translations = json.load(f)
                st.session_state.translations_lang = lang
        except Exception:
            st.session_state.translations = {}
            st.session_state.translations_lang = lang
    return st.session_state.translations.get(key, key).format(**kwargs)

def set_page_direction(lang: str = None):
    if lang is None:
        lang = st.session_state.get("language", "en")
    if lang == "he":
        st.markdown(
            """
            <style>
                body, .main, .stApp, .stButton button, .stTextInput input, .stTextArea textarea {
                    direction: rtl !important;
                    text-align: right !important;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
                body, .main, .stApp, .stButton button, .stTextInput input, .stTextArea textarea {
                    direction: ltr !important;
                    text-align: left !important;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

def setup_env_closed():
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "closed_stage" not in st.session_state:
        st.session_state.closed_stage = 0
    if "closed_feedback" not in st.session_state:
        st.session_state.closed_feedback = None
    if "closed_feedback_color" not in st.session_state:
        st.session_state.closed_feedback_color = None
    if "closed_answered" not in st.session_state:
        st.session_state.closed_answered = False
    if "closed_correct_count" not in st.session_state:
        st.session_state.closed_correct_count = 0
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "session_saved" not in st.session_state:
        st.session_state.session_saved = False
    if "translations" not in st.session_state or st.session_state.get("translations_lang") != st.session_state.language:
        tr("app_title", st.session_state.language)
    if "closed_user_choices" not in st.session_state:
        st.session_state.closed_user_choices = []
    if "voice" not in st.session_state:
        st.session_state.voice = True
    if "input_locked" not in st.session_state:
        st.session_state.input_locked = False
    if "audio_played_stage" not in st.session_state:
        st.session_state.audio_played_stage = -1

def set_language_closed(lang):
    st.session_state.language = lang
    st.session_state.closed_stage = 0
    st.session_state.closed_feedback = None
    st.session_state.closed_feedback_color = None
    st.session_state.closed_answered = False
    st.session_state.closed_correct_count = 0
    st.query_params["language"] = lang
    st.rerun()

def recognize_option_from_text(text: str, options: list[str]) -> int:
    response = client.chat.completions.create(
        model=config.BASIC_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant that recognizes the correct option from a list of options. I will give you a text, and you will reply with the correct index of option best matching the text. The options are: {options}"
            },
            {
                "role": "user",
                "content": text
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "index_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": len(options) - 1
                        },
                    },
                    "required": ["index"],
                    "additionalProperties": False
                }
            }
        }
    )
    response = response.choices[0].message.content
    return json.loads(response)["index"] if json.loads(response)["index"] is not None else random.randint(0, len(options) - 1)

def render_closed_end_screen():
    current_lang = st.session_state.get("language", "en")
    set_page_direction(current_lang)
    script = load_closed_script(current_lang)
    total = sum(1 for entry in script if "correct_answer" in entry)
    correct = st.session_state.get("closed_correct_count", 0)
    st.title(tr("session_ended_title", current_lang))
    st.write(tr("session_ended_subtitle", current_lang))
    st.success(tr("thank_you_message", current_lang))
    st.info(tr("closed_stats_message", current_lang, correct=correct, total=total))

    # --- Firestore Save Logic ---
    if not hasattr(st.session_state, "_firestore_initialized"):
        if not firebase_admin._apps:
            cred = service_account.Credentials.from_service_account_info(json.loads(st.secrets["firestore_creds"]))
            firebase_admin.initialize_app(cred, {'projectId': 'noabotprompts',})
        st.session_state._firestore_initialized = True
    db = firestore.client()

    # Prepare save_data (transcript and stats)
    save_data = f"""\
Closed Script Completed: True\n\
Number of questions: {total}\n\
Number of correct answers: {correct}\n\
--- Transcript ---\n"""
    user_choices = st.session_state.get("closed_user_choices", [])
    for i, entry in enumerate(script):
        noa = entry["Noa"]
        user_choice = user_choices[i] if i < len(user_choices) else "(no answer)"
        if "correct_answer" in entry:
            correct_answer = entry["correct_answer"]
            is_correct = "Yes" if user_choice == correct_answer else "No"
            save_data += f"\nNoa: {noa}\nUser: {user_choice}\nCorrect: {is_correct}\n"
        else:
            save_data += f"\nNoa: {noa}\nUser: {user_choice}\n"
    save_data += "\n--------------------------\n"

    # --- Save to Firestore ---
    try:
        session_id = st.session_state.get("session_id")
        if session_id and not st.session_state.get("session_saved", False):
            with st.spinner(tr("autosave_spinner", current_lang)):
                # Use incremental save function for final save
                save_closed_session_incrementally("completed")
                st.session_state.session_saved = True
            st.success(tr("save_results_button", current_lang) + " - Saved to database!")
        elif st.session_state.get("session_saved", False):
            st.info("Session already saved.")
    except Exception as e:
        st.warning(f"Failed to save session to Firestore: {e}")

    st.download_button(
        tr("save_results_button", current_lang),
        save_data,
        file_name=f"closed_transcript_results_{time.strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

    # --- Add buttons for new conversation and main menu ---
    col1, col2 = st.columns(2)
    with col1:
        if st.button(tr("try_again_button", current_lang)):
            st.session_state.closed_stage = 0
            st.session_state.closed_feedback = None
            st.session_state.closed_feedback_color = None
            st.session_state.closed_answered = False
            st.session_state.closed_correct_count = 0
            st.session_state.session_saved = False
            st.rerun()
    with col2:
        if st.button(tr("back_to_menu_button", current_lang)):
            session_id = st.session_state.get("session_id")
            st.session_state.clear()
            st.session_state.session_id = session_id
            st.session_state.pre_done = False
            st.session_state.url_params_processed = False
            st.query_params.clear()
            st.rerun()

def render_closed_screen():
    theme = st_theme()
    # Sync language from query params if needed
    if "language" in st.query_params:
        if st.session_state.get("language") != st.query_params["language"]:
            st.session_state.language = st.query_params["language"]
    current_lang = st.session_state.get("language", "en")
    set_page_direction(current_lang)
    script = load_closed_script(current_lang)
    stage = st.session_state.get("closed_stage", 0)
    if stage >= len(script):
        render_closed_end_screen()
        return
    # Add sound voice checkbox
    st.checkbox(tr("sound_voice_checkbox", current_lang), value=st.session_state.voice, on_change=lambda: st.session_state.update({"voice": not st.session_state.voice}))
    # After rerun, recalculate current_lang and script
    current_lang = st.session_state.get("language", "en")
    script = load_closed_script(current_lang)
    stage = st.session_state.get("closed_stage", 0)
    entry = script[stage]
    # --- UI rendering starts here ---
    st.title(tr("app_title", current_lang))
    st.write(tr("app_subtitle", current_lang))
    lang_options = {"English": "en", "עברית": "he"}
    lang_keys = list(lang_options.keys())
    lang_values = list(lang_options.values())
    try:
        current_lang_display_name = lang_keys[lang_values.index(current_lang)]
    except ValueError:
        current_lang_display_name = lang_keys[0]
    try:
        selectbox_index = lang_keys.index(current_lang_display_name)
    except ValueError:
        selectbox_index = 0
    selected_lang_key = st.sidebar.selectbox(
        tr("language_label", current_lang),
        options=lang_keys,
        index=selectbox_index,
        key="closed_lang_select"
    )
    if lang_options[selected_lang_key] != current_lang:
        set_language_closed(lang_options[selected_lang_key])
        st.stop()
    entry = script[stage]
    st.header("Noa:")
    st.write(entry["Noa"])
    # --- Audio logic (AFTER Noa's text, BEFORE any button rendering) ---
    if st.session_state.voice and not st.session_state.get("closed_answered", False):
        if st.session_state.get("audio_played_stage", -1) != stage:
            with st.spinner(tr("generating_audio_spinner", current_lang)):
                audio_file = text_to_speech(entry["Noa"])
                if audio_file:
                    autoplay_audio(audio_file)
                    # Mark this stage as played so we don't regenerate audio on next rerun
                    st.session_state.audio_played_stage = stage
                else:
                    st.warning(tr("audio_generation_error", current_lang, error="Failed to generate audio file path."))
                    # Mark as played even on failure to avoid retry loop
                    st.session_state.audio_played_stage = stage

    # Only show thank you/stats/continue if on last question and NOT answered yet
    if stage == len(script) - 1 and not st.session_state.get("closed_answered", False):
        st.success(tr("thank_you_message", current_lang))
        total_questions = sum(1 for entry in script if "correct_answer" in entry)
        st.info(tr("closed_stats_message", current_lang, correct=st.session_state.get("closed_correct_count", 0), total=total_questions))
        if st.button(tr("continue_button", current_lang), key=f"cont_{stage}"):
            st.session_state.closed_stage += 1
            st.session_state.closed_feedback = None
            st.session_state.closed_answered = False
            st.session_state.closed_selected_key = None
            st.session_state.closed_selected_idx = None
            st.session_state.closed_correct_idx = None
            st.session_state.audio_played_stage = -1
            st.rerun()
        return
    answers = [
        (entry["correct_answer"], "correct", entry["correct_answer_feedback"]),
        (entry["incorrect_answer_1"], "incorrect1", entry["incorrect_answer_1_feedback"]),
        (entry["incorrect_answer_2"], "incorrect2", entry["incorrect_answer_2_feedback"]),
    ]
    import random
    random.seed(stage)
    random.shuffle(answers)
    # --- AUDIO OPTION LOGIC ---
    if "audio_iteration_count" not in st.session_state:
        st.session_state.audio_iteration_count = 0
    audio_input_key = f"audio_data_{current_lang}_{st.session_state.audio_iteration_count}"
    input_locked = st.session_state.get("input_locked", False)
    audio_bytes = None
    audio_option_label = tr("audio_input_label", current_lang)
    # Only allow answering if not already answered
    if not st.session_state.get("closed_answered", False):
        # Render the three answer buttons, each on its own row
        for idx, (ans, key, feedback) in enumerate(answers):
            if st.button(ans, key=f"ans_{stage}_{idx}"):
                st.session_state.closed_selected_key = key
                st.session_state.closed_feedback = feedback
                st.session_state.closed_answered = True
                st.session_state.closed_selected_idx = idx
                st.session_state.closed_correct_idx = [i for i, (_, k, _) in enumerate(answers) if k == "correct"][0]
                if key == "correct":
                    st.session_state.closed_correct_count = st.session_state.get("closed_correct_count", 0) + 1
                if len(st.session_state.closed_user_choices) <= stage:
                    st.session_state.closed_user_choices.append(ans)
                else:
                    st.session_state.closed_user_choices[stage] = ans
                
                # Save session data incrementally after each answer
                save_closed_session_incrementally("ongoing")
                
                if stage == len(script) - 1:
                    st.session_state.closed_stage += 1
                    st.session_state.closed_feedback = None
                    st.session_state.closed_answered = False
                    st.session_state.closed_selected_key = None
                    st.session_state.closed_selected_idx = None
                    st.session_state.closed_correct_idx = None
                    st.session_state.audio_played_stage = -1
                    st.rerun()
                else:
                    st.session_state.audio_played_stage = -1
                    st.rerun()
        # Render the audio input below the answer buttons
        audio_bytes = st.audio_input(audio_option_label, key=audio_input_key, disabled=input_locked)
        # Handle audio input
        if audio_bytes:
            st.session_state.input_locked = True
            with st.spinner(tr("processing_speech_spinner", current_lang)):
                try:
                    audio_file_for_transcription = ("audio.wav", audio_bytes, "audio/wav")
                    transcript = client.audio.transcriptions.create(
                        model=config.TRANSCRIPTION_MODEL,
                        file=audio_file_for_transcription,
                        language=current_lang
                    ).text
                    # Recognize which option was said
                    idx = recognize_option_from_text(transcript, [ans for ans, _, _ in answers])
                    if idx < 0 or idx > 2:
                        st.warning(tr("transcription_error", current_lang, error="Could not recognize a valid option from your speech."))
                        st.session_state.input_locked = False
                        st.session_state.audio_iteration_count += 1
                        st.rerun()
                    else:
                        ans, key, feedback = answers[idx]
                        st.session_state.closed_selected_key = key
                        st.session_state.closed_feedback = feedback
                        st.session_state.closed_answered = True
                        st.session_state.closed_selected_idx = idx
                        st.session_state.closed_correct_idx = [i for i, (_, k, _) in enumerate(answers) if k == "correct"][0]
                        if key == "correct":
                            st.session_state.closed_correct_count = st.session_state.get("closed_correct_count", 0) + 1
                        if len(st.session_state.closed_user_choices) <= stage:
                            st.session_state.closed_user_choices.append(ans)
                        else:
                            st.session_state.closed_user_choices[stage] = ans
                        
                        # Save session data incrementally after each audio answer
                        save_closed_session_incrementally("ongoing")
                        
                        st.session_state.input_locked = False
                        st.session_state.audio_iteration_count += 1
                        if stage == len(script) - 1:
                            st.session_state.closed_stage += 1
                            st.session_state.closed_feedback = None
                            st.session_state.closed_answered = False
                            st.session_state.closed_selected_key = None
                            st.session_state.closed_selected_idx = None
                            st.session_state.closed_correct_idx = None
                            st.session_state.audio_played_stage = -1
                            st.rerun()
                        else:
                            st.session_state.audio_played_stage = -1
                            st.rerun()
                except Exception as e:
                    st.error(tr("transcription_error", current_lang, error=str(e)))
                    st.session_state.input_locked = False
                    st.session_state.audio_iteration_count += 1
                    st.rerun()
    else:
        selected_idx = st.session_state.get("closed_selected_idx")
        correct_idx = st.session_state.get("closed_correct_idx")
        # Use theme-aware colors for borders and backgrounds
        if theme and theme.get('base') == 'dark':
            bg_color = theme.get('secondaryBackgroundColor', '#000000')
            correct_border = '#34a853'
            incorrect_border = '#ea4335'
            default_border = '#cccccc'
            correct_bg = 'rgba(52, 168, 83, 0.2)'  # Faded green background
            incorrect_bg = 'rgba(234, 67, 53, 0.2)'  # Faded red background
            default_text = 'gray'
        else:
            bg_color = theme.get('secondaryBackgroundColor', '#fff')
            correct_border = '#34a853'
            incorrect_border = '#ea4335'
            default_border = '#cccccc'
            correct_bg = 'rgba(52, 168, 83, 0.15)'  # Faded green background
            incorrect_bg = 'rgba(234, 67, 53, 0.15)'  # Faded red background
            default_text = 'gray'
        
        for idx, (ans, key, feedback) in enumerate(answers):
            if idx == correct_idx:
                st.markdown(f'<div style="background-color:{correct_bg};border:1px solid {correct_border};padding:8px;border-radius:8px;margin-bottom:4px;">{ans}</div>', unsafe_allow_html=True)
            elif idx == selected_idx:
                st.markdown(f'<div style="background-color:{incorrect_bg};border:1px solid {incorrect_border};padding:8px;border-radius:8px;margin-bottom:4px;">{ans}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color:{default_text};background-color:{bg_color};border:1px solid {default_border};padding:8px;border-radius:8px;margin-bottom:4px;">{ans}</div>', unsafe_allow_html=True)
        if st.session_state.closed_feedback:
            st.info(st.session_state.closed_feedback)
        if stage != len(script) - 1:
            if st.button(tr("continue_button", current_lang), key=f"cont_{stage}"):
                st.session_state.closed_stage += 1
                st.session_state.closed_feedback = None
                st.session_state.closed_answered = False
                st.session_state.closed_selected_key = None
                st.session_state.closed_selected_idx = None
                st.session_state.closed_correct_idx = None
                st.session_state.audio_played_stage = -1
                st.rerun()

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

def save_closed_session_incrementally(status="ongoing"):
    """Save current closed session data to Firestore incrementally"""
    try:
        session_id = st.session_state.get("session_id")
        current_lang = st.session_state.get("language", "en")
        
        if not session_id:
            return
            
        # Initialize Firestore if needed
        if not hasattr(st.session_state, "_firestore_initialized"):
            if not firebase_admin._apps:
                cred = service_account.Credentials.from_service_account_info(json.loads(st.secrets["firestore_creds"]))
                firebase_admin.initialize_app(cred, {'projectId': 'noabotprompts',})
            st.session_state._firestore_initialized = True
        db = firestore.client()
        
        script = load_closed_script(current_lang)
        total = sum(1 for entry in script if "correct_answer" in entry)
        correct = st.session_state.get("closed_correct_count", 0)
        current_stage = st.session_state.get("closed_stage", 0)
        
        # Determine session completion status (ongoing vs finished)
        session_finished = status == "completed" or current_stage >= len(script)
        if session_finished:
            status = "completed"
            
        # Determine success status (all answers correct)
        is_successful = session_finished and correct == total

        # Prepare save_data (transcript and stats)
        save_data = f"""\
Status: {status}\n\
Closed Script Completed: {is_successful}\n\
Number of questions: {total}\n\
Number of correct answers: {correct}\n\
Current Stage: {current_stage}\n\
--- Transcript ---\n"""
        
        user_choices = st.session_state.get("closed_user_choices", [])
        for i, entry in enumerate(script):
            if i >= current_stage and not session_finished:
                break  # Don't include future questions for ongoing sessions
            noa = entry["Noa"]
            user_choice = user_choices[i] if i < len(user_choices) else "(no answer)"
            if "correct_answer" in entry:
                correct_answer = entry["correct_answer"]
                is_correct = "Yes" if user_choice == correct_answer else "No"
                save_data += f"\nNoa: {noa}\nUser: {user_choice}\nCorrect: {is_correct}\n"
            else:
                save_data += f"\nNoa: {noa}\nUser: {user_choice}\n"
        save_data += "\n--------------------------\n"

        # Ensure parent session document exists
        db.collection("sessions").document(session_id).set({
            "created": firestore.SERVER_TIMESTAMP,
            "last_updated": firestore.SERVER_TIMESTAMP,
            "mode": "closed",
            "status": status,
            "language": current_lang
        }, merge=True)
        
        # Use a fixed document ID for ongoing sessions, create new for completed
        doc_id = "current" if status == "ongoing" else f"final_{int(time.time())}"
        
        db.collection("sessions").document(session_id).collection("conversations").document(doc_id).set({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "data": save_data,
            "mode": "closed",
            "status": status,
            "is_successful": is_successful,
            "session_finished": session_finished,
            "current_stage": current_stage,
            "total_questions": total,
            "correct_answers": correct
        }, merge=True)
        
        # Delete the ongoing session document when saving completed session
        if status == "completed":
            try:
                db.collection("sessions").document(session_id).collection("conversations").document("current").delete()
            except Exception:
                pass  # Ignore if current document doesn't exist
        
        return True
    except Exception as e:
        st.warning(f"Failed to save closed session incrementally: {e}")
        return False
