import streamlit as st
import json
import os

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
    if "translations" not in st.session_state or st.session_state.get("translations_lang") != st.session_state.language:
        tr("app_title", st.session_state.language)

def set_language_closed(lang):
    st.session_state.language = lang
    st.session_state.closed_stage = 0
    st.session_state.closed_feedback = None
    st.session_state.closed_feedback_color = None
    st.session_state.closed_answered = False
    st.session_state.closed_correct_count = 0
    st.query_params["language"] = lang
    st.rerun()

def render_closed_screen():
    # Sync language from query params if needed
    if "language" in st.query_params:
        if st.session_state.get("language") != st.query_params["language"]:
            st.session_state.language = st.query_params["language"]
    current_lang = st.session_state.get("language", "en")
    set_page_direction(current_lang)
    script = load_closed_script(current_lang)
    stage = st.session_state.get("closed_stage", 0)
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
    # After rerun, recalculate current_lang and script
    current_lang = st.session_state.get("language", "en")
    script = load_closed_script(current_lang)
    stage = st.session_state.get("closed_stage", 0)
    if stage >= len(script):
        total = len(script)
        correct = st.session_state.get("closed_correct_count", 0)
        st.success(tr("thank_you_message", current_lang))
        st.info(tr("closed_stats_message", current_lang, correct=correct, total=total))
        return
    entry = script[stage]
    st.header("Noa:")
    st.write(entry["Noa"])
    if stage == len(script) - 1:
        total = len(script)
        correct = st.session_state.get("closed_correct_count", 0)
        st.success(tr("thank_you_message", current_lang))
        st.info(tr("closed_stats_message", current_lang, correct=correct, total=total))
        return
    answers = [
        (entry["correct_answer"], "correct", entry["correct_answer_feedback"]),
        (entry["incorrect_answer_1"], "incorrect1", entry["incorrect_answer_1_feedback"]),
        (entry["incorrect_answer_2"], "incorrect2", entry["incorrect_answer_2_feedback"]),
    ]
    import random
    random.seed(stage)
    random.shuffle(answers)
    if not st.session_state.get("closed_answered", False):
        for idx, (ans, key, feedback) in enumerate(answers):
            if st.button(ans, key=f"ans_{stage}_{idx}"):
                st.session_state.closed_selected_key = key
                st.session_state.closed_feedback = feedback
                st.session_state.closed_answered = True
                st.session_state.closed_selected_idx = idx
                st.session_state.closed_correct_idx = [i for i, (_, k, _) in enumerate(answers) if k == "correct"][0]
                # Track correct answers
                if key == "correct":
                    st.session_state.closed_correct_count = st.session_state.get("closed_correct_count", 0) + 1
                st.rerun()
    else:
        # Show all answers, marking the correct one in green, selected in red if incorrect
        selected_idx = st.session_state.get("closed_selected_idx")
        correct_idx = st.session_state.get("closed_correct_idx")
        for idx, (ans, key, feedback) in enumerate(answers):
            if idx == correct_idx:
                st.markdown(f'<div style="background-color:#000000;border:1px solid #34a853;padding:8px;border-radius:8px;margin-bottom:4px;">{ans}</div>', unsafe_allow_html=True)
            elif idx == selected_idx:
                st.markdown(f'<div style="background-color:#000000;border:1px solid #ea4335;padding:8px;border-radius:8px;margin-bottom:4px;">{ans}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color:gray;background-color:#000000;border:1px solid #cccccc;padding:8px;border-radius:8px;margin-bottom:4px;">{ans}</div>', unsafe_allow_html=True)
        # Show feedback for the selected answer
        if st.session_state.closed_feedback:
            st.info(st.session_state.closed_feedback)
        # Continue button
        if st.button(tr("continue_button", current_lang), key=f"cont_{stage}"):
            st.session_state.closed_stage += 1
            st.session_state.closed_feedback = None
            st.session_state.closed_answered = False
            st.session_state.closed_selected_key = None
            st.session_state.closed_selected_idx = None
            st.session_state.closed_correct_idx = None
            st.rerun() 