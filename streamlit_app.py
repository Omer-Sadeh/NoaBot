import streamlit as st

from openScript import setup_env, render_screen, render_end_screen
from closedScript import setup_env_closed, render_closed_screen

GAME_MODES = {"Open Mode": "open", "Closed Mode": "closed"}
LANGUAGES = {"English": "en", "עברית": "he"}

def pre_game_menu():
    st.title("NoaBot")
    mode_display = st.selectbox("Choose Mode", list(GAME_MODES.keys()), key="mode_select")
    lang_display = st.selectbox("Choose Language", list(LANGUAGES.keys()), key="lang_select")
    if st.button("Start Game", key="start_btn"):
        st.session_state.game_mode = GAME_MODES[mode_display]
        st.session_state.language = LANGUAGES[lang_display]
        st.session_state.pre_done = True
        st.rerun()

if __name__ == "__main__":
    if not st.session_state.get("pre_done", False):
        pre_game_menu()
    else:
        GAME_MODE = st.session_state.get("game_mode", "open")
        LANGUAGE = st.session_state.get("language", "he")
        st.session_state.language = LANGUAGE
        if GAME_MODE == "open":
            setup_env()
            if not st.session_state.running:
                render_end_screen()
            else:
                render_screen()
        else:
            setup_env_closed()
            render_closed_screen()
