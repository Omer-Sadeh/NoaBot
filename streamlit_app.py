import streamlit as st
import _version

from openScript import setup_env, render_screen, render_end_screen
from closedScript import setup_env_closed, render_closed_screen
from database_screen import render_database_screen

GAME_MODES = {"Open Mode": "open", "Closed Mode": "closed"}
LANGUAGES = {"English": "en", "עברית": "he"}

MENU_OPTIONS = ["Open Mode", "Closed Mode", "Database"]

def pre_game_menu():
    st.title("NoaBot")
    menu_choice = st.selectbox("Choose Option", MENU_OPTIONS, key="menu_select")
    lang_display = st.selectbox("Choose Language", list(LANGUAGES.keys()), key="lang_select")
    if st.button("Start", key="start_btn"):
        st.session_state.language = LANGUAGES[lang_display]
        if menu_choice == "Database":
            st.session_state.menu_mode = "database"
        else:
            st.session_state.menu_mode = "game"
            st.session_state.game_mode = GAME_MODES[menu_choice]
        st.session_state.pre_done = True
        st.rerun()

if __name__ == "__main__":
    st.markdown(
        f"""
        <div style="position: fixed; bottom: 10px; right: 10px; opacity: 0.5; font-size: 12px; z-index: 9999; pointer-events: none;">
            v{_version.__version__}
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.get("pre_done", False):
        pre_game_menu()
    else:
        if st.session_state.get("menu_mode") == "database":
            render_database_screen()
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
