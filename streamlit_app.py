import streamlit as st

from openScript import setup_env, render_screen, render_end_screen
from closedScript import setup_env_closed, render_closed_screen

if __name__ == "__main__":
    GAME_MODE = "open"
    LANGUAGE = "he"
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
