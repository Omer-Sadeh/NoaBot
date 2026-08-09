from streamlit.testing.v1 import AppTest


def test_menu_offers_analysis_screen():
    app = AppTest.from_file("streamlit_app.py")

    app.run()

    assert app.selectbox[0].options == ["Open Mode", "Closed Mode", "Database", "Analysis"]
