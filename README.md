# 💬 Chatbot template

A simple Streamlit app that shows how to build a chatbot using OpenAI's GPT-3.5.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Set up secrets

   Create `.streamlit/secrets.toml` with the required secrets:
   ```toml
   openai_key = "sk-..."
   firestore_creds = '{"type": "service_account", ...}'
   ```
   
   You can find the secret values in the [Streamlit Cloud dashboard](https://share.streamlit.io/) → Select the app → Settings → Secrets.

3. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
