"""
Centralized configuration for OpenAI models used across the application.
"""

# --- Available OpenAI Models (Reference) ---
# https://platform.openai.com/docs/models
#
# Text Generation / Chat:
# - gpt-5.1          : The best model for coding and agentic tasks with configurable reasoning effort.
# - gpt-5.1-mini     : A faster, cost-efficient version of GPT-5 for well-defined tasks.
# - gpt-5.1-nano     : Fastest, most cost-efficient version of GPT-5.
# - gpt-5-pro        : Version of GPT-5 that produces smarter and more precise responses.
# - gpt-5            : Previous intelligent reasoning model for coding and agentic tasks with configurable reasoning effort.
# - gpt-4.1          : Smartest non-reasoning model.
# - gpt-4o           : High-intelligence flagship model for complex, multi-step tasks.
# - gpt-4o-mini      : Small, fast, cost-efficient model for simple tasks.
# - o1-preview       : Reasoning model for hard problems.
# - o1-mini          : Faster, cheaper reasoning model.
# - o3-mini          : (Specific model version if applicable/available to your org)
# - o3-deep-research : Deep research model for complex, multi-step tasks.
# - gpt-3.5-turbo    : Legacy fast model.
#
# Audio Transcription:
# - gpt-4o-transcribe
# - gpt-4o-mini-transcribe
# - gpt-4o-transcribe-diarize
#
# Text to Speech (TTS):
# - gpt-4o-mini-tts    : The Standard text-to-speech model.    
# - gpt-4o-mini-tts-hd : High definition text-to-speech model.
# - gpt-4o-mini-tts-1  : Another text-to-speech model.

# Used for:
# 1. closed script: recognizing user intent from options
# 2. open script: Main conversation loop (generating Noa's responses)
BASIC_CHAT_MODEL = "gpt-4o-mini"

# Used for:
# 1. open script: Checking if the user met criteria
# 2. open script: Generating advice for the user
ADVANCED_REASONING_MODEL = "o3-mini"

# Used for:
# 1. closed script: Audio transcription
# 2. open script: Audio transcription
TRANSCRIPTION_MODEL = "gpt-4o-transcribe"

# Used for:
# 1. closed script: Text-to-speech generation
# 2. open script: Text-to-speech generation
TTS_MODEL = "gpt-4o-mini-tts"
