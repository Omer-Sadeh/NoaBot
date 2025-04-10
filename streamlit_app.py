import streamlit as st
from openai import OpenAI
import json
import concurrent.futures

client = OpenAI(api_key=st.secrets["openai_key"])
async_context = concurrent.futures.ThreadPoolExecutor()

def start_promise(function: callable, *args, **kwargs) -> concurrent.futures.Future:
    global async_context
    return async_context.submit(function, *args, **kwargs)

def reset_session():
    st.session_state.messages = [
        {"role": "assistant", "content": "Yesterday evening I was really upset with Dana, we had arranged to meet for coffee yesterday after a long time we hadn't met. It's really hard for me to make plans and leave the house during this period. Half an hour before the time we arranged, after I had already arranged and come to leave, she wrote to me that she was really sorry but she couldn't, her husband was at work or something and she had to take care of the kids. Never mind, this is already the third time she's done this to me. I don't know what to think anymore.. "}
    ]
    st.session_state.current_stage = 0
    st.session_state.guidelines = [
        [
            "Identifying the conflict and initial frustration - Did the therapist recognize and validate Noa’s frustration regarding the conflict?"
        ],
        [
            "Identifying and framing the emotions involved in the conflict - Did the therapist help Noa articulate and frame her emotions instead of rushing to solutions?",
            "Recognizing Noa’s coping or behavioral patterns in conflicts - Did the therapist explore Noa’s typical ways of handling conflicts?",
            "Defining a goal for the conflict - Did the therapist help Noa define a goal for the conversation?"
        ],
        [
            "Practicing assertive communication - Did the therapist prompt Noa to practice formulating an assertive message for the conversation?",
            "Creating a compromise or summary that supports a healthy resolution - Did the therapist assist Noa in drawing conclusions and considering a compromise?"
        ]
    ]
    st.session_state.rounds_since_last_completion = 0
    st.session_state.system_prompt = "You are a patient named Noa, who struggles with difficulties in dealing with conflicts with other people. You hate conflicts, in which you only do one of two options: start arguing and shouting at the other person, or leave the place, ignore, repress. Sometimes there are situations where you simply prefer to do what the other side wants even if it doesn't suit you. You are very sensitive to criticism. Your husband's name is Dor. Your good friend's name is Dana, and she has children which sometimes means she doesn't have time for you. The situation is a conversation with your psychologist. I am the psychologist and you are Noa. You should only answer Noa's responses and nothing else. You should be as responsive as possible to what the therapist said, while maintaining Noa's character and coping style. You are not the therapist, and you don't ask questions to the therapist, only respond with things related to yourself."

def evaluate_guidelines(state: dict):
    system = f"You are an expert in human emotions and psychology. \
    I will show you a part of a therapy session with Noa, a patient who is struggling with a conflict. \
    Your goal is to evaluate the therapist's performance according to the following guidelines:"

    current_guidelines = state['guidelines'][state['current_stage']]

    for idx, guideline in enumerate(current_guidelines):
        system += f"\n\n{idx+1}. {guideline}"

    system += f"\n\nMark a guideline as completed if and only if the therapist has completed it, and it has absolutely \
    been a part of the conversation, not indirectly but very obviously. \
    A guideline is completed only if has been attempted by the therapist. \
    If the guideline was completed by Noa with no help from the therapist, \
    please do not mark it as completed."

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

    for idx, guideline in enumerate(current_guidelines):
        if idx + 1 not in completed_idxs:
            updated_guidelines.append(guideline)
        else:
            completed_guidelines.append(guideline.split(" - ")[0])

    state['guidelines'][state['current_stage']] = updated_guidelines

    completed = False

    if len(updated_guidelines) == 0:
        state['current_stage'] += 1
        if state['current_stage'] >= len(state['guidelines']):
            completed = True

    return completed_guidelines, completed, state

def get_director_tip(state: dict):
    system = f"You are an expert in human emotions and psychology. \
    I will show you a part of a therapy session with Noa, a patient who is struggling with a conflict. \
    Your goal is to give the therapist the best tip for improving the therapy session. \
    The therapist has the following guidelines to accomplish:"

    current_guidelines = state['guidelines'][state['current_stage']]

    for idx, guideline in enumerate(current_guidelines):
        system += f"\n\n{idx+1}. {guideline}"

    system += f"\n\nIf the therapist has completely lost direction and is not behaving as a therapist should, \
    please give a tip to the therapist on how to get back on track. \
    Be gentle but direct, and keep it a one short sentence."

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

def add_to_sidebar(text):
    sidebar = st.sidebar
    info_container = sidebar.container()

    with info_container:
        if text.startswith("Tip:"):
            st.warning(text)
        else:
            st.success(text)

def render_screen():
    if "messages" not in st.session_state:
        reset_session()

    st.sidebar.title("Tips & Progress")

    st.title("Talk with Noa")
    st.write("You are the therapist, help Noa with her conflict.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": st.session_state.system_prompt}] + st.session_state.messages,
            stream=True,
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

        guidelines_promise = start_promise(evaluate_guidelines, st.session_state.to_dict())
        tip_promise = start_promise(get_director_tip, st.session_state.to_dict())
        completed_guidelines, done, new_state = guidelines_promise.result()
        tip = tip_promise.result()

        for key, value in new_state.items():
            st.session_state[key] = value

        if len(completed_guidelines) > 0:
            st.session_state.rounds_since_last_completion = 0
        else:
            st.session_state.rounds_since_last_completion += 1

        for guideline in completed_guidelines:
            add_to_sidebar(f"Guideline '{guideline}' completed successfully.")

        if st.session_state.rounds_since_last_completion > 0:
            add_to_sidebar(f"Tip: {tip}")

        if done:
            add_to_sidebar("All guidelines completed successfully!")

if __name__ == "__main__":
    render_screen()