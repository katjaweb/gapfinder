import asyncio
import streamlit as st
import dotenv

from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from minsearch import Index
from gitsource import chunk_documents

from gapfinder_agent.yt_agent import (
    create_agent,
    run_agent,
    GapFinderAgentConfig,
)

from gapfinder_agent.tools import GapFinderAgentTools


# ---------------------------------------------------
# ENV
# ---------------------------------------------------

dotenv.load_dotenv()


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="GapFinder Agent",
    page_icon="🎥",
    layout="wide",
)

st.title("🎥 GapFinder YouTube Agent")


# ---------------------------------------------------
# AGENT SETUP
# ---------------------------------------------------

@st.cache_resource
def setup_agent():

    client = OpenAI()

    ytt_api = YouTubeTranscriptApi()

    config = GapFinderAgentConfig()

    agent_tools = GapFinderAgentTools(
        client=client,
        model="gpt-4o-mini",
        ytt_api=ytt_api,
        chunk_func=chunk_documents,
        index_cls=Index,
    )

    agent = create_agent(
        config=config,
        agent_tools=agent_tools,
    )

    return agent


agent = setup_agent()


# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# ---------------------------------------------------
# VIDEO INPUT
# ---------------------------------------------------

video_url = st.text_input(
    "YouTube URL",
    value="https://www.youtube.com/watch?v=wjZofJX0v4M"
)


# ---------------------------------------------------
# DISPLAY CHAT
# ---------------------------------------------------

for msg in st.session_state.chat_messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------------------------------------------------
# AGENT CALL
# ---------------------------------------------------

async def ask_agent(user_prompt: str):

    result = await run_agent(
        agent,
        user_prompt,
        st.session_state.message_history,
    )

    st.session_state.message_history = result.all_messages()

    return result.output


# ---------------------------------------------------
# INITIAL ANALYSIS BUTTON
# ---------------------------------------------------

if st.button("Analyze Video"):

    initial_prompt = (
        f"I want to learn more about this video: {video_url}"
    )

    st.session_state.chat_messages.append({
        "role": "user",
        "content": initial_prompt
    })

    with st.chat_message("user"):
        st.markdown(initial_prompt)

    with st.chat_message("assistant"):

        with st.spinner("Analyzing video..."):

            response = asyncio.run(
                ask_agent(initial_prompt)
            )

            st.markdown(response)

    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": response
    })


# ---------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------

user_input = st.chat_input(
    "Ask questions about the video..."
)

if user_input:

    st.session_state.chat_messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = asyncio.run(
                ask_agent(user_input)
            )

            st.markdown(response)

    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": response
    })
