import asyncio
import dotenv
import logfire
import streamlit as st

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
# LOGFIRE
# ---------------------------------------------------

logfire.configure(
    service_name="gapfinder-agent",
    send_to_logfire=True,
)

logfire.instrument_pydantic_ai()

# Session-Span
if "logfire_context" not in st.session_state:

    session_span = logfire.span(
        "streamlit_session"
    )

    session_span.__enter__()

    st.session_state.logfire_span = session_span
    st.session_state.logfire_context = logfire.get_context()

    logfire.info("streamlit_session_started")


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

    with logfire.span("setup_agent"):

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

        logfire.info("agent_initialized")

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
    value="https://www.youtube.com/watch?v=wjZofJX0v4M",
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

    with logfire.attach_context(
        st.session_state.logfire_context
    ):

        with logfire.span(
            "agent_request",
            prompt=user_prompt,
        ):

            logfire.info(
                "user_prompt",
                prompt=user_prompt,
            )

            result = await run_agent(
                agent,
                user_prompt,
                st.session_state.message_history,
            )

            st.session_state.message_history = result.all_messages()

            logfire.info(
                "assistant_response",
                response=result.output,
            )

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
        "content": initial_prompt,
    })

    with st.chat_message("user"):
        st.markdown(initial_prompt)

    with st.chat_message("assistant"):

        with st.spinner("Analyzing video..."):

            with logfire.attach_context(
                st.session_state.logfire_context
            ):

                with logfire.span(
                    "initial_video_analysis",
                    video_url=video_url,
                ):

                    response = asyncio.run(
                        ask_agent(initial_prompt)
                    )

                    st.markdown(response)

    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": response,
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
        "content": user_input,
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            with logfire.attach_context(
                st.session_state.logfire_context
            ):

                with logfire.span(
                    "chat_interaction",
                ):

                    response = asyncio.run(
                        ask_agent(user_input)
                    )

                    st.markdown(response)

    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": response,
    })


# ---------------------------------------------------
# FEEDBACK
# ---------------------------------------------------

if st.session_state.chat_messages:

    st.divider()

    feedback = st.feedback(
        "thumbs",
        key="conversation_feedback",
    )

    if feedback is not None:

        score = "thumbs_up" if feedback == 1 else "thumbs_down"

        with logfire.attach_context(
            st.session_state.logfire_context
        ):

            with logfire.span(
                "conversation_feedback",
                feedback=score,
            ):

                logfire.info(
                    "user_feedback",
                    feedback=score,
                    total_messages=len(
                        st.session_state.chat_messages
                    ),
                    video_url=video_url,
                )

        st.success("Feedback received — thank you!")