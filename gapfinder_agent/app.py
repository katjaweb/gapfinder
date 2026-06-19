import asyncio
import concurrent.futures
import dotenv
import logfire
import streamlit as st

from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from gitsource import chunk_documents

from gapfinder_agent.yt_agent import (
    create_agent,
    run_agent,
    GapFinderAgentConfig,
)

from gapfinder_agent.tools import GapFinderAgentTools
from gapfinder_agent.ingest import YouTubePipeline, VideoMetadataService, TranscriptService, StorageService, ChunkService


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

st.title("🎥 GapFinder learning assistant for video tutorials")

# ---------------------------------------------------
# VIDEO PROCESSING
# ---------------------------------------------------

metadata = VideoMetadataService()
transcripts = TranscriptService(YouTubeTranscriptApi())
storage = StorageService()
chunking = ChunkService(storage)

pipeline = YouTubePipeline(
    metadata,
    transcripts,
    storage,
    chunking,
    chunk_documents_fn=chunk_documents
)

if "video_url" not in st.session_state:
    st.session_state.video_url = "https://www.youtube.com/watch?v=wjZofJX0v4M"

if "rag_index" not in st.session_state:
    st.session_state.rag_index = None

if "agent" not in st.session_state:
    st.session_state.agent = None


# ---------------------------------------------------
# AGENT SETUP
# ---------------------------------------------------

def setup_agent(index):

    with logfire.span("setup_agent"):

        client = OpenAI()

        config = GapFinderAgentConfig(
            client=client,
            model='gpt-4o-mini'
        )

        agent_tools = GapFinderAgentTools(
            client=OpenAI(),
            model='gpt-4o-mini',
            index_cls=index
        )

        agent = create_agent(
            config=config,
            agent_tools=agent_tools,
        )

        logfire.info("agent_initialized")

        return agent


agent = None


# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


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
                st.session_state.agent,
                user_prompt,
                st.session_state.message_history,
            )

            st.session_state.message_history = result.all_messages()

            logfire.info(
                "assistant_response",
                response=result.output,
            )

            return result.output


def run_coroutine_sync(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_thread)
            return future.result()

# ---------------------------------------------------
# VIDEO INPUT
# ---------------------------------------------------

with st.sidebar:
    st.header("Video URL")
    st.markdown(
        """
        **How to use**

        1. Use the default URL or enter your own YouTube URL and click **Analyze Video**.
        2. Wait while the video is processed.
        3. Start chatting with the assistant.
        4. When you think you are finished, ask for evaluation and get your gap report.
        """
    )
    video_url = st.text_input(
        "YouTube URL",
        value=st.session_state.video_url,
    )
    analyze_button = st.button("Analyze Video")

    if analyze_button:
        st.session_state.video_url = video_url

        with st.spinner("Processing video..."):
            pipeline.process_video(video_url)
            index = pipeline.create_rag_index()
            st.session_state.rag_index = index
            st.session_state.agent = setup_agent(index)

        st.success("Video processed successfully.")

        initial_prompt = (
            f"I want to learn more about this video: {video_url}"
        )

        st.session_state.chat_messages.append({
            "role": "user",
            "content": initial_prompt,
        })

        with st.spinner("Analyzing video..."):
            with logfire.attach_context(
                st.session_state.logfire_context
            ):
                with logfire.span(
                    "initial_video_analysis",
                    video_url=video_url,
                ):
                    response = run_coroutine_sync(
                        ask_agent(initial_prompt)
                    )

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

    with st.spinner("Thinking..."):

        with logfire.attach_context(
            st.session_state.logfire_context
        ):

            with logfire.span(
                "chat_interaction",
            ):

                response = run_coroutine_sync(
                    ask_agent(user_input)
                )

    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": response,
    })

for msg in st.session_state.chat_messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


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