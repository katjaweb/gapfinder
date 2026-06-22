import argparse
from pydantic_ai import Agent, RunUsage

from gapfinder_agent.ingest import VideoMetadataService
from gapfinder_agent.yt_agent import create_agent, run_agent, GapFinderAgentConfig
from gapfinder_agent.tools import GapFinderAgentTools
from gapfinder_agent.ingest import YouTubePipeline, VideoMetadataService, TranscriptService, StorageService, ChunkService

from minsearch import Index
from gitsource import chunk_documents
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

client = OpenAI()
ytt_api = YouTubeTranscriptApi()
storage_service = StorageService()

pipeline = YouTubePipeline(
    metadata_service=VideoMetadataService(),
    transcript_service=TranscriptService(ytt_api),
    storage_service=storage_service,
    chunk_service=ChunkService(storage_service),
    chunk_documents_fn=chunk_documents,
)


def build_agent(url: str):
    """Build and return a GapFinder agent after ingesting the provided YouTube video."""
    result = pipeline.process_video(url, generate_summary=True)
    logger.info(f"Processed video: {result[0]['title']}")
    index = pipeline.create_rag_index()
    logger.info(f"RAG index created with {len(index.docs)} chunks")

    config = GapFinderAgentConfig(
        client=client,
        model='gpt-4o-mini'
    )

    agent_tools = GapFinderAgentTools(
        client=OpenAI(),
        model='gpt-4o-mini',
        index_cls=index
    )

    return create_agent(config=config, agent_tools=agent_tools)


async def run_qna(agent: Agent):
    """Run a simple Q&A loop with the given agent until the user types stop."""
    messages = []
    usage = RunUsage()

    while True:
        user_prompt = input('You:')
        if user_prompt.lower().strip() == 'stop':
            break

        # user_prompt = "What is this video about: https://www.youtube.com/watch?v=wjZofJX0v4M?"
        result = await run_agent(agent, user_prompt, messages)

        usage = usage + result.usage()
        messages.extend(result.new_messages())

async def chat(agent: Agent):
    """Start an interactive agent chat loop in the terminal."""

    logger.info("GapFinder Agent is ready!")
    logger.info("Type 'stop' to exit.\n")

    message_history = []

    while True:

        user_prompt = input("\nYou: ")

        if user_prompt.lower().strip() == "stop":
            logger.info("Goodbye!")
            break

        logger.info("\nAgent is thinking...\n")

        result = await run_agent(
            agent,
            user_prompt,
            message_history
        )

        logger.info(f"**ASSISTANT:**\n{result.output}")

        message_history = result.all_messages()


def parse_args():
    """Parse command-line arguments and return the chosen YouTube URL."""
    default_url = "https://www.youtube.com/watch?v=wjZofJX0v4M"
    parser = argparse.ArgumentParser(description="Run GapFinderAgent with a YouTube URL")
    parser.add_argument(
        "url",
        nargs="?",
        help="YouTube video URL to ingest before starting the agent",
    )
    parser.add_argument(
        "--url",
        "-u",
        dest="url",
        help="YouTube video URL to ingest before starting the agent",
    )
    args = parser.parse_args()
    if args.url is None:
        args.url = default_url
    return args


if __name__ == "__main__":
    import asyncio

    args = parse_args()
    agent = build_agent(args.url)
    asyncio.run(chat(agent))
