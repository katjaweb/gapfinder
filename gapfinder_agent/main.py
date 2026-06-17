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


url = "https://www.youtube.com/watch?v=wjZofJX0v4M"
result = pipeline.process_video(url)
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


async def run_qna(agent: Agent):
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

async def chat(config=config, agent_tools=agent_tools):

    agent = create_agent(config=config, agent_tools=agent_tools)

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

if __name__ == "__main__":
    import asyncio
    asyncio.run(chat())

# uv run python -m gapfinder_agent.main