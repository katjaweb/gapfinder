from pydantic_ai import Agent, RunUsage

from gapfinder_agent.yt_rag_pipeline import VideoMetadataService
from gapfinder_agent.yt_agent import create_agent, run_agent, GapFinderAgentConfig
from gapfinder_agent.tools import GapFinderAgentTools
from gapfinder_agent.yt_rag_pipeline import YouTubePipeline, VideoMetadataService, TranscriptService, StorageService, ChunkService

from minsearch import Index
from gitsource import chunk_documents
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import dotenv
dotenv.load_dotenv()

client = OpenAI()
ytt_api = YouTubeTranscriptApi()

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

url = "https://www.youtube.com/watch?v=wjZofJX0v4M"
result = pipeline.process_video(url)
print(result)
index = pipeline.create_rag_index()
print("RAG index created with", len(index.docs), "chunks")

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

    print("GapFinder Agent is ready!")
    print("Type 'stop' to exit.\n")

    message_history = []

    while True:

        user_prompt = input("\nYou: ")

        if user_prompt.lower().strip() == "stop":
            print("Goodbye!")
            break

        print("\nAgent is thinking...\n")

        result = await run_agent(
            agent,
            user_prompt,
            message_history
        )

        print(f"**ASSISTANT:**\n{result.output}")

        message_history = result.all_messages()

if __name__ == "__main__":
    import asyncio
    asyncio.run(chat())

# uv run python -m gapfinder_agent.main