import pytest
from minsearch import Index
from gitsource import chunk_documents
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import dotenv
dotenv.load_dotenv()

client = OpenAI()
ytt_api = YouTubeTranscriptApi()

from gapfinder_agent.yt_agent import create_agent, run_agent, GapFinderAgentConfig
from gapfinder_agent.ingest import YouTubePipeline, VideoMetadataService, TranscriptService, StorageService, ChunkService
from gapfinder_agent.tools import GapFinderAgentTools

from tests.judge import assert_criteria

storage_service = StorageService()

pipeline = YouTubePipeline(
    storage_service=storage_service,
    chunk_service=ChunkService(storage_service),
    chunk_documents_fn=chunk_documents,
)

index = pipeline.create_rag_index()

@pytest.fixture(scope="module")
def agent(output_type=None):
    tools = GapFinderAgentTools(
        client=client,
        model='gpt-4o',
        index_cls=index
    )
    
    agent_config = GapFinderAgentConfig(
        client=client,
        model='gpt-4o'
        )
   
    agent = create_agent(agent_config, tools, output_type=output_type)

    return agent


@pytest.mark.asyncio
async def test_judge_agent_uses_tools(agent):
    user_prompt = 'What is this video about: https://www.youtube.com/watch?v=wjZofJX0v4M?'
    result = await run_agent(agent, user_prompt)

    await assert_criteria(result, [
        "makes at least 2 tool calls",
        "provides key concepts of the video in the output",
        "asks the user if they want to explore the concepts",
    ])