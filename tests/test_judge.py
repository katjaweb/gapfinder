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
from gapfinder_agent.tools import GapFinderAgentTools

from tests.utils import run_agent_test
from tests.judge import assert_criteria


@pytest.fixture(scope="module")
def agent(output_type=None):
    tools = GapFinderAgentTools(
        client=client,
        model="gpt-4o-mini",
        ytt_api=ytt_api,
        chunk_func=chunk_documents,
        index_cls=Index
    )
   
    agent_config = GapFinderAgentConfig()
    agent = create_agent(agent_config, tools, output_type=output_type)

    return agent


@pytest.mark.asyncio
async def test_judge_agent_uses_tools(agent):
    user_prompt = 'What is this video about: https://www.youtube.com/watch?v=wjZofJX0v4M?'
    result = await run_agent_test(agent, user_prompt)

    await assert_criteria(result, [
        "makes at least 2 tool calls",
        "provides key concepts of the video in the output",
        "asks the user if they want to explore the concepts",
        "suggests questions for the concepts"
    ])