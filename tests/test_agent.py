import pytest
import json

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
from tests.utils import SearchResult, ToolCall, collect_tools


def create_test_agent(output_type=None):
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
async def test_agent_runs():
    agent = create_test_agent(output_type=SearchResult)

    user_prompt = "What is this video about: https://www.youtube.com/watch?v=wjZofJX0v4M?"
    result = await run_agent(agent, user_prompt)
    search_result = result.output

    
    assert search_result.answer is not None
    assert search_result.confidence >= 0.0
    assert search_result.found_answer is True
    assert len(search_result.followup_questions) > 0

    print("\n=== TEST AGENT RUNS RESULT ===")
    print("Result:\n", search_result.answer)
    print("Confidence:\n", search_result.confidence)
    print("Found Answer:\n", search_result.found_answer)
    print("Followup Questions:\n", search_result.followup_questions)
    print("==================\n")


@pytest.mark.asyncio
async def test_agent_uses_tools():
    agent = create_test_agent(output_type=ToolCall)

    user_prompt = 'What is this video about: https://www.youtube.com/watch?v=wjZofJX0v4M?'
    result = await run_agent(agent, user_prompt)

    messages = result.new_messages()

    tool_calls = collect_tools(messages)
    # print('Tool Calls: {tool_calls}')
    assert len(tool_calls) >= 2

    first_call = tool_calls[0]
    assert first_call.name == 'process_video_transcript'
    print('First tool call:', first_call.name, first_call.args)

    second_call = tool_calls[1]
    assert second_call.name == 'extract_concepts'
    print('Second tool call:', second_call.name, second_call.args[0:200])
