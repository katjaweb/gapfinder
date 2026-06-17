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
from gapfinder_agent.ingest import YouTubePipeline, VideoMetadataService, TranscriptService, StorageService, ChunkService
from gapfinder_agent.tools import GapFinderAgentTools
from tests.utils import SearchResult, ToolCall, collect_tools


# ytt_api = YouTubeTranscriptApi()
storage_service = StorageService()

pipeline = YouTubePipeline(
    # metadata_service=VideoMetadataService(),
    # transcript_service=TranscriptService(ytt_api),
    storage_service=storage_service,
    chunk_service=ChunkService(storage_service),
    chunk_documents_fn=chunk_documents,
)


index = pipeline.create_rag_index()
print("RAG index created with", len(index.docs), "chunks")


def create_test_agent(output_type=None):
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
async def test_agent_runs():
    agent = create_test_agent(output_type=SearchResult)

    user_prompt = "Ask me a question about embeddings for this video: https://www.youtube.com/watch?v=wjZofJX0v4M?"
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

    user_prompt = 'I want to leran more about this video: https://www.youtube.com/watch?v=wjZofJX0v4M?'
    result = await run_agent(agent, user_prompt)

    messages = result.new_messages()

    tool_calls = collect_tools(messages)
    print('Tool Calls: {tool_calls}')
    assert len(tool_calls) <= 7, f"Expected at most 6 tool calls, but got {len(tool_calls)}"

    first_call = tool_calls[0]
    assert first_call.name == 'get_video_id'
    print('First tool call:', first_call.name, first_call.args)

    second_call = tool_calls[1]
    assert second_call.name == 'get_summary'
    print('Second tool call:', second_call.name, second_call.args[0:200])
