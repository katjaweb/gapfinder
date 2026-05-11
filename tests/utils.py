from dataclasses import dataclass
from typing import Dict, Any

from pydantic import BaseModel
from pydantic_ai import Agent

class SearchResult(BaseModel):
    answer: str
    confidence: float
    found_answer: bool
    followup_questions: list[str]

@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]


def collect_tools(messages):
    tool_calls = []

    for m in messages:
        for p in m.parts:
            part_kind = p.part_kind

            if part_kind != 'tool-call':
                continue

            if p.tool_name == 'final_result':
                continue

            tool_calls.append(ToolCall(p.tool_name, p.args))

    return tool_calls


def get_model_name(agent):
    provider = agent.model.system
    model_name = agent.model.model_name
    return f'{provider}:{model_name}'

async def run_agent_test(agent, user_prompt, message_history=None, output_type=None):
    result = await agent.run(user_prompt, message_history=message_history, output_type=output_type)

    model = get_model_name(agent)

    return result

class MockYouTubeTranscriptAPI:
    """Mock class for YouTubeTranscriptApi."""

    def fetch(self, video_id):
        return self.MockTranscript(self.transcript)


    def __init__(self, transcript):
        self.transcript = transcript

    class MockTranscript:
        def __init__(self, snippets):
            self.snippets = snippets


class MockTranscriptSnippet:
    """Mock class for YouTubeTranscriptApi.TranscriptSnippet."""
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text    