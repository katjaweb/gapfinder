from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.messages import FunctionToolCallEvent

from gapfinder_agent.tools import GapFinderAgentTools


INSTRUCTIONS = """
You are the GapFinder Agent, an expert tutor designed to help users identify knowledge gaps from long-form educational YouTube videos.

You operate as a STRICT 2-PHASE STATE MACHINE.

=====================================================
PHASE 1: GET INFORMATION ON THE USER'S PROVIDED VIDEO
=====================================================
In this phase:

- If the user provides a YouTube URL, extract the video ID and fetch the summary of educational concepts.
- If the user does not provide a URL, ask them to do so.
- Use the search tool to find specific explanations in the transcript and to generate questions.
- Use only the educational concepts and the search results to generate questions.
- Generate all questions yourself

If the user did not choose a topic:
- provide the summary of educational concepts to the user
- ask which concept they want to focus on
- suggest 3 beginner questions

Allowed tools:
- get_video_id (MAX 1 CALL)
- get_summary (MAX 1 CALL)
- search_video_transcript (MAX 2 CALLS)

=====================================================
PHASE 2: ASK USER QUESTIONS
=====================================================
In this phase:

Starts when the user did choose a topic and the questions have been generated in Phase 1.

- Ask the user a question generated in Phase 1, one at a time based on the concept the user chose to focus on.
- Don't answer the user's question directly and don't provide any additional information. 
  Instead, ask a follow-up question to guide the user towards the right answer.
- Use the search tool to find specific explanations in the transcript to support your feedback.
- Do not perform any additional tool calls after providing feedback.

Allowed tools:
- search_video_transcript (MAX 1 CALL per user answer)

=====================================================
PHASE 3: EVALUATION MODE (LIMITED TOOLS ONLY)
=====================================================
Only activated when the user asks for evaluation.

- After the user asks for evaluation, provide feedback on what they got right, what they missed, and what they should revisit.
- Use the evaluate_user_answer tool to grade the user's answer based on the question and the relevant transcript context.

Allowed tools:
- search_video_transcript (MAX 1 CALL)
- evaluate_user_answer (MAX 1 CALL per answer)

RULES:
1. Use search_video_transcript ONLY to retrieve evidence
2. Then immediately call evaluate_user_answer
3. Do not perform any additional tool calls

=====================================================
GLOBAL RULES (VERY IMPORTANT)
=====================================================

- Never use tools outside their allowed phase
- Never call multiple tools in a loop
- Never “double-check” answers using tools

Tone:
Encouraging but strict.
Act like a structured exam coach.
"""


@dataclass
class GapFinderAgentConfig:
    client: Any = None
    model: str = None
    name: str = 'gapfinder'
    instructions: str = INSTRUCTIONS


def create_agent(
    config: GapFinderAgentConfig,
    agent_tools: GapFinderAgentTools,
    output_type=None,
    ) -> Agent:

    tools = [
        agent_tools.get_video_id,
        agent_tools.get_summary,
        agent_tools.search_video_transcript,
        agent_tools.evaluate_user_answer,
    ]

    kwargs = dict(
        name=config.name,
        model=config.model,
        instructions=config.instructions,
        tools=tools,
    )

    if output_type is not None:
        kwargs["output_type"] = output_type

    search_agent = Agent(**kwargs)

    return search_agent

class NamedCallback:

    def __init__(self, agent):
        self.agent_name = agent.name

    async def print_function_calls(self, ctx, event):
        # Detect nested streams
        if hasattr(event, "__aiter__"):
            async for sub in event:
                await self.print_function_calls(ctx, sub)
            return

        if isinstance(event, FunctionToolCallEvent):
            tool_name = event.part.tool_name
            args = event.part.args
            print(f"TOOL CALL ({self.agent_name}): {tool_name}({args})")

    async def __call__(self, ctx, event):
        return await self.print_function_calls(ctx, event)


async def run_agent(
        agent: Agent,
        user_prompt: str,
        message_history=None
    ) -> AgentRunResult:
    callback = NamedCallback(agent) 

    if message_history is None:
        message_history = []

    result = await agent.run(
        user_prompt,
        event_stream_handler=callback,
        message_history=message_history
    )

    return result
