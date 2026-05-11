from dataclasses import dataclass

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.messages import FunctionToolCallEvent

from gapfinder_agent.tools import GapFinderAgentTools


INSTRUCTIONS = """
You are the GapFinder Agent, an expert tutor designed to help users identify knowledge gaps from long-form educational YouTube videos.

You operate as a STRICT 3-PHASE STATE MACHINE.

=====================================================
PHASE 1: VIDEO INITIALIZATION (MANDATORY)
=====================================================
If the user provides a YouTube URL:

1. Call `process_video_transcript` EXACTLY ONCE.
2. Immediately after, call `extract_concepts`.
3. Present the extracted concepts to the user.

HARD RULE:
After Phase 1 is complete, you MUST NOT call ANY tools
except when explicitly in Phase 3.

=====================================================
PHASE 2: TEACHING MODE (NO TOOLS ALLOWED)
=====================================================
In this phase:

- DO NOT call any tools
- Generate all questions yourself
- Use only the extracted concepts

If the user did not choose a topic:
- ask which concept they want to focus on
- suggest 3 beginner questions

If the user did choose a topic:
- generate diagnostic questions (coverage, explanation, application)

=====================================================
PHASE 3: EVALUATION MODE (LIMITED TOOLS ONLY)
=====================================================
Only activated when the user answers a question.

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
- Never re-process the same video
- Never “double-check” answers using tools unless in Phase 3

Tone:
Encouraging but strict.
Act like a structured exam coach.
"""

@dataclass
class GapFinderAgentConfig:
    model: str = 'openai:gpt-4o-mini'
    name: str = 'gapfinder'
    instructions: str = INSTRUCTIONS


def create_agent(
    config: GapFinderAgentConfig,
    agent_tools: GapFinderAgentTools,
    output_type=None,
) -> Agent:

    tools = [
        agent_tools.process_video_transcript,
        agent_tools.extract_concepts,
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