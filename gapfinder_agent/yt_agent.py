from dataclasses import dataclass
from contextvars import ContextVar
from typing import Any
import logging
from pydantic_ai import Agent, AgentRunResult, Tool
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from gapfinder_agent.tools import GapFinderAgentTools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TOOL_CALL_LIMIT = 4
TOOL_LIMIT_FALLBACK = (
    "I reached the tool-call limit for this turn, so I stopped using tools. "
    "Please try again with a narrower question, or ask me to continue from the "
    "information we already have."
)

_tool_call_limit: ContextVar[int | None] = ContextVar(
    "gapfinder_tool_call_limit",
    default=None,
)
_tool_call_count: ContextVar[int] = ContextVar(
    "gapfinder_tool_call_count",
    default=0,
)

INSTRUCTIONS = """
You are the GapFinder Agent, an expert tutor for educational YouTube videos.

TOOL USAGE RULES
- Use tools only when the user’s request cannot be answered from the current conversation, summary, or previous tool results.
- Do not call tools in a loop.
- Do not call tools if the answer is already available or can be given directly.
- Maximum 4 tool calls per assistant turn.
- If you need a tool, count the tool calls in this response before calling it.
- If the remaining tool budget is 0, stop tool calling and provide an answer based on the information you have.

PHASE 1: VIDEO SETUP
- Maximum 4 tool calls per assistant turn for this phase.
- If the user does not provide a URL, ask for it.
- If the user provides a YouTube URL, call get_video_id once and then call get_summary once.
- Provide the summary to the user and ask which topic they want to explore.
- If they ask a question about the summary, answer it using the summary and conversation and use at maximum two search_video_transcript tool calls.
- If the user selects a topic, the next phase will start. 

Allowed tools in Phase 1:
- get_video_id (max 1)
- get_summary (max 1)
- search_video_transcript (max 2)

PHASE 2: QUESTIONING
- Maximum 4 tool calls per assistant turn for this phase.
- Do not give an overview of the choosen topic; start asking questions right away.
- Ask one guided question at a time based on the selected concept.
- Start with easier questions and progressively increase difficulty based on the user's answers.
- Do not answer the user’s question directly; guide them with a follow-up question.
- Use search_video_transcript at most once per response to support your feedback.
- Do not call any additional tools after giving feedback.
- Do not ask questions for other topics until the user explicitly asks to switch topics.

Allowed tool in Phase 2:
- search_video_transcript (max 1 per response)

PHASE 3: EVALUATION
- Only enter evaluation mode when the user explicitly asks for it.
- Use search_video_transcript only to gather evidence.
- Use evaluate_user_answer once to grade the answer.
- Do not use any additional tools in this phase.

Allowed tools in Phase 3:
- search_video_transcript (max 1)
- evaluate_user_answer (max 1)

GLOBAL RULES
- Never exceed 4 tool calls in a single assistant turn.
- Never call tools when not necessary.
- Never use tools to “double-check” an answer.
- If the answer is available from the summary, conversation, or previous results, do not call any tools.
- Be concise, structured, and helpful.
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
    """Create and return a GapFinder Agent configured with tools and instructions."""

    tools = [
        Tool(agent_tools.get_video_id, prepare=prepare_tool_with_call_limit),
        Tool(agent_tools.get_summary, prepare=prepare_tool_with_call_limit),
        Tool(agent_tools.search_video_transcript, prepare=prepare_tool_with_call_limit),
        Tool(agent_tools.evaluate_user_answer, prepare=prepare_tool_with_call_limit),
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


def prepare_tool_with_call_limit(ctx, tool_def):
    """Disable a tool if the current tool-call limit has already been reached."""
    limit = _tool_call_limit.get()

    if limit is not None and _tool_call_count.get() >= limit:
        logger.info("Tool call limit reached; hiding tool: %s", tool_def.name)
        return None

    return tool_def

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
            if tool_name != "final_result":
                _tool_call_count.set(_tool_call_count.get() + 1)
            logger.info(f"TOOL CALL ({self.agent_name}): {tool_name}({args})")

    async def __call__(self, ctx, event):
        return await self.print_function_calls(ctx, event)


async def run_agent(
        agent: Agent,
        user_prompt: str,
        message_history=None,
        tool_calls_limit: int | None = DEFAULT_TOOL_CALL_LIMIT,
    ) -> AgentRunResult:
    """Execute the agent on a user prompt while enforcing tool-call limits."""
    callback = NamedCallback(agent) 

    if message_history is None:
        message_history = []

    limit_token = _tool_call_limit.set(tool_calls_limit)
    count_token = _tool_call_count.set(0)

    try:
        result = await agent.run(
            user_prompt,
            usage_limits=UsageLimits(tool_calls_limit=tool_calls_limit),
            event_stream_handler=callback,
            message_history=message_history
        )
    except UsageLimitExceeded as e:
        logger.warning("Tool call limit exceeded; retrying without tools: %s", e)
        result = await run_without_tools_after_limit(
            agent=agent,
            user_prompt=user_prompt,
            message_history=message_history,
            callback=callback,
            error=e,
        )
    finally:
        _tool_call_limit.reset(limit_token)
        _tool_call_count.reset(count_token)

    return result


async def run_without_tools_after_limit(
        agent: Agent,
        user_prompt: str,
        message_history: list[Any],
        callback: NamedCallback,
        error: UsageLimitExceeded,
    ) -> AgentRunResult:
    """Retry the agent request without tools after a tool-call limit exception."""
    recovery_prompt = f"""
{user_prompt}

The tool-call limit for this assistant turn has been reached. Do not call any
tools. Answer using only the conversation history and any tool results already
available. If the available information is not enough, say what is missing and
ask the user for the next useful detail.
"""

    try:
        with agent.override(tools=[]):
            return await agent.run(
                recovery_prompt,
                usage_limits=UsageLimits(tool_calls_limit=None, request_limit=2),
                event_stream_handler=callback,
                message_history=message_history,
            )
    except Exception:
        logger.exception("Tool-limit recovery run failed.")
        return fallback_result_after_limit(
            user_prompt=user_prompt,
            message_history=message_history,
            error=error,
        )


def fallback_result_after_limit(
        user_prompt: str,
        message_history: list[Any],
        error: UsageLimitExceeded,
    ) -> AgentRunResult[str]:
    """Return a fallback AgentRunResult when tool-limit recovery fails."""
    output = f"{TOOL_LIMIT_FALLBACK}\n\nDetails: {error}"
    new_messages = [
        ModelRequest(parts=[UserPromptPart(content=user_prompt)]),
        ModelResponse(parts=[TextPart(content=output)]),
    ]

    result = AgentRunResult(output=output)
    result._state.message_history = [
        *message_history,
        *new_messages,
    ]
    result._new_message_index = len(message_history)
    return result
