from pydantic_ai import Agent, RunUsage

from gapfinder_agent.yt_agent import create_agent, run_agent, GapFinderAgentConfig
from gapfinder_agent.tools import GapFinderAgentTools

from minsearch import Index
from gitsource import chunk_documents
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import dotenv
dotenv.load_dotenv()

client = OpenAI()
ytt_api = YouTubeTranscriptApi()
config = GapFinderAgentConfig()

agent_tools = GapFinderAgentTools(
    client=client,
    model="gpt-4o-mini",
    ytt_api=ytt_api,
    chunk_func=chunk_documents,
    index_cls=Index
)

async def run_qna(agent: Agent):
    messages = []
    usage = RunUsage()

    while True:
        user_prompt = input('You:')
        if user_prompt.lower().strip() == 'stop':
            break

        user_prompt = "What is this video about: https://www.youtube.com/watch?v=wjZofJX0v4M?"
        result = await run_agent(agent, user_prompt, messages)

        usage = usage + result.usage()
        messages.extend(result.new_messages())

async def chat(config=config, agent_tools=agent_tools):

    agent = create_agent(config=config, agent_tools=agent_tools)

    print("GapFinder Agent is ready!")
    print("Type 'stop' to exit.\n")

    message_history = []


    initial_prompt = (
        "I want to learn more about this video: "
        "https://www.youtube.com/watch?v=wjZofJX0v4M"
    )

    print(f"You: {initial_prompt}")

    result = await run_agent(
        agent,
        initial_prompt,
        message_history
    )

    print(f"**ASSISTANT:**\n{result.output}")

    message_history = result.all_messages()

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