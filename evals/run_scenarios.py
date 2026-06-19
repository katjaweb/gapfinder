import argparse
import asyncio
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from gitsource import chunk_documents
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gapfinder_agent.ingest import ChunkService, StorageService, YouTubePipeline
from gapfinder_agent.tools import GapFinderAgentTools
from gapfinder_agent.yt_agent import GapFinderAgentConfig, create_agent, run_agent


try:
    import dotenv
except ModuleNotFoundError:
    dotenv = None

if dotenv is not None:
    dotenv.load_dotenv(ROOT_DIR / ".env")

SCENARIOS_FILE = Path(__file__).resolve().parent / "scenarios.csv"
RESULTS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = os.getenv("GAPFINDER_MODEL", "gpt-4o-mini")

MODEL_PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


@dataclass
class ToolCall:
    name: str
    args: Any


def build_agent(model: str):
    client = OpenAI()
    storage_service = StorageService()
    pipeline = YouTubePipeline(
        storage_service=storage_service,
        chunk_service=ChunkService(storage_service),
        chunk_documents_fn=chunk_documents,
        openai_client=client,
    )
    index = pipeline.create_rag_index()

    tools = GapFinderAgentTools(
        client=client,
        model=model,
        index_cls=index,
    )
    agent_config = GapFinderAgentConfig(
        client=client,
        model=model,
    )
    return create_agent(agent_config, tools)


def calculate_cost(usage, model: str) -> float | None:
    prices = MODEL_PRICES.get(model)
    if prices is None:
        return None

    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return round(input_cost + output_cost, 6)


def collect_tool_context(messages: list[Any]) -> list[Any]:
    tool_context = []

    for msg in messages:
        for part in getattr(msg, "parts", []):
            if part.__class__.__name__ == "ToolReturnPart":
                tool_context.append(getattr(part, "content", None))

    return tool_context


def collect_tool_calls(messages: list[Any]) -> list[ToolCall]:
    tool_calls = []

    for msg in messages:
        for part in getattr(msg, "parts", []):
            if getattr(part, "part_kind", None) != "tool-call":
                continue
            if getattr(part, "tool_name", None) == "final_result":
                continue

            tool_calls.append(
                ToolCall(
                    name=getattr(part, "tool_name", ""),
                    args=getattr(part, "args", None),
                )
            )

    return tool_calls


def serialize_output(output: Any) -> Any:
    if hasattr(output, "model_dump"):
        return output.model_dump()
    return output


def usage_to_dict(usage) -> dict[str, int]:
    return {
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }


def load_scenarios(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


async def run_all(
    scenarios_path: Path = SCENARIOS_FILE,
    output_dir: Path = RESULTS_DIR,
    model: str = DEFAULT_MODEL,
    limit: int | None = None,
) -> Path:
    scenarios = load_scenarios(scenarios_path)
    if limit is not None:
        scenarios = scenarios[:limit]

    agent = build_agent(model)
    results = []

    for i, scenario in enumerate(scenarios):
        question = scenario["question"]
        print(f"[{i + 1}/{len(scenarios)}] {scenario['category']} / {scenario['type']}")

        start = time.time()
        result = await run_agent(agent, question)
        elapsed = time.time() - start

        messages = result.all_messages()
        usage = result.usage()
        cost = calculate_cost(usage, model)

        row = {
            "question": question,
            "user_answer": scenario.get("user_answer", ""),
            "category": scenario.get("category", ""),
            "type": scenario.get("type", ""),
            "url": scenario.get("url", ""),
            "expected_quality": scenario.get("expected_quality", ""),
            "judge_criteria": scenario.get("judge_criteria", ""),
            "tool_calls": [asdict(tool_call) for tool_call in collect_tool_calls(messages)],
            "tool_context": collect_tool_context(messages),
            "output": serialize_output(result.output),
            "execution_time": round(elapsed, 2),
            "tokens": usage_to_dict(usage),
            "cost": cost,
            "model": model,
        }
        results.append(row)

        cost_label = f"${cost}" if cost is not None else "cost unavailable"
        print(f"  Done in {elapsed:.1f}s ({cost_label})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    known_costs = [row["cost"] for row in results if row["cost"] is not None]
    if known_costs:
        print(f"\nTotal cost: ${sum(known_costs):.4f}")
    else:
        print("\nTotal cost: unavailable for this model")

    print(f"Saved {len(results)} results to {output_file}")
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GapFinder YouTube-agent evaluation scenarios."
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=SCENARIOS_FILE,
        help="Path to the scenarios CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for results_*.json.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model used by the GapFinder agent and evaluation tool.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N scenarios.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_all(
            scenarios_path=args.scenarios,
            output_dir=args.output_dir,
            model=args.model,
            limit=args.limit,
        )
    )
