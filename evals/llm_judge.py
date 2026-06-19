import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent


try:
    import dotenv
except ModuleNotFoundError:
    dotenv = None


ROOT_DIR = Path(__file__).resolve().parent.parent
EVALUATION_DIR = Path(__file__).resolve().parent

MODEL_PRICES = {
    "openai:gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai:gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

if dotenv is not None:
    dotenv.load_dotenv(ROOT_DIR / ".env")

DEFAULT_JUDGE_MODEL = os.getenv("GAPFINDER_JUDGE_MODEL", "openai:gpt-4o-mini")


class JudgeEvaluation(BaseModel):
    reasoning: str = Field(
        description=(
            "Brief explanation of the judgment, citing the agent output, "
            "expected answer quality, criteria, and tool behavior where relevant."
        )
    )
    label: Literal["good", "bad"] = Field(
        description=(
            "'good' means the GapFinder agent handled this scenario well; "
            "'bad' means it failed a material requirement."
        )
    )
    matched_expected_quality: bool = Field(
        description=(
            "Whether the agent correctly treated the learner answer as correct, "
            "partially correct, or incorrect according to expected_quality."
        )
    )
    followed_tool_policy: bool = Field(
        description=(
            "Whether tool calls are appropriate for explicit evaluation mode: "
            "at most one transcript search, exactly one evaluate_user_answer call, "
            "and no unrelated setup or looping."
        )
    )
    feedback_quality: Literal["strong", "adequate", "weak"] = Field(
        description="Quality of the tutoring feedback in the agent response."
    )


JUDGE_INSTRUCTIONS = """
You are an expert evaluator for GapFinder, a tutoring agent for educational
YouTube videos.

You judge the AGENT RESPONSE, not the learner answer directly.

Each item includes:
- question: the user's prompt to GapFinder
- user_answer: the learner's answer
- expected_quality: whether the learner answer is expected to be good or bad
- judge_criteria: scenario-specific requirements
- tool_calls: tools the GapFinder agent used
- tool_context: transcript/evaluation evidence returned by tools
- output: the GapFinder agent's final response

Label the agent response "good" only if all of these are true:
1. It correctly evaluates the learner answer according to expected_quality.
2. It satisfies the scenario-specific judge_criteria.
3. Its feedback is pedagogically useful: it names what is right, what is missing
   or wrong when applicable, and gives a focused next step.
4. It stays grounded in the video/tool context and does not invent unsupported
   facts about the video.
5. For explicit evaluation prompts, tool behavior is appropriate: it should use
   evaluate_user_answer exactly once; it may use search_video_transcript at most
   once for evidence; it should not call get_video_id/get_summary or loop tools.

Label the agent response "bad" if any material issue occurs, including:
- It praises an incorrect misconception as correct.
- It marks a good learner answer as wrong.
- It ignores the judge_criteria or gives generic feedback only.
- It gives a lecture instead of evaluating the answer and guiding the learner.
- It hallucinates video-specific details not supported by tool_context.
- It uses the wrong tools, too many tools, no evaluation tool for explicit
  evaluation, or unrelated setup tools.

Be strict but fair. Minor wording differences are fine when the feedback is
accurate and useful.
""".strip()


def create_judge_agent(model: str = DEFAULT_JUDGE_MODEL) -> Agent:
    return Agent(
        model=model,
        output_type=JudgeEvaluation,
        instructions=JUDGE_INSTRUCTIONS,
    )


def calculate_cost(usage, model: str) -> float | None:
    prices = MODEL_PRICES.get(model)
    if prices is None:
        return None

    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return round(input_cost + output_cost, 6)


def latest_results_file(directory: Path) -> Path:
    candidates = sorted(
        path
        for path in directory.glob("results_*.json")
        if not path.name.startswith("results_judged_")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No results_*.json files found in {directory}. "
            "Run evaluation/run_scenarios.py first or pass --results."
        )
    return candidates[-1]


def load_results(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_results(results: list[dict], output_file: Path) -> None:
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def build_judge_prompt(row: dict) -> str:
    return f"""
Evaluate this GapFinder agent run.

<QUESTION>
{row.get("question", "")}
</QUESTION>

<USER_ANSWER>
{row.get("user_answer", "")}
</USER_ANSWER>

<EXPECTED_LEARNER_ANSWER_QUALITY>
{row.get("expected_quality", "")}
</EXPECTED_LEARNER_ANSWER_QUALITY>

<SCENARIO_CATEGORY>
{row.get("category", "")}
</SCENARIO_CATEGORY>

<SCENARIO_TYPE>
{row.get("type", "")}
</SCENARIO_TYPE>

<JUDGE_CRITERIA>
{row.get("judge_criteria", "")}
</JUDGE_CRITERIA>

<TOOL_CALLS>
{json.dumps(row.get("tool_calls", []), ensure_ascii=False, indent=2)}
</TOOL_CALLS>

<TOOL_CONTEXT>
{json.dumps(row.get("tool_context", []), ensure_ascii=False, indent=2)}
</TOOL_CONTEXT>

<AGENT_RESPONSE>
{json.dumps(row.get("output", ""), ensure_ascii=False, indent=2)}
</AGENT_RESPONSE>
""".strip()


def judge_results(
    results_path: Path,
    output_dir: Path,
    model: str = DEFAULT_JUDGE_MODEL,
    limit: int | None = None,
) -> Path:
    results = load_results(results_path)
    rows_to_judge = results[:limit] if limit is not None else results
    judge_agent = create_judge_agent(model)
    total_judge_cost = 0.0
    known_cost_count = 0

    for i, row in enumerate(rows_to_judge):
        print(f"[{i + 1}/{len(rows_to_judge)}] {row.get('category', '')} / {row.get('type', '')}")

        evaluation = judge_agent.run_sync(build_judge_prompt(row))
        usage = evaluation.usage()
        cost = calculate_cost(usage, model)

        row["judge_label"] = evaluation.output.label
        row["judge_reasoning"] = evaluation.output.reasoning
        row["judge_matched_expected_quality"] = evaluation.output.matched_expected_quality
        row["judge_followed_tool_policy"] = evaluation.output.followed_tool_policy
        row["judge_feedback_quality"] = evaluation.output.feedback_quality
        row["judge_model"] = model
        row["judge_tokens"] = {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
        row["judge_cost"] = cost

        if cost is not None:
            total_judge_cost += cost
            known_cost_count += 1

        print(f"  {evaluation.output.label}: {evaluation.output.reasoning}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"results_judged_{timestamp}.json"
    save_results(results, output_file)

    if known_cost_count:
        print(f"\nJudge cost: ${total_judge_cost:.4f}")
    else:
        print("\nJudge cost: unavailable for this model")

    print(f"Saved judged results to {output_file}")
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge GapFinder evaluation results with an LLM."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to a results_*.json file from run_scenarios.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVALUATION_DIR,
        help="Directory for results_judged_*.json.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_JUDGE_MODEL,
        help="Judge model, e.g. openai:gpt-4o-mini.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Judge only the first N rows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results_path = args.results or latest_results_file(EVALUATION_DIR)
    judge_results(
        results_path=results_path,
        output_dir=args.output_dir,
        model=args.model,
        limit=args.limit,
    )
