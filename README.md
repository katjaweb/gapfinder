# GapFinder

## Overview

GapFinder is an AI-assisted learning tool for long-form educational videos. It helps learners discover what they truly understood, what concepts they missed, and which parts of the video deserve a second review.

The assistant uses the video transcript to build knowledge structure, generate tailored comprehension questions, capture learner answers, and identify gaps or misunderstandings. This supports more efficient study and better retention.

---

## Problem Statement

Many learners watch long tutorials and lecture videos without a reliable way to check whether they actually understood the content. They may feel confident but still miss key concepts, leading to inefficient rewatching and shallow learning.

GapFinder addresses this problem by turning video transcripts into a diagnostic learning experience: the system asks focused questions, evaluates answers against the underlying content, and highlights the exact concepts that need review.

---

## What GapFinder Does

1. Fetches a YouTube transcript and metadata.
2. Breaks the transcript into chunks and labels the main concepts.
3. Generates a sequence of diagnostic questions, from comprehension to application.
4. Accepts learner answers and evaluates them against the transcript’s key concepts.
5. Produces a structured feedback report showing:
   - Concepts the learner understood well
   - Concepts they missed
   - Specific sections of the video worth revisiting

---

## System Workflow

### 1. Knowledge Extraction

- Download or access the YouTube transcript
- Create transcript metadata and store it in `data/`
- Split the transcript into searchable chunks for retrieval

### 2. Question Generation

- Generate concept-specific questions instead of generic prompts.
- Include:
  - concept coverage questions
  - explain-in-your-own-words prompts
  - transfer/application questions

### 3. Learner Response

- The learner answers the questions in the chat interface.
- Answers are captured for evaluation.

### 4. Gap Detection

- Compare expected concepts from the transcript with learner answers
- Detect missing concepts and misunderstandings
- Recommend video segments and topics for review

---

## Agent Tools

- `get_video_id` — Extracts the YouTube video ID from a URL and helps select the correct transcript.
- `get_summary` — Summarizes the main concepts and structure from the transcript.
- `search_video_transcript` — Performs a lexical search over transcript chunks to retrieve detailed explanations.
- `evaluate_user_answer` — Grades learner answers using the GapFinder rubric and identifies content gaps.

---

## Architecture

```
gapfinder/
│
├── data/                   # generated transcript and chunk data
│   ├── transcript.json     # created by gapfinder_agent/ingest.py
│   └── yt_chunks.json      # created by gapfinder_agent/ingest.py
│
├── evals/                  # evaluation and labeling tools
│   ├── evaluation.ipynb
│   ├── label_streamlit.py  # Streamlit UI for label comparison
│   ├── llm_judge.py        # judge LLM evaluation pipeline
│   ├── run_scenarios.py    # run test scenarios and collect output
│   ├── results_*.json      # scenario output files
│   ├── results_judged_*.json # judged evaluation output
│   └── scenarios.csv       # test scenarios for evaluation
│
├── gapfinder_agent/        # main application code
│   ├── app.py              # Streamlit chat UI
│   ├── ingest.py           # transcript ingestion and indexing
│   ├── main.py             # terminal agent runner
│   ├── tools.py            # agent tool implementations
│   └── yt_agent.py         # agent setup and orchestration
│
├── notebooks/              # exploratory notebooks and demos
│   ├── 01-setup.ipynb
│   ├── 02-rag.ipynb
│   └── 03-gapfinder.ipynb
│
├── tests/                  # automated tests
│   ├── conftest.py
│   ├── judge.py            # judge configuration
│   ├── test_agent.py       # agent behavior tests
│   ├── test_judge.py       # evaluation tests
│   └── utils.py            # test helpers
│
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## Technology Stack

- Python 3.13+
- `pydantic-ai` for the agent framework
- `openai` for language model inference
- `streamlit` for the interactive UI
- `logfire` for monitoring and observability
- `minsearch` for retrieval over transcript chunks
- `pytest` for automated testing
- `uv` for dependency and runtime management

---

## Setup

1. Install `uv` if you do not already have it:
   https://docs.astral.sh/uv/getting-started/installation/

2. Clone the repository.

3. Create a `.env` file with your API keys:

```env
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
LOGFIRE_TOKEN="YOUR_LOGFIRE_TOKEN"
```

4. Install dependencies:

```bash
uv sync
```

5. Authenticate with Logfire:

```bash
uv run logfire auth
```

---

## Usage

### Run the terminal agent

```bash
make run
```

This starts the agent in the terminal so you can interact with it by chat.

You can also run a specific video URL directly:

```bash
v run python -m gapfinder_agent.main "replace_your_url_here"
```

A good starter prompt is:

```text
What are the main concepts of this video: "your_video_url"?
```

When you are finished, enter `stop`.

### Start the Streamlit UI

```bash
make app
```

This launches the assistant in your browser through Streamlit.

**How to use**

1. Enter a YouTube URL and click **Analyze Video**.
2. Wait while the video is processed.
3. Start chatting with the assistant.
4. When you think you are finished, ask for evaluation of your answers to questions about the video and get your gap report.

---

## Monitoring

This project includes `logfire` integration for telemetry and dashboarding. Authenticate with `logfire auth` before using monitoring features.

Follow the Logfire project URL shown in your terminal after the app starts. There you can view logs and traces of your interaction with the assistant. Learner feedback is collected with thumbs-up/thumbs-down reactions.

---

## Testing

Run the core test suite:

```bash
make tests
```

Run the judge evaluation tests:

```bash
make tests-judge
```

---

## Evaluation

The `evals/` folder contains tools for scenario-based evaluation, human labeling, and automated judge evaluation. Use `python evals/run_scenarios.py` to generate scenario outputs and `python evals/llm_judge.py` to run judge assessments.

---

## Notes

- The system is designed to support learners by surfacing concept-level gaps rather than only providing generic quiz feedback.
- The transcript ingestion pipeline stores results in `data/` and builds a retrieval index for smarter question generation and comparison.
- The evaluation workflow is intended to align agent output with human feedback through both manual labeling and LLM judging.

