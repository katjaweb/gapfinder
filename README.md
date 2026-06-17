# GapFinder

## The Problem

People who learn from long-form videos (e.g., students, self-learners, engineers watching tutorials) often feel like they understand the material but can’t identify what they’ve actually missed. This leads to inefficient rewatching and shallow learning because there’s no clear feedback on gaps in understanding.

## What It Does

The user provides a YouTube video link and answers a small set of generated questions about the content. The system analyzes their responses against the video’s key concepts and returns a structured report highlighting what they understood well, what they misunderstood or missed, and which specific parts of the video they should revisit.

## What the system actually does

Input:
YouTube video URL 
User answers to questions 

## System flow

Step 1 — Extract & structure knowledge
Transcribe video 
Break into concepts (chunking + labeling) 

Step 2 — Generate diagnostic questions
Not generic questions — but:
Concept coverage questions 
“Explain in your own words” prompts 
Application questions (transfer knowledge) 

Step 3 — User answers
User types responses

Step 4 — Gap detection
The system compares:
Expected concepts (from transcript) 
User answers 
And identifies:
Missing concepts 
Misunderstandings 
What to revisit

## Agent structure:

Planner Agent
Decides:
Which concepts to test
Which question types to generate

Question Generator Tool
Creates diagnostic questions

Evaluation Tool (LLM-as-judge)
Grades answers against concept checklist

Gap Analyzer Tool
Maps errors → missing concepts


## System architecture

gapfinder/
│
├── data/                   # output saved from gapfinder_agent/ingest.py
│   ├── transcript.json     # will be created with initial run if not exisitng yet
│   └── yt_chunks.json      # will be created with initial run if not exisitng yet
│
├── evals/
│   ├── evaluation.ipynb
│   ├── label_streamlit.py  # UI to label  human and llm feedback
│   ├── llm_judge.py        # llm to judge agent results
│   ├── run_scenarios.py    # run agent on ground truth dataset for evaluation
│   ├── results_20260617_164013.json
│   ├── results_judged_20260617_205710.json
│   └── scenarios.csv       # test scenarios for evaluation
│
├── gapfinder_agent/
│   ├── app.py              # Streamlit UI to chat with agent
│   ├── ingest.py           # YouTube → Transcript → Chunks → Index
│   ├── main.py             # Run agent in temrinal
│   ├── tools.py            # agents tools
│   └── yt_agent.py         # agent setup
│
├── notebooks/
│   ├── 01-setup.ipynb
│   ├── 02-rag.ipynb
│   └── 03-gapfinder.ipynb
│
├── tests/
│   ├── conftest.py
│   ├── judge.py
│   ├── test_agent.py
│   ├── test_judge.py
│   └── tutils.py
│
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock


## Setup

1. Install uv if you don't have it yet: https://docs.astral.sh/uv/getting-started/installation/

2. Clone this repository (or download the zip and extract it).

3. Create a `.env` file and add your OPENAI_API and LOGFIRE_TOKEN key:

       OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
       LOGFIRE_TOKEN="YOUR_LOGFIRE_TOKEN"

4. Install dependencies:

       uv sync

5. Authenticate to logfire

       uv run logfire auth


## Notebooks

