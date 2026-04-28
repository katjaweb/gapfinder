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

Step 4 — Gap detection (core innovation)
The system compares:
Expected concepts (from transcript) 
User answers 
And identifies:
Missing concepts 
Misunderstandings 
Shallow explanations

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


## Setup

1. Install uv if you don't have it yet: https://docs.astral.sh/uv/getting-started/installation/

2. Clone this repository (or download the zip and extract it).

3. Create a `.env` file from the template and add your API key:

       cp .env.example .env

4. Install dependencies:

       uv sync

5. Start Jupyter:

       uv run jupyter notebook

## Notebooks

- `notebooks/01-setup.ipynb` - smoke test that confirms your environment works
- `notebooks/02-rag.ipynb` - a minimal RAG baseline you can adapt to your own data

## Data

Put your project data in the `data/` folder. See `notebooks/02-rag.ipynb` for how to load it.
