.PHONY: ingest app run tests tests-judge test-all run_scenarios label_streamlit llm_judge

ingest:
	uv run python gapfinder_agent/ingest.py "https://www.youtube.com/watch?v=wjZofJX0v4M"

app:
	uv run python -m streamlit run gapfinder_agent/app.py

run:
	uv run python -m gapfinder_agent.main

tests:
	uv run pytest tests/test_agent.py -s

tests-judge:
	uv run pytest tests/test_judge.py -n 4

test-all:
	uv run pytest

run_scenarios:
	uv run python evals/run_scenarios.py

label_streamlit:
	uv run streamlit run evals/label_streamlit.py

llm_judge:
	uv run python evals/llm_judge.py
