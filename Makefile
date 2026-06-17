.PHONY: tests run streamlit

app:
	uv run python -m streamlit run gapfinder_agent/app.py

run:
	uv run python -m gapfinder_agent.main

tests:
	uv run pytest tests/test_agent.py -s

tests-judge:
	uv run pytest tests/test_judge.py -n 4

run_scenarios:
	uv run python evals/run_scenarios.py

label_streamlit:
	uv run streamlit run evals/label_streamlit.py

llm_judge:
	uv run python evals/llm_judge.py