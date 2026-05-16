.PHONY: tests run streamlit

app:
	uv run python -m streamlit run gapfinder_agent/app.py

run:
	uv run python main.py

streamlit:
	uv run python -m streamlit run gapfinder_agent/app.py

tests:
	uv run pytest tests/test_agent.py -s

tests-judge:
	uv run pytest tests/test_judge.py -n 4