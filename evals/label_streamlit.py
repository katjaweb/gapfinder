import json
from pathlib import Path

import streamlit as st


EVALUATION_DIR = Path(__file__).resolve().parent


def result_files() -> list[Path]:
    files = sorted(EVALUATION_DIR.glob("results*.json"))
    return [path for path in files if path.is_file()]


def default_result_file(files: list[Path]) -> Path | None:
    judged = [path for path in files if path.name.startswith("results_judged_")]
    if judged:
        return judged[-1]
    plain = [path for path in files if path.name.startswith("results_")]
    return plain[-1] if plain else None


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        st.error(f"File not found: {path}")
        st.stop()

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        st.error("Result file must contain a JSON list.")
        st.stop()

    return data


def save_results(data: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def label_icon(item: dict) -> str:
    label = item.get("human_label") or item.get("label")
    if label == "good":
        return "[good]"
    if label == "bad":
        return "[bad]"
    return "[open]"


def compact_title(item: dict, index: int) -> str:
    category = item.get("category", "unknown")
    scenario_type = item.get("type", "unknown")
    return f"{label_icon(item)} {index + 1}. {category} / {scenario_type}"


def markdown_or_json(value):
    if isinstance(value, str):
        st.markdown(value)
    else:
        st.json(value)


st.set_page_config(layout="wide", page_title="GapFinder Evaluation Labeling")
st.title("GapFinder Evaluation Labeling")

files = result_files()
if not files:
    st.error("No results*.json files found. Run evaluation/run_scenarios.py first.")
    st.stop()

default_file = default_result_file(files)
default_index = files.index(default_file) if default_file in files else 0

selected_file = st.sidebar.selectbox(
    "Results file",
    options=files,
    index=default_index,
    format_func=lambda path: path.name,
)

results_path = Path(selected_file)
data = load_results(results_path)

if not data:
    st.info("This results file is empty.")
    st.stop()

state_key = f"current_idx::{results_path.name}"
if state_key not in st.session_state:
    st.session_state[state_key] = 0

st.sidebar.header("Navigate")
selected = st.sidebar.selectbox(
    "Scenario",
    options=list(range(len(data))),
    format_func=lambda i: compact_title(data[i], i),
    index=st.session_state[state_key],
)
if selected != st.session_state[state_key]:
    st.session_state[state_key] = selected
    st.rerun()

col_prev, col_next = st.sidebar.columns(2)
with col_prev:
    if st.button("Prev", use_container_width=True):
        st.session_state[state_key] = max(0, st.session_state[state_key] - 1)
        st.rerun()
with col_next:
    if st.button("Next", use_container_width=True):
        st.session_state[state_key] = min(len(data) - 1, st.session_state[state_key] + 1)
        st.rerun()

current_idx = st.session_state[state_key]
item = data[current_idx]

labeled = sum(1 for row in data if row.get("human_label") or row.get("label"))
good = sum(1 for row in data if (row.get("human_label") or row.get("label")) == "good")
bad = sum(1 for row in data if (row.get("human_label") or row.get("label")) == "bad")

st.sidebar.markdown(f"**Labeled:** {labeled} / {len(data)}")
st.sidebar.markdown(f"**Good:** {good}")
st.sidebar.markdown(f"**Bad:** {bad}")
st.sidebar.progress((current_idx + 1) / len(data))

st.caption(f"{results_path.name} | Scenario {current_idx + 1} of {len(data)}")

summary_cols = st.columns(5)
summary_cols[0].metric("Category", item.get("category", ""))
summary_cols[1].metric("Type", item.get("type", ""))
summary_cols[2].metric("Expected", item.get("expected_quality", ""))
summary_cols[3].metric("Agent", item.get("model", ""))
summary_cols[4].metric("Cost", item.get("cost", ""))

left, right = st.columns([1, 1])

with left:
    st.subheader("Scenario")
    st.markdown("**User prompt**")
    st.write(item.get("question", ""))

    st.markdown("**Learner answer**")
    st.write(item.get("user_answer", ""))

    st.markdown("**Judge criteria**")
    st.write(item.get("judge_criteria", ""))

    with st.expander("Tool calls", expanded=False):
        st.json(item.get("tool_calls", []))

    with st.expander("Tool context", expanded=False):
        st.json(item.get("tool_context", []))

with right:
    st.subheader("Agent Response")
    markdown_or_json(item.get("output", ""))

    if item.get("judge_label"):
        st.subheader("LLM Judge")
        judge_cols = st.columns(3)
        judge_cols[0].metric("Label", item.get("judge_label", ""))
        judge_cols[1].metric(
            "Expected Match",
            str(item.get("judge_matched_expected_quality", "")),
        )
        judge_cols[2].metric("Feedback", item.get("judge_feedback_quality", ""))
        st.markdown("**Reasoning**")
        st.write(item.get("judge_reasoning", ""))

        with st.expander("Judge details", expanded=False):
            st.json(
                {
                    "followed_tool_policy": item.get("judge_followed_tool_policy"),
                    "judge_model": item.get("judge_model"),
                    "judge_tokens": item.get("judge_tokens"),
                    "judge_cost": item.get("judge_cost"),
                }
            )

st.divider()

st.subheader("Human Label")
existing_label = item.get("human_label") or item.get("label")
label_options = ["not_labeled", "good", "bad"]
label = st.radio(
    "Is the agent response good?",
    label_options,
    index=label_options.index(existing_label) if existing_label in label_options else 0,
    horizontal=True,
)

comments = st.text_area(
    "Comments",
    value=item.get("human_comments", item.get("comments", "")),
)

save_col, status_col = st.columns([1, 4])
with save_col:
    if st.button("Save label", type="primary", use_container_width=True):
        item["human_label"] = None if label == "not_labeled" else label
        item["human_comments"] = comments
        save_results(data, results_path)
        st.success("Saved.")

with status_col:
    if item.get("judge_label"):
        agreement = item.get("judge_label") == (None if label == "not_labeled" else label)
        st.info(f"Human label agrees with LLM judge: {agreement}")
