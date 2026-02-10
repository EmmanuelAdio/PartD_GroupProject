from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

from processor_LLM import processor_agent


def _load_questions() -> List[str]:
    repo_root = Path(__file__).resolve().parent.parent
    questions_path = repo_root / "ExampleQuestions.txt"
    questions = questions_path.read_text(encoding="utf-8").splitlines()
    return [q.strip() for q in questions if q.strip()]


def _validate_output(output: Dict) -> None:
    required_keys = {
        "raw_text",
        "clean_text",
        "domain",
        "intent",
        "slots",
        "retrieval_query",
        "confidence",
    }
    missing = required_keys.difference(output.keys())
    assert not missing, f"Missing keys: {sorted(missing)}"
    assert isinstance(output["raw_text"], str)
    assert isinstance(output["clean_text"], str)
    assert output["domain"] is None or isinstance(output["domain"], str)
    assert output["intent"] is None or isinstance(output["intent"], str)
    assert isinstance(output["slots"], dict)
    assert isinstance(output["retrieval_query"], str)
    assert isinstance(output["confidence"], dict)


def test_process_example_questions() -> None:
    questions = _load_questions()

    # print("\n\n Questions: " + ", ".join(questions))  # Debug: print loaded questions
    assert questions, "No questions found in ExampleQuestions.txt"

    for question in questions:
        output = processor_agent.process(question)
        _validate_output(output)
        print(f"\n\n Question: {question}\n Output: {output}")  # Debug: print output for each question
        

if __name__ == "__main__":
    test_process_example_questions()
    print("All example questions processed successfully.")
