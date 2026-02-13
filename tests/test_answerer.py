import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.processor_LLM import processor_agent
from agents.answerer import answer


def run_demo(question: str):
    print("=" * 80)
    print("QUESTION:")
    print(question)
    print()

    processed = processor_agent.process(question)
    resp = answer(processed)

    print("PROCESSOR OUTPUT:")
    print(processed)
    print()

    print("ANSWER:")
    print(resp["answer"])
    print()

    print("SOURCES:")
    for s in resp.get("sources", []):
        print("-", s)

    print()
    print("CONFIDENCE:", resp.get("confidence"))
    print("DEBUG:", resp.get("debug"))
    print("=" * 80)
    print()


if __name__ == "__main__":
    questions = [
        "Tell me about Butler Court accommodation and prices",
        "Is Butler Court good for social students?",
        "What halls are budget friendly?",
        "Where should an undergraduate stay?",
    ]

    for q in questions:
        run_demo(q)
