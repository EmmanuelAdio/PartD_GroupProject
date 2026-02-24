import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.processor_LLM import processor_agent
from agents.answerer import answer

questions = [
    "Tell me about Butler Court prices",
    "What are the entry requirements for Accounting and Finance ?",
    "Tell me the modules for Computer Science BSc",
]

for q in questions:
    processed = processor_agent.process(q)
    resp = answer(processed)

    print("=" * 80)
    print("Q:", q)
    print("Processor:", processed.get("domain"), processed.get("intent"))
    print()
    print(resp["answer"])
    print()
    print("DEBUG:", resp.get("debug"))
    print("=" * 80)
    print()