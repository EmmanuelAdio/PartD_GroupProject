import json
import re
from typing import List, Tuple, Dict, Optional


FLAG_MAP = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL
}

def load_intent_patterns(path: str) -> List[Tuple[str, re.Pattern]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    compiled = []

    for intent_block in data:
        intent = intent_block["intent"]
        for p in intent_block["patterns"]:
            flags = 0
            for flag in p.get("flags", []):
                flags |= FLAG_MAP.get(flag, 0)

            compiled.append(
                (intent, re.compile(p["regex"], flags))
            )

    return compiled