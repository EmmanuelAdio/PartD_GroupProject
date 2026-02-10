import json
from typing import Dict, List, Any


class GazetteerLoader:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> Dict[str, Dict[str, List[str]]]:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self._normalise(data)

    @staticmethod
    def _normalise(data: Any) -> Dict[str, Dict[str, List[str]]]:
        if isinstance(data, dict):
            return data

        result: Dict[str, Dict[str, List[str]]] = {}

        if not isinstance(data, list):
            return result

        for slot_block in data:
            slot_id = slot_block.get("_id")
            if not slot_id:
                continue

            items = slot_block.get("items", [])
            slot_entries: Dict[str, List[str]] = result.setdefault(slot_id, {})

            for item in items:
                canonical = item.get("canonical")
                if not canonical:
                    continue

                aliases = item.get("aliases", [])
                if not isinstance(aliases, list):
                    aliases = []

                existing = slot_entries.get(canonical, [])
                merged = list(dict.fromkeys(existing + aliases))
                slot_entries[canonical] = merged

        return result
