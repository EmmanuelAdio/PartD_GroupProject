import os
import re
from typing import Dict, List, Optional, Tuple

from pymongo import MongoClient

FLAG_MAP = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
}

class MongoStore:
    """
    Loads intents + gazetteers from MongoDB and returns them in the shapes
    your ProcessorAgent expects.
    """

    def __init__(self, mongo_uri: str, db_name: str):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.intents_col = self.db["intents"]
        self.gazetteers_col = self.db["gazetteers"]

    def load_intent_patterns(self) -> List[Tuple[str, re.Pattern]]:
        """
        Returns: [(intent_label, compiled_regex), ...]
        Reads documents like:
          { _id, patterns: [{regex, flags}], ... }
        """
        compiled: List[Tuple[str, re.Pattern]] = []

        for doc in self.intents_col.find({}):
            intent_label = doc["_id"]
            for p in doc.get("patterns", []):
                flags = 0
                for f in p.get("flags", []):
                    flags |= FLAG_MAP.get(f, 0)

                compiled.append((intent_label, re.compile(p["regex"], flags)))

        return compiled

    def load_gazetteer_for_slots(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Converts Mongo gazetteer docs into your old JSON shape:

        Mongo docs look like:
          { _id: "campus_location", items: [{canonical, aliases}, ...] }

        Output shape (what _extract_slots expects):
          {
            "campus_location": {
               "haslegrave building": ["haslegrave", "computer science building"],
               ...
            },
            ...
          }
        """
        out: Dict[str, Dict[str, List[str]]] = {}

        for g in self.gazetteers_col.find({}):
            slot_type = g["_id"]
            slot_entries: Dict[str, List[str]] = {}

            for item in g.get("items", []):
                canonical = item["canonical"]
                aliases = item.get("aliases", [])
                slot_entries[canonical] = aliases

            out[slot_type] = slot_entries

        return out
