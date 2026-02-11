from dataclasses import dataclass, asdict
import sys
from typing import Dict, List, Optional, Tuple
import re
import unicodedata

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv() # loads .env file if present, but doesn't error if it's missing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mongo_db.mongo_store import MongoStore
from utils.llm_intent import BaseIntentClassifier, NullIntentClassifier, build_intent_classifier_from_env


 # -------------------------------------------------------------

ALLOWED_INTENTS = {
    "ask_location", "ask_directions", "ask_time",
    "ask_entry_requirements", "ask_course_info",
    "ask_fees", "ask_funding", "ask_accommodation",
    "ask_it_help", "ask_library", "other"
}

@dataclass
class ProcessorOutput:
    raw_text: str
    clean_text: str
    domain: Optional[str]
    intent: Optional[str]
    slots: Dict[str, List[str]]
    retrieval_query: str
    confidence: Dict[str, float]


class ProcessorAgent_LLM:
    def __init__(
        self,
        gazetteer: Dict[str, Dict[str, List[str]]], # can later be replaced with a DB or more complex structure
        intent_patterns: List[Tuple[str, re.Pattern]], # list of (intent_label, regex_pattern) may be extended later
        intent_classifier: Optional[BaseIntentClassifier] = None,
    ):
        self.gazetteer = gazetteer
        self.intent_patterns = intent_patterns
        self.intent_classifier = intent_classifier or NullIntentClassifier()
        self.intent_labels = sorted({label for label, _ in intent_patterns})

    # ---------- PUBLIC API ----------
    def process(self, text: str) -> Dict:
        raw_text = text
        clean_text = self._normalise(text)

        intent, intent_conf = self._detect_intent(clean_text)
        domain = self._map_intent_to_domain(intent)
        slots = self._extract_slots(clean_text)

        retrieval_query = self._build_retrieval_query(clean_text, intent, domain, slots)

        output = ProcessorOutput(
            raw_text=raw_text,
            clean_text=clean_text,
            domain=domain,
            intent=intent,
            slots=slots,
            retrieval_query=retrieval_query,
            confidence={
                "intent": intent_conf,
                "domain": 0.8 if domain else 0.0  # very rough for now
            }
        )
        return asdict(output)

    # ---------- INTERNAL HELPERS ----------

    def _normalise(self, text: str) -> str:
        text = text.strip().lower()
        # remove weird unicode accents etc.
        text = unicodedata.normalize("NFKC", text)
        return text

    def _detect_intent(self, clean_text: str) -> Tuple[Optional[str], float]:
        for intent_label, pattern in self.intent_patterns:
            if pattern.search(clean_text):
                # later you can make this more nuanced
                return intent_label, 0.9
        llm_result = self.intent_classifier.classify_intent(clean_text, self.intent_labels)
        if llm_result.intent:
            return llm_result.intent, llm_result.confidence
        return None, 0.0

    def _map_intent_to_domain(self, intent: Optional[str]) -> Optional[str]:
        if intent is None:
            return None

        if intent in {"ask_location", "ask_directions"}:
            return "location"
        if intent in {"ask_time"}:
            return "event_info"
        if intent in {"ask_entry_requirements", "ask_course_info"}:
            return "course_info"
        if intent in {"ask_fees", "ask_funding"}:
            return "fees_funding"
        if intent in {"ask_accommodation"}:
            return "accommodation"
        if intent in {"ask_it_help"}:
            return "it_support"
        if intent in {"ask_library"}:
            return "library"

        return None

    def _extract_slots(self, clean_text: str) -> Dict[str, List[str]]:
        found: Dict[str, List[str]] = {}

        for slot_type, entries in self.gazetteer.items():
            for canonical_name, synonyms in entries.items():
                for phrase in synonyms + [canonical_name]:
                    # basic substring match; can be improved later
                    if phrase in clean_text: 
                        found.setdefault(slot_type, [])
                        if canonical_name not in found[slot_type]:
                            found[slot_type].append(canonical_name) 

        # extract free-text phrase after some location patterns as "place"
        place_phrases = self._extract_place_phrases(clean_text)
        if place_phrases:
            found.setdefault("place", [])
            found["place"].extend(place_phrases)

        return found

    def _extract_place_phrases(self, clean_text: str) -> List[str]:
        """
        For questions like 'how do I get to the computer science building?'
        grab the bit after 'get to' (very rough).
        """
        patterns = [
            re.compile(r"how\s+do\s+i\s+get\s+to\s+(.*)"),
            re.compile(r"where\s+is\s+(.*)")
        ]
        results = []
        for p in patterns:
            m = p.search(clean_text)
            if m:
                # strip punctuation-ish characters at the end
                phrase = m.group(1).strip(" ?.!").strip()
                if phrase:
                    results.append(phrase)
        return results

    def _build_retrieval_query(
        self,
        clean_text: str,
        intent: Optional[str],
        domain: Optional[str],
        slots: Dict[str, List[str]]
    ) -> str:
        """
        Build a text query that your retrieval system (FAISS + embeddings)
        can use. For now, just combine detected slots with the question.
        """
        slot_terms = []
        for slot_type, values in slots.items():
            slot_terms.extend(values)

        extra = ""
        if domain == "location":
            extra = "location on campus"
        elif domain == "course_info":
            extra = "course information and entry requirements"
        elif domain == "event_info":
            extra = "date and time"

        parts = [clean_text]
        if slot_terms:
            parts.append(" | ".join(slot_terms))
        if extra:
            parts.append(extra)

        return " ; ".join(parts)


# -------------------------------------------------------------------
# Processor agent setup: load patterns, gazetteer, connect to DB, etc.
# -------------------------------------------------------------------

MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB", "partd_group")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set. Put it in your environment or .env file.")
else:
    print("MONGODB_URI found in environment.")
    print(f"MONGODB connected: {MONGODB_URI}, DB: {MONGODB_DB}")

store = MongoStore(mongo_uri=MONGODB_URI, db_name=MONGODB_DB)

INTENT_PATTERNS = store.load_intent_patterns()
GAZETTEER = store.load_gazetteer_for_slots()

# Instantiate a global processor agent
intent_classifier = build_intent_classifier_from_env()

processor_agent = ProcessorAgent_LLM(
    gazetteer=GAZETTEER,
    intent_patterns=INTENT_PATTERNS,
    intent_classifier=intent_classifier,
)
