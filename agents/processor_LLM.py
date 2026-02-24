# agents/processor_LLM.py
from __future__ import annotations

"""
ProcessorAgent_LLM
------------------
This file turns a raw user question into a structured payload the Answerer can use.

Key idea (your current setup):
- You DO NOT have a separate "config DB" for intent_patterns / gazetteer.
- So we rely primarily on OpenAI to extract:
    1) intent (from ALLOWED_INTENTS)
    2) entities (course_title, hall_name, ucas_code, degree)
    3) requested_fields (modules, entry_requirements, accommodation_prices, etc.)
- Optional: entity resolution against your KNOWLEDGE DB (open_day_knowledge) to find the exact document.

Outputs a dict with:
{
  raw_text, clean_text, domain, intent,
  slots, requested_fields, retrieval_query,
  confidence, _debug
}
"""

from dataclasses import dataclass, asdict
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# Ensure project root is importable when running from tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Optional: keep your existing intent-classifier wrapper as a fallback.
# If you don't have utils/llm_intent configured, this still runs.
try:
    from utils.llm_intent import BaseIntentClassifier, NullIntentClassifier, build_intent_classifier_from_env
except Exception:  # pragma: no cover
    BaseIntentClassifier = object  # type: ignore

    class NullIntentClassifier:  # type: ignore
        def classify_intent(self, text: str, labels: List[str]):
            class R:
                intent = None
                confidence = 0.0
            return R()

    def build_intent_classifier_from_env():  # type: ignore
        return NullIntentClassifier()


# OpenAI SDK (required for LLM extraction)
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# -------------------------------------------------------------
# 1) Allowed intents + entities + requested fields
# -------------------------------------------------------------
ALLOWED_INTENTS = {
    "ask_location", "ask_directions", "ask_time",
    "ask_entry_requirements", "ask_course_info",
    "ask_fees", "ask_funding",
    "ask_accommodation",
    "ask_it_help", "ask_library",
    "other"
}

ENTITY_KEYS = ["course_title", "hall_name", "ucas_code", "degree"]

REQUESTED_FIELDS = [
    # course-related
    "modules", "entry_requirements", "fees", "funding",
    "overview", "course_content", "duration", "start_date",

    # accommodation-related
    "accommodation_prices", "room_types", "facilities", "contact",
]


@dataclass
class ProcessorOutput:
    raw_text: str
    clean_text: str
    domain: Optional[str]
    intent: Optional[str]
    slots: Dict[str, List[str]]
    requested_fields: List[str]
    retrieval_query: str
    confidence: Dict[str, float]


class ProcessorAgent_LLM:
    """
    Pipeline:
      A) Normalise question text (lowercase, unicode normalize)
      B) Determine intent + entities + requested_fields via OpenAI
      C) Map intent -> domain
      D) (Optional) Fallback intent classifier if OpenAI not available
      E) (Optional) Entity resolution in MongoDB (find exact doc)
      F) Build retrieval_query (debuggable string)
    """

    def __init__(
        self,
        intent_classifier: Optional["BaseIntentClassifier"] = None,
    ):
        self.intent_classifier = intent_classifier or NullIntentClassifier()
        self.intent_labels = sorted(ALLOWED_INTENTS)

        # --- OpenAI ---
        self._openai_client = None
        self._openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")

        if OpenAI is not None and api_key:
            self._openai_client = OpenAI(api_key=api_key)

        # --- Knowledge DB (where accommodation + undergraduate_courses live) ---
        self._mongo_client = None
        self._knowledge_db = None

        mongo_uri = os.getenv("MONGODB_URI")
        knowledge_db_name = os.getenv("MONGODB_DB", "open_day_knowledge")

        if mongo_uri:
            try:
                self._mongo_client = MongoClient(mongo_uri)
                self._knowledge_db = self._mongo_client[knowledge_db_name]
            except Exception:
                self._mongo_client = None
                self._knowledge_db = None

        # Toggle: resolve entities to a specific document _id (recommended ON)
        self._enable_resolution = (os.getenv("ENABLE_ENTITY_RESOLUTION", "1") == "1")

    # ----------------------------
    # Public API
    # ----------------------------
    def process(self, text: str) -> Dict[str, Any]:
        raw_text = text
        clean_text = self._normalise(text)

        # 1) OpenAI extraction (intent + entities + requested_fields)
        intent: Optional[str] = None
        intent_conf: float = 0.0
        llm_reason: str = ""
        requested_fields: List[str] = []
        llm_entities: Dict[str, List[str]] = {}

        intent_source = "none"

        if self._openai_client is not None:
            intent, llm_entities, requested_fields, intent_conf, llm_reason = self._llm_extract(clean_text)
            if intent:
                intent_source = f"openai:{llm_reason}"

        # 2) fallback intent classifier if OpenAI missing or returned None
        if intent is None:
            try:
                res = self.intent_classifier.classify_intent(clean_text, self.intent_labels)
                if getattr(res, "intent", None) in ALLOWED_INTENTS:
                    intent = res.intent
                    intent_conf = float(getattr(res, "confidence", 0.0) or 0.0)
                    intent_source = "intent_classifier"
            except Exception:
                pass

        # 3) map intent -> domain
        domain = self._map_intent_to_domain(intent)

        # 4) slots come primarily from LLM entities
        slots: Dict[str, List[str]] = {}
        self._merge_entities_into_slots(slots, llm_entities)

        # 5) optional: resolve entities to exact doc in MongoDB
        resolved: Dict[str, Any] = {}
        if self._enable_resolution and self._knowledge_db is not None:
            try:
                resolved = self._resolve_entities(domain, slots)
            except Exception:
                resolved = {}

        # 6) retrieval query (useful for debugging + future embedding search)
        retrieval_query = self._build_retrieval_query(
            clean_text=clean_text,
            intent=intent,
            domain=domain,
            slots=slots,
            requested_fields=requested_fields,
            resolved=resolved,
        )

        output = ProcessorOutput(
            raw_text=raw_text,
            clean_text=clean_text,
            domain=domain,
            intent=intent,
            slots=slots,
            requested_fields=requested_fields,
            retrieval_query=retrieval_query,
            confidence={
                "intent": float(intent_conf or 0.0),
                "domain": 0.8 if domain else 0.0,
            },
        )

        out = asdict(output)
        out["_debug"] = {
            "intent_source": intent_source,
            "llm_reason": llm_reason,
            "openai_model": self._openai_model if self._openai_client else None,
            "resolution_enabled": self._enable_resolution,
            "resolved": resolved,
        }
        return out

    # ----------------------------
    # Internals
    # ----------------------------
    def _normalise(self, text: str) -> str:
        text = (text or "").strip().lower()
        text = unicodedata.normalize("NFKC", text)
        return text

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
        if intent == "other":
            return None

        return None

    # ---- OpenAI extraction ----
    def _llm_extract(self, clean_text: str) -> Tuple[Optional[str], Dict[str, List[str]], List[str], float, str]:
        """
        Returns:
          (intent, entities_dict, requested_fields, confidence, reason)

        entities_dict keys: course_title, hall_name, ucas_code, degree
        """
        assert self._openai_client is not None

        system = (
            "You are an information extraction system for a university open-day assistant.\n"
            "Return STRICT JSON only. No markdown. No commentary.\n"
            "If unsure, set fields to null or empty lists.\n"
        )

        user = {
            "task": "Extract intent, entities, and requested_fields from the user question.",
            "allowed_intents": sorted(ALLOWED_INTENTS),
            "allowed_requested_fields": REQUESTED_FIELDS,
            "schema": {
                "intent": "string | null",
                "confidence": "number 0..1",
                "requested_fields": ["string"],
                "entities": {
                    "course_title": ["string"],
                    "hall_name": ["string"],
                    "ucas_code": ["string"],
                    "degree": ["string"],
                },
                "reason": "short explanation of intent/fields chosen"
            },
            "question": clean_text,
            "hints": [
                "modules/course content -> intent ask_course_info, requested_fields include modules",
                "entry requirements/typical offer -> intent ask_entry_requirements, requested_fields include entry_requirements",
                "tuition fees -> intent ask_fees, requested_fields include fees",
                "accommodation prices -> intent ask_accommodation, requested_fields include accommodation_prices",
                "hall_name should be the accommodation hall name phrase",
                "course_title should be the course name phrase",
                "degree examples: BSc, BSc (Hons), MSci, BA, BEng, MEng",
                "ucas_code examples: G400, G401, GG47, H611",
            ]
        }

        content = None
        try:
            resp = self._openai_client.chat.completions.create(
                model=self._openai_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user)},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
        except Exception:
            resp = self._openai_client.chat.completions.create(
                model=self._openai_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user)},
                ],
                temperature=0,
            )
            content = resp.choices[0].message.content

        if not content:
            return None, {}, [], 0.0, "no_content"

        # Parse JSON
        try:
            data = json.loads(content)
        except Exception:
            data = self._try_parse_json_from_text(content)
            if data is None:
                return None, {}, [], 0.0, "json_parse_failed"

        # intent
        intent = data.get("intent")
        if isinstance(intent, str):
            intent = intent.strip()
        else:
            intent = None
        if intent not in ALLOWED_INTENTS:
            intent = None

        # confidence + reason
        conf = float(data.get("confidence") or 0.0)
        conf = max(0.0, min(1.0, conf))
        reason = str(data.get("reason") or "ok")

        # requested_fields
        rf = data.get("requested_fields") or []
        if isinstance(rf, str):
            rf = [rf]

        requested_fields: List[str] = []
        if isinstance(rf, list):
            for x in rf:
                if isinstance(x, str):
                    xx = x.strip()
                    if xx in REQUESTED_FIELDS and xx not in requested_fields:
                        requested_fields.append(xx)

        # entities
        entities = data.get("entities") or {}
        out_entities: Dict[str, List[str]] = {}

        for k in ENTITY_KEYS:
            vals = entities.get(k) or []
            if isinstance(vals, str):
                vals = [vals]
            if isinstance(vals, list):
                cleaned: List[str] = []
                for v in vals:
                    if isinstance(v, str):
                        vv = v.strip()
                        if vv:
                            cleaned.append(vv)
                deduped = list(dict.fromkeys(cleaned))
                if deduped:
                    out_entities[k] = deduped

        return intent, out_entities, requested_fields, conf, reason

    def _try_parse_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    # ---- merge LLM entities into slots ----
    def _merge_entities_into_slots(self, slots: Dict[str, List[str]], entities: Dict[str, List[str]]) -> None:
        if not entities:
            return

        all_entities: List[str] = []
        for k, vals in entities.items():
            if not vals:
                continue
            slots.setdefault(k, [])
            for v in vals:
                if v not in slots[k]:
                    slots[k].append(v)
                if v not in all_entities:
                    all_entities.append(v)

        # Generic entity list is convenient for answerer matching
        if all_entities:
            slots.setdefault("entity", [])
            for v in all_entities:
                if v not in slots["entity"]:
                    slots["entity"].append(v)

    # ---- OPTIONAL: resolve entities to exact DB document ----
    def _resolve_entities(self, domain: Optional[str], slots: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Resolve extracted entities to a specific DB doc.
        This helps the Answerer avoid "token guessing" and increases accuracy.

        Assumptions (based on your DB screenshots):
          accommodation collection docs have: name, official_url/doc_id, etc.
          undergraduate_courses collection docs have: course_title (or title), source_url/doc_id, etc.
        """
        if self._knowledge_db is None:
            return {}

        # Accommodation: resolve hall_name -> accommodation doc
        if domain == "accommodation":
            names = slots.get("hall_name") or slots.get("entity") or []
            if not names:
                return {}
            query = names[0]

            rx_exact = re.compile(rf"^{re.escape(query)}$", re.I)
            doc = self._knowledge_db["accommodation"].find_one({"name": rx_exact})
            if not doc:
                rx_contains = re.compile(re.escape(query), re.I)
                doc = self._knowledge_db["accommodation"].find_one({"name": rx_contains})

            if doc:
                return {
                    "collection": "accommodation",
                    "resolved_name": doc.get("name"),
                    "resolved_id": str(doc.get("_id")),
                    "doc_id": doc.get("doc_id") or doc.get("official_url"),
                }
            return {}

        # Courses: resolve course_title -> undergraduate_courses doc
        if domain in {"course_info", "fees_funding"}:
            titles = slots.get("course_title") or slots.get("entity") or []
            if not titles:
                return {}
            query = titles[0]

            # Your course docs might use course_title OR title depending on your scraper.
            # We'll try both.
            rx_exact = re.compile(rf"^{re.escape(query)}$", re.I)

            doc = self._knowledge_db["undergraduate_courses"].find_one({"course_title": rx_exact})
            if not doc:
                doc = self._knowledge_db["undergraduate_courses"].find_one({"title": rx_exact})

            if not doc:
                rx_contains = re.compile(re.escape(query), re.I)
                doc = self._knowledge_db["undergraduate_courses"].find_one({"course_title": rx_contains})
                if not doc:
                    doc = self._knowledge_db["undergraduate_courses"].find_one({"title": rx_contains})

            if doc:
                return {
                    "collection": "undergraduate_courses",
                    "resolved_title": doc.get("course_title") or doc.get("title"),
                    "resolved_id": str(doc.get("_id")),
                    "doc_id": doc.get("doc_id") or doc.get("source_url") or doc.get("official_url"),
                    "degree": doc.get("degree") or doc.get("degree_classification") or doc.get("qualification"),
                }
            return {}

        return {}

    # ---- retrieval query ----
    def _build_retrieval_query(
        self,
        clean_text: str,
        intent: Optional[str],
        domain: Optional[str],
        slots: Dict[str, List[str]],
        requested_fields: List[str],
        resolved: Dict[str, Any],
    ) -> str:
        """
        A structured string that makes debugging easy and can later be used
        in embedding/keyword retrieval.
        """
        parts = [
            f"text={clean_text}",
            f"domain={domain or '?'}",
            f"intent={intent or '?'}",
        ]

        if requested_fields:
            parts.append("requested=" + "|".join(requested_fields[:6]))

        for k in ["ucas_code", "degree", "course_title", "hall_name", "entity"]:
            vals = slots.get(k) or []
            if vals:
                parts.append(f"{k}=" + "|".join(vals[:3]))

        if resolved.get("resolved_id"):
            parts.append(f"resolved_id={resolved['resolved_id']}")
        if resolved.get("resolved_name"):
            parts.append(f"resolved_name={resolved['resolved_name']}")
        if resolved.get("resolved_title"):
            parts.append(f"resolved_title={resolved['resolved_title']}")

        return " ; ".join(parts)


# -------------------------------------------------------------------
# Global processor agent instance
# -------------------------------------------------------------------

# Your processor no longer depends on a "config DB".
# It only needs:
#   - OPENAI_API_KEY (for intent/entities)
#   - MONGODB_URI + MONGODB_DB (for entity resolution)
MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB", "open_day_knowledge")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set. Put it in your environment or .env file.")

print("MONGODB_URI found in environment.")
print(f"MONGODB connected. KNOWLEDGE DB: {MONGODB_DB}")

intent_classifier = build_intent_classifier_from_env()

processor_agent = ProcessorAgent_LLM(
    intent_classifier=intent_classifier,
)