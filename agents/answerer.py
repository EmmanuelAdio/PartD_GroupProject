# agents/answerer.py
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "open_day_knowledge")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set. Put it in your environment or .env file.")

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]


@dataclass
class AnswererOutput:
    raw_text: str
    clean_text: str
    domain: Optional[str]
    intent: Optional[str]
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    debug: Dict[str, Any]


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _get_all_halls() -> List[Dict[str, Any]]:
    """
    Your 'accommodation' collection currently stores ONE document containing many halls:
      { "_id": ..., "0": {...Butler Court...}, "1": {...}, ... }

    We read that single document, drop _id, and return a list of hall dicts.
    """
    doc = db["accommodation"].find_one() or {}
    doc.pop("_id", None)

    halls: List[Dict[str, Any]] = []
    for _, v in doc.items():
        if isinstance(v, dict) and v.get("name"):
            halls.append(v)
    return halls


def _looks_like_accommodation(text: str) -> bool:
    t = _norm(text)
    keywords = [
        "accommodation", "accomodation", "hall", "halls", "room", "rooms",
        "ensuite", "en-suite", "en suite",
        "self catered", "self-catered", "catered",
        "tenancy", "contract", "deposit", "rent", "price", "cost", "per week",
        "shared bathroom", "laundry", "laundrette",
        "kitchen", "wardrobe", "desk", "wifi"
    ]
    return any(k in t for k in keywords)


def _hall_name_in_text(halls: List[Dict[str, Any]], text: str) -> bool:
    t = _norm(text)
    for h in halls:
        name = _norm(h.get("name", ""))
        if name and name in t:
            return True
    return False


def _find_halls_by_name(halls: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """
    Find halls whose name appears in the text (substring match).
    Also supports partial matches (e.g., 'butler' matches 'butler court').
    """
    t = _norm(text)
    if not t:
        return []

    hits: List[Dict[str, Any]] = []
    for h in halls:
        name = _norm(h.get("name", ""))
        if not name:
            continue
        # exact substring either way
        if name in t or t in name:
            hits.append(h)
            continue
        # partial token match (e.g. 'butler' in 'butler court')
        tokens = [tok for tok in t.split() if len(tok) >= 4]
        if any(tok in name for tok in tokens):
            hits.append(h)
    return hits


def _format_prices(room_types: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for rt in room_types or []:
        rt_name = rt.get("name", "Room")
        ensuite = "En-suite" if rt.get("ensuite") else "Shared bathroom"
        tenancy = rt.get("tenancy_weeks")
        tenancy_str = f"{tenancy} weeks" if tenancy else ""

        prices = rt.get("prices") or []
        price_str = ""
        if prices:
            # take last entry (often the newest)
            p = prices[-1]
            year = p.get("year", "")
            per_week = p.get("per_week_gbp")
            total = p.get("total_contract_gbp")

            if isinstance(per_week, (int, float)) and isinstance(total, (int, float)):
                price_str = f"{year}: £{per_week:.2f}/week, £{total:.2f} total"
            else:
                price_str = f"{year}".strip()

        line = f"- {rt_name} ({ensuite}{', ' + tenancy_str if tenancy_str else ''})"
        if price_str:
            line += f" — {price_str}"
        lines.append(line)

    return "\n".join(lines) if lines else "No room pricing data available."


def _format_hall_answer(h: Dict[str, Any]) -> str:
    name = h.get("name", "Hall")
    short = (h.get("short_description") or "").strip()
    address = (h.get("address") or "").strip()
    catering = (h.get("catering_type") or "").strip()
    tags = ", ".join(h.get("tags") or [])
    lifestyle = ", ".join(h.get("lifestyle_tags") or [])
    facilities = ", ".join(h.get("facilities") or [])
    room_features = ", ".join(h.get("room_features_common") or [])
    services = "; ".join(h.get("services") or [])

    prices_block = _format_prices(h.get("room_types") or [])
    url = (h.get("official_url") or "").strip()
    email = (h.get("contact_email") or "").strip()
    phone = (h.get("contact_phone") or "").strip()

    return (
        f"**{name}**\n"
        f"{short}\n\n"
        f"**Address:** {address or '—'}\n"
        f"**Catering:** {catering or '—'}\n"
        f"**Tags:** {tags or '—'}\n"
        f"**Lifestyle:** {lifestyle or '—'}\n\n"
        f"**Facilities:** {facilities or '—'}\n"
        f"**Common room features:** {room_features or '—'}\n"
        f"**Cleaning/services:** {services or '—'}\n\n"
        f"**Room types & prices:**\n{prices_block}\n\n"
        f"**Contact:** {email or '—'} | {phone or '—'}\n"
        f"**Official page:** {url or '—'}\n"
    )


def answer(processed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Answerer uses MongoDB knowledge.

    Current DB reality:
      - Only open_day_knowledge.accommodation exists (1 doc containing many halls)

    Behaviour:
      - If processor domain/intent are None, we still try to infer accommodation
        by keywords or hall-name match.
      - If we can identify a hall, return a detailed hall answer.
      - Otherwise return a shortlist (when possible).
    """
    raw_text = processed.get("raw_text", "") or ""
    clean_text = processed.get("clean_text", "") or ""
    domain = processed.get("domain")
    intent = processed.get("intent")
    slots = processed.get("slots") or {}
    retrieval_query = processed.get("retrieval_query") or clean_text

    conf = processed.get("confidence") or {}
    base_conf = 0.6 * float(conf.get("intent", 0.0) or 0.0) + 0.4 * float(conf.get("domain", 0.0) or 0.0)

    halls = _get_all_halls()

    # ✅ NEW: infer domain if processor failed
    if domain is None:
        if _looks_like_accommodation(clean_text) or _hall_name_in_text(halls, clean_text):
            domain = "accommodation"

    debug: Dict[str, Any] = {
        "domain": domain,
        "intent": intent,
        "hall_count": len(halls),
    }

    # If still no domain, give a helpful fallback
    if domain is None:
        return asdict(AnswererOutput(
            raw_text=raw_text,
            clean_text=clean_text,
            domain=None,
            intent=intent,
            answer="I couldn’t determine the topic of your question (domain). Try asking about accommodation, fees, course information, entry requirements, directions, IT support, or the library.",
            sources=[],
            confidence=max(0.1, base_conf),
            debug={**debug, "reason": "domain_none_after_fallback"},
        ))

    # Right now we only have accommodation data loaded
    if domain != "accommodation":
        return asdict(AnswererOutput(
            raw_text=raw_text,
            clean_text=clean_text,
            domain=domain,
            intent=intent,
            answer="I can answer accommodation questions right now. Other knowledge domains haven’t been loaded into MongoDB yet.",
            sources=[],
            confidence=max(0.1, base_conf),
            debug={**debug, "reason": "domain_not_supported_yet"},
        ))

    # --------- Accommodation answering ---------

    # Combine text + slot values to help hall name matching
    slot_terms: List[str] = []
    for values in slots.values():
        if isinstance(values, list):
            slot_terms.extend(values)

    combined = " ".join([clean_text] + slot_terms + [retrieval_query])

    # 1) If a hall name is mentioned, answer that hall
    name_hits = _find_halls_by_name(halls, combined)
    if name_hits:
        hall = name_hits[0]
        return asdict(AnswererOutput(
            raw_text=raw_text,
            clean_text=clean_text,
            domain=domain,
            intent=intent,
            answer=_format_hall_answer(hall),
            sources=[{
                "title": hall.get("name", "Accommodation"),
                "url": hall.get("official_url"),
                "snippet": (hall.get("short_description") or "")[:240]
            }],
            confidence=min(1.0, base_conf + 0.25),
            debug={**debug, "strategy": "hall_name_match", "matched_hall": hall.get("name")},
        ))

    # 2) Otherwise, give a shortlist (basic tag preference)
    q = _norm(combined)
    wanted_tags = []
    if any(w in q for w in ["budget", "cheap", "affordable"]):
        wanted_tags.append("budget")
    if any(w in q for w in ["close", "near", "distance", "campus"]):
        wanted_tags.append("close_to_campus")
    if "social" in q:
        wanted_tags.append("social")
    if any(w in q for w in ["undergraduate", "first year", "first-year"]):
        wanted_tags.append("undergraduate")

    def score(h: Dict[str, Any]) -> int:
        s = 0
        hall_tags = set(_norm(t) for t in (h.get("tags") or []))
        life_tags = set(_norm(t) for t in (h.get("lifestyle_tags") or []))
        for t in wanted_tags:
            if t in hall_tags or t in life_tags:
                s += 1
        return s

    ranked = sorted(halls, key=score, reverse=True)
    top = ranked[:3] if ranked else []

    if top:
        lines = ["Here are a few accommodation options I found:\n"]
        sources = []
        for h in top:
            lines.append(f"- **{h.get('name')}** — {(h.get('short_description') or '')[:140].rstrip()}…")
            lines.append(f"  Tags: {', '.join(h.get('tags') or [])} | Lifestyle: {', '.join(h.get('lifestyle_tags') or [])}")
            lines.append(f"  Official page: {h.get('official_url','')}")
            lines.append("")
            sources.append({
                "title": h.get("name"),
                "url": h.get("official_url"),
                "snippet": (h.get("short_description") or "")[:240]
            })

        return asdict(AnswererOutput(
            raw_text=raw_text,
            clean_text=clean_text,
            domain=domain,
            intent=intent,
            answer="\n".join(lines).strip(),
            sources=sources,
            confidence=min(1.0, base_conf + 0.15),
            debug={**debug, "strategy": "shortlist", "wanted_tags": wanted_tags},
        ))

    # 3) No hall data
    return asdict(AnswererOutput(
        raw_text=raw_text,
        clean_text=clean_text,
        domain=domain,
        intent=intent,
        answer="I couldn’t find any accommodation entries in the database yet.",
        sources=[],
        confidence=max(0.1, base_conf),
        debug={**debug, "strategy": "no_data"},
    ))
