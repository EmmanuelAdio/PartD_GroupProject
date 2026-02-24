# agents/answerer.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
# Knowledge DB (where courses/accommodation live)
MONGODB_KNOWLEDGE_DB = os.getenv("MONGODB_KNOWLEDGE_DB") or os.getenv("MONGODB_DB") or "open_day_knowledge"

# Strict-mode toggle:
# - False (default): if requested_fields is present, ONLY show requested sections (no base metadata)
# - True: include base metadata even in strict mode
STRICT_INCLUDE_METADATA = os.getenv("STRICT_INCLUDE_METADATA", "0") == "1"

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set. Put it in your environment or .env file.")

client = MongoClient(MONGODB_URI)
db = client[MONGODB_KNOWLEDGE_DB]


# ----------------------------
# Output format
# ----------------------------
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


# ----------------------------
# Helpers: text + tokens
# ----------------------------
def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _tokenize(s: str) -> List[str]:
    s = _norm(s)
    s = re.sub(r"[^a-z0-9\s\-\+]", " ", s)
    toks = [t for t in s.split() if len(t) >= 2]
    return toks


def _safe_get_url(doc: Dict[str, Any]) -> str:
    return (
        (doc.get("official_url") or "").strip()
        or (doc.get("source_url") or "").strip()
        or (doc.get("doc_id") or "").strip()
        or "—"
    )


# ----------------------------
# DB helpers
# ----------------------------
def _list_collection_names_safe() -> List[str]:
    try:
        return db.list_collection_names()
    except Exception:
        return []


def _find_by_object_id(collection: str, oid: str) -> Optional[Dict[str, Any]]:
    if not oid:
        return None
    try:
        return db[collection].find_one({"_id": ObjectId(oid)})
    except Exception:
        return None


def _find_one_exact(collection: str, field: str, query: str) -> Optional[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return None
    rx = re.compile(rf"^{re.escape(q)}$", re.I)
    return db[collection].find_one({field: rx})


def _find_one_contains(collection: str, field: str, query: str) -> Optional[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return None
    rx = re.compile(re.escape(q), re.I)
    return db[collection].find_one({field: rx})


def _find_one_token_and(collection: str, field: str, query: str, min_tokens: int = 2) -> Optional[Dict[str, Any]]:
    toks = _tokenize(query)
    if len(toks) < min_tokens:
        return None
    and_clauses = [{field: re.compile(re.escape(t), re.I)} for t in toks[:6]]
    return db[collection].find_one({"$and": and_clauses})


# ----------------------------
# Accommodation formatting
# ----------------------------
def _format_prices(room_types: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for rt in room_types or []:
        rt_name = rt.get("name", "Room")
        ensuite = "En-suite" if rt.get("ensuite") else "Shared bathroom"
        tenancy = rt.get("tenancy_weeks")
        tenancy_str = f"{tenancy} weeks" if tenancy else ""

        prices = rt.get("prices") or []
        price_str = ""
        if prices and isinstance(prices, list):
            p = prices[-1] if prices else {}
            year = p.get("year", "")
            per_week = p.get("per_week_gbp")
            total = p.get("total_contract_gbp")
            if isinstance(per_week, (int, float)) and isinstance(total, (int, float)):
                price_str = f"{year}: £{per_week:.2f}/week, £{total:.2f} total"
            else:
                price_str = str(year).strip()

        line = f"- {rt_name} ({ensuite}{', ' + tenancy_str if tenancy_str else ''})"
        if price_str:
            line += f" — {price_str}"
        lines.append(line)

    return "\n".join(lines) if lines else "No room pricing data available."


def _format_hall_answer(h: Dict[str, Any], requested_fields: List[str]) -> str:
    """
    Precision rule:
    - If requested_fields is non-empty => STRICT mode: only return requested sections.
    - If requested_fields empty => full "profile" answer.
    """
    name = h.get("name", "Hall")
    url = _safe_get_url(h)

    rf = set(requested_fields or [])
    strict = bool(rf)

    short = (h.get("short_description") or "").strip()
    address = (h.get("address") or "").strip()
    catering = (h.get("catering_type") or "").strip()
    facilities = ", ".join(h.get("facilities") or [])
    services = "; ".join(h.get("services") or [])
    email = (h.get("contact_email") or "").strip()
    phone = (h.get("contact_phone") or "").strip()

    if strict:
        sections: List[str] = [f"**{name}**"]  # keep title for halls (usually helpful)

        if "facilities" in rf:
            sections.append(f"**Facilities:** {facilities or '—'}")
            sections.append(f"**Cleaning/services:** {services or '—'}")

        if "contact" in rf:
            sections.append(f"**Contact:** {email or '—'} | {phone or '—'}")

        if "room_types" in rf or "accommodation_prices" in rf:
            prices_block = _format_prices(h.get("room_types") or [])
            sections.append(f"**Room types & prices:**\n{prices_block}")

        if len(sections) == 1:
            prices_block = _format_prices(h.get("room_types") or [])
            sections.append(f"**Room types & prices:**\n{prices_block}")

        sections.append(f"**Official page:** {url}")
        return "\n\n".join(sections)

    # Non-strict: full answer
    sections2: List[str] = [f"**{name}**"]
    if short:
        sections2.append(short)
    sections2.append(f"**Address:** {address or '—'}")
    sections2.append(f"**Catering:** {catering or '—'}")
    sections2.append(f"**Facilities:** {facilities or '—'}")
    sections2.append(f"**Cleaning/services:** {services or '—'}")
    prices_block = _format_prices(h.get("room_types") or [])
    sections2.append(f"**Room types & prices:**\n{prices_block}")
    sections2.append(f"**Contact:** {email or '—'} | {phone or '—'}")
    sections2.append(f"**Official page:** {url}")
    return "\n\n".join([s for s in sections2 if str(s).strip() != ""])


# ----------------------------
# Course formatting
# ----------------------------
def _format_ucas(ucas: Any) -> str:
    if ucas is None:
        return "—"
    if isinstance(ucas, str):
        return ucas.strip() or "—"
    if isinstance(ucas, list):
        vals = [v.strip() for v in ucas if isinstance(v, str) and v.strip()]
        return ", ".join(vals) if vals else "—"
    return str(ucas)


def _format_fees(fees: Any) -> str:
    if not isinstance(fees, dict):
        return "—"

    lines: List[str] = []
    for k in ["uk", "international"]:
        block = fees.get(k)
        if isinstance(block, dict):
            heading = block.get("heading") or k.upper()
            kv = []
            for kk, vv in block.items():
                if kk == "heading":
                    continue
                kv.append(f"{kk}: {vv}")
            if kv:
                lines.append(f"- {heading}: " + " | ".join(kv))
            else:
                lines.append(f"- {heading}")
    return "\n".join(lines) if lines else "—"


def _format_modules(course: Dict[str, Any]) -> str:
    cby = course.get("course_content_by_year")
    if not isinstance(cby, list) or not cby:
        return "No module breakdown found in the database for this course."

    lines: List[str] = []
    for year_block in cby:
        if not isinstance(year_block, dict):
            continue
        year_title = year_block.get("year") or year_block.get("title") or year_block.get("label") or "Year"
        lines.append(f"**{year_title}**")

        mods = (
            year_block.get("modules")
            or year_block.get("module_list")
            or year_block.get("content")
            or year_block.get("items")
        )

        if isinstance(mods, list):
            printed = 0
            for m in mods:
                if printed >= 25:
                    break
                if isinstance(m, str) and m.strip():
                    lines.append(f"- {m.strip()}")
                    printed += 1
                elif isinstance(m, dict):
                    nm = m.get("name") or m.get("title")
                    if isinstance(nm, str) and nm.strip():
                        lines.append(f"- {nm.strip()}")
                        printed += 1
        elif isinstance(mods, str) and mods.strip():
            lines.append(mods.strip())
        else:
            maybe_text = year_block.get("description") or year_block.get("summary")
            if isinstance(maybe_text, str) and maybe_text.strip():
                lines.append(maybe_text.strip())
            else:
                lines.append("- (No module list stored for this year)")

        lines.append("")

    return "\n".join(lines).strip()


def _format_course_answer(course: Dict[str, Any], intent: Optional[str], requested_fields: List[str]) -> str:
    """
    Precision rule:
    - If requested_fields is non-empty => STRICT mode: only return requested sections.
      (By default, strict mode does NOT include base metadata.)
    - If requested_fields empty => intent-based default.
    """
    title = course.get("course_title") or course.get("title") or "Course"
    degree = course.get("degree_classification") or course.get("degree") or "—"
    subject = course.get("subject_area") or course.get("subject") or "—"
    ucas = _format_ucas(course.get("ucas_codes") or course.get("ucas_code"))
    start_date = course.get("start_date") or "—"
    url = _safe_get_url(course)

    overview = (course.get("course_overview") or course.get("overview") or "").strip() or "—"
    entry = course.get("typical_offer") or course.get("entry_requirements") or "—"
    fees = course.get("fees")

    base = (
        f"**{title}**\n"
        f"**Degree:** {degree}\n"
        f"**Subject area:** {subject}\n"
        f"**UCAS codes:** {ucas}\n"
        f"**Start date:** {start_date}\n"
    )

    rf = set(requested_fields or [])
    strict = bool(rf)

    # STRICT mode: only requested sections
    if strict:
        sections: List[str] = []

        if STRICT_INCLUDE_METADATA:
            sections.append(base)

        if "overview" in rf:
            sections.append(f"**Overview:**\n{overview}")

        if "entry_requirements" in rf:
            sections.append(f"**Typical offer / entry requirements:**\n{entry}")

        if "fees" in rf:
            sections.append(f"**Fees (if available):**\n{_format_fees(fees)}")

        if "modules" in rf or "course_content" in rf:
            sections.append(f"**Modules / course content (if available):**\n{_format_modules(course)}")

        if not sections:
            # fall back using intent
            if intent == "ask_fees":
                sections.append(f"**Fees (if available):**\n{_format_fees(fees)}")
            elif intent == "ask_entry_requirements":
                sections.append(f"**Typical offer / entry requirements:**\n{entry}")
            else:
                sections.append(f"**Overview:**\n{overview}")

        sections.append(f"**Official page:** {url}")
        return "\n\n".join(sections)

    # Non-strict: intent-based default (includes metadata)
    sections2: List[str] = [base]

    if intent == "ask_fees":
        sections2.append(f"**Fees (if available):**\n{_format_fees(fees)}")
        sections2.append(f"**Official page:** {url}")
        return "\n\n".join(sections2)

    if intent == "ask_entry_requirements":
        sections2.append(f"**Typical offer / entry requirements:**\n{entry}")
        sections2.append(f"**Official page:** {url}")
        return "\n\n".join(sections2)

    if intent == "ask_course_info":
        sections2.append(f"**Overview:**\n{overview}")
        sections2.append(f"**Modules / course content (if available):**\n{_format_modules(course)}")
        sections2.append(f"**Official page:** {url}")
        return "\n\n".join(sections2)

    sections2.append(f"**Overview:**\n{overview}")
    sections2.append(f"**Official page:** {url}")
    return "\n\n".join(sections2)


# ----------------------------
# Course & hall DB-side finders
# ----------------------------
def _find_course_by_ucas(ucas_code: str) -> Optional[Dict[str, Any]]:
    u = (ucas_code or "").strip()
    if not u:
        return None
    rx = re.compile(rf"^{re.escape(u)}$", re.I)
    doc = db["undergraduate_courses"].find_one({"ucas_codes": rx})
    if not doc:
        doc = db["undergraduate_courses"].find_one({"ucas_code": rx})
    return doc


def _find_course_by_title(query: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return None, {"strategy": "no_query"}

    doc = _find_one_exact("undergraduate_courses", "course_title", q)
    if doc:
        return doc, {"strategy": "exact_title"}

    doc = _find_one_contains("undergraduate_courses", "course_title", q)
    if doc:
        return doc, {"strategy": "contains_title"}

    doc = _find_one_token_and("undergraduate_courses", "course_title", q, min_tokens=2)
    if doc:
        return doc, {"strategy": "token_and_title"}

    return None, {"strategy": "no_match"}


def _find_hall_by_name(query: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return None, {"strategy": "no_query"}

    doc = _find_one_exact("accommodation", "name", q)
    if doc:
        return doc, {"strategy": "exact_name"}

    doc = _find_one_contains("accommodation", "name", q)
    if doc:
        return doc, {"strategy": "contains_name"}

    doc = _find_one_token_and("accommodation", "name", q, min_tokens=2)
    if doc:
        return doc, {"strategy": "token_and_name"}

    return None, {"strategy": "no_match"}


def _extract_ucas_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\b[A-Z]{1,2}\d{3}\b", text)
    return m.group(0) if m else None


# ----------------------------
# Main Answerer
# ----------------------------
def answer(processed: Dict[str, Any]) -> Dict[str, Any]:
    raw_text = processed.get("raw_text", "") or ""
    clean_text = processed.get("clean_text", "") or ""
    domain = processed.get("domain")
    intent = processed.get("intent")
    slots = processed.get("slots") or {}
    requested_fields = processed.get("requested_fields") or []

    pdebug = processed.get("_debug") or {}
    resolved = pdebug.get("resolved") or {}
    resolved_id = resolved.get("resolved_id")
    resolved_collection = resolved.get("collection")

    conf = processed.get("confidence") or {}
    base_conf = 0.6 * float(conf.get("intent", 0.0) or 0.0) + 0.4 * float(conf.get("domain", 0.0) or 0.0)

    debug: Dict[str, Any] = {
        "domain": domain,
        "intent": intent,
        "requested_fields": requested_fields,
        "collections": _list_collection_names_safe(),
        "resolved": resolved,
    }

    if not domain:
        return asdict(AnswererOutput(
            raw_text=raw_text,
            clean_text=clean_text,
            domain=None,
            intent=intent,
            answer="I couldn’t determine the topic of your question.",
            sources=[],
            confidence=max(0.1, base_conf),
            debug={**debug, "reason": "domain_missing"},
        ))

    # ----------------------------
    # Accommodation
    # ----------------------------
    if domain == "accommodation":
        coll = resolved_collection or "accommodation"
        hall = _find_by_object_id(coll, resolved_id) if resolved_id else None
        if hall:
            debug["strategy"] = "resolved_id"
            return asdict(AnswererOutput(
                raw_text=raw_text,
                clean_text=clean_text,
                domain=domain,
                intent=intent,
                answer=_format_hall_answer(hall, requested_fields),
                sources=[{
                    "title": hall.get("name", "Accommodation"),
                    "url": _safe_get_url(hall),
                    "snippet": (hall.get("short_description") or "")[:240]
                }],
                confidence=min(1.0, base_conf + 0.25),
                debug=debug,
            ))

        if slots.get("hall_name"):
            hall_query = slots["hall_name"][0]
        elif slots.get("entity"):
            hall_query = slots["entity"][0]
        else:
            hall_query = clean_text

        hall, match_dbg = _find_hall_by_name(hall_query)
        debug.update(match_dbg)
        debug["hall_query"] = hall_query

        if hall:
            debug["strategy"] = "db_name_match"
            return asdict(AnswererOutput(
                raw_text=raw_text,
                clean_text=clean_text,
                domain=domain,
                intent=intent,
                answer=_format_hall_answer(hall, requested_fields),
                sources=[{
                    "title": hall.get("name", "Accommodation"),
                    "url": _safe_get_url(hall),
                    "snippet": (hall.get("short_description") or "")[:240]
                }],
                confidence=min(1.0, base_conf + 0.2),
                debug=debug,
            ))

        return asdict(AnswererOutput(
            raw_text=raw_text,
            clean_text=clean_text,
            domain=domain,
            intent=intent,
            answer="I couldn’t find a matching accommodation hall. Try the full hall name (e.g., 'Butler Court').",
            sources=[],
            confidence=max(0.1, base_conf),
            debug={**debug, "strategy": "no_hall_match"},
        ))

    # ----------------------------
    # Course info (+ fees via requested_fields / intent)
    # ----------------------------
    if domain in {"course_info", "fees_funding"}:
        coll = resolved_collection or "undergraduate_courses"

        course = _find_by_object_id(coll, resolved_id) if resolved_id else None
        if course:
            debug["strategy"] = "resolved_id"
            return asdict(AnswererOutput(
                raw_text=raw_text,
                clean_text=clean_text,
                domain=domain,
                intent=intent,
                answer=_format_course_answer(course, intent, requested_fields),
                sources=[{
                    "title": course.get("course_title", "Undergraduate course"),
                    "url": _safe_get_url(course),
                    "snippet": (course.get("course_overview") or "")[:240]
                }],
                confidence=min(1.0, base_conf + 0.25),
                debug=debug,
            ))

        ucas_query = ""
        if slots.get("ucas_code"):
            ucas_query = slots["ucas_code"][0]
        else:
            maybe_ucas = _extract_ucas_from_text(raw_text)
            if maybe_ucas:
                ucas_query = maybe_ucas

        if ucas_query:
            course = _find_course_by_ucas(ucas_query)
            debug["ucas_query"] = ucas_query
            if course:
                debug["strategy"] = "ucas_match"
                return asdict(AnswererOutput(
                    raw_text=raw_text,
                    clean_text=clean_text,
                    domain=domain,
                    intent=intent,
                    answer=_format_course_answer(course, intent, requested_fields),
                    sources=[{
                        "title": course.get("course_title", "Undergraduate course"),
                        "url": _safe_get_url(course),
                        "snippet": (course.get("course_overview") or "")[:240]
                    }],
                    confidence=min(1.0, base_conf + 0.22),
                    debug=debug,
                ))

        if slots.get("course_title"):
            course_query = slots["course_title"][0]
        elif slots.get("entity"):
            course_query = slots["entity"][0]
        else:
            course_query = clean_text

        course, match_dbg = _find_course_by_title(course_query)
        debug.update(match_dbg)
        debug["course_query"] = course_query

        if course:
            debug["strategy"] = "db_title_match"
            return asdict(AnswererOutput(
                raw_text=raw_text,
                clean_text=clean_text,
                domain=domain,
                intent=intent,
                answer=_format_course_answer(course, intent, requested_fields),
                sources=[{
                    "title": course.get("course_title", "Undergraduate course"),
                    "url": _safe_get_url(course),
                    "snippet": (course.get("course_overview") or "")[:240]
                }],
                confidence=min(1.0, base_conf + 0.2),
                debug=debug,
            ))

        return asdict(AnswererOutput(
            raw_text=raw_text,
            clean_text=clean_text,
            domain=domain,
            intent=intent,
            answer="I couldn’t find a matching undergraduate course. Try the full course title or include the UCAS code (e.g., G400).",
            sources=[],
            confidence=max(0.1, base_conf),
            debug={**debug, "strategy": "no_course_match"},
        ))

    # ----------------------------
    # Other domains not implemented yet
    # ----------------------------
    return asdict(AnswererOutput(
        raw_text=raw_text,
        clean_text=clean_text,
        domain=domain,
        intent=intent,
        answer=f"I can’t answer questions for domain '{domain}' yet (no retrieval/formatting implemented).",
        sources=[],
        confidence=max(0.1, base_conf),
        debug={**debug, "reason": "domain_not_supported"},
    ))