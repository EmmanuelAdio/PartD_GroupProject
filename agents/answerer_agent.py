from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError

try:
    from schemas.models import AnswerCitation, AnswerResult, EvidenceItem, RetrievalQuery
except ImportError:  # pragma: no cover
    from ..schemas.models import AnswerCitation, AnswerResult, EvidenceItem, RetrievalQuery

try:
    from services.llm_services import LLMService
except ImportError:  # pragma: no cover
    from ..services.llm_services import LLMService


class _AnswererLLMOutput(BaseModel):
    """Internal validation model for structured LLM answer output."""

    answer: str
    grounded: bool = True
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_ids: List[int] = Field(default_factory=list)


class AnswererAgent:
    """Grounded answer generator over retrieved `EvidenceItem`s.

    This agent is the final stage in the current RAG pipeline:
    user query -> retrieval plan -> hybrid retrieval -> answer synthesis.

    Design contract:
    - Uses only retrieved evidence as its knowledge source.
    - Prefers structured JSON output from the LLM for robust orchestration.
    - Falls back to deterministic evidence summarization when no LLM key is available.
    """

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        llm_model: str = "gpt-4o-mini",
        max_evidence_items: int = 6,
        max_chars_per_evidence: int = 1200,
    ) -> None:
        self.llm = llm_service
        if self.llm is None and self._has_openai_api_key():
            self.llm = LLMService(model=llm_model)

        self.max_evidence_items = max(1, int(max_evidence_items))
        self.max_chars_per_evidence = max(200, int(max_chars_per_evidence))

    def answer(
        self,
        *,
        user_query: str,
        evidence_items: Sequence[EvidenceItem | Dict[str, Any]],
        processor_plan: Optional[RetrievalQuery] = None,
    ) -> AnswerResult:
        """Generate a grounded answer from retrieval evidence."""
        normalized_query = self._clean_text(user_query)
        if not normalized_query:
            raise ValueError("user_query must not be empty.")

        evidence = self._normalize_evidence(evidence_items)
        if not evidence:
            return AnswerResult(
                answer="I couldn't find enough relevant information in the knowledge base to answer that confidently.",
                grounded=False,
                confidence=0.0,
                citations=[],
                used_evidence_count=0,
                fallback_used=self.llm is None,
            )

        # For min/max accommodation price questions, deterministic comparison is
        # more reliable than asking the LLM to reason over a partial set of prices.
        if self._looks_like_price_extreme_query(normalized_query):
            extreme_answer = self._build_price_extreme_answer(
                user_query=normalized_query,
                evidence=evidence,
            )
            if extreme_answer is not None:
                return extreme_answer

        if self.llm is None:
            return self._fallback_answer(
                user_query=normalized_query,
                evidence=evidence,
            )

        try:
            llm_payload = self.llm.generate_json(
                system_prompt=self._build_system_prompt(),
                user_prompt=self._build_user_prompt(
                    user_query=normalized_query,
                    processor_plan=processor_plan,
                    evidence=evidence,
                ),
                max_tokens=500,
            )
            llm_result = self._coerce_llm_answer(llm_payload=llm_payload, evidence=evidence)
            fallback_result = self._fallback_answer(
                user_query=normalized_query,
                evidence=evidence,
            )
            if self._should_prefer_fallback(
                user_query=normalized_query,
                llm_result=llm_result,
                fallback_result=fallback_result,
            ):
                return fallback_result
            return llm_result
        except Exception:
            # Keep the answering stage robust even if LLM generation fails.
            return self._fallback_answer(
                user_query=normalized_query,
                evidence=evidence,
            )

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "You are the Answerer Agent for a university RAG system.\n"
            "Answer the user's question using only the provided evidence snippets.\n"
            "Do not invent facts that are not directly supported by the evidence.\n"
            "If the evidence is insufficient, say so clearly and set grounded=false.\n"
            "Prefer concise, student-facing answers.\n"
            "Treat structured key-value evidence as factual support.\n"
            "If a snippet contains fields like per_week_gbp, total_contract_gbp, fees, or entry requirements, "
            "use them directly instead of saying the evidence is missing.\n"
            "Return only valid JSON with keys: answer, grounded, confidence, citation_ids.\n"
            "citation_ids must contain the evidence_id integers you actually used.\n"
            "confidence must be a float between 0 and 1."
        )

    def _build_user_prompt(
        self,
        *,
        user_query: str,
        processor_plan: Optional[RetrievalQuery],
        evidence: Sequence[EvidenceItem],
    ) -> str:
        plan_payload = processor_plan.model_dump() if processor_plan is not None else None
        evidence_payload = []
        for idx, item in enumerate(evidence[: self.max_evidence_items], start=1):
            evidence_payload.append(
                {
                    "evidence_id": idx,
                    "source_id": item.source_id,
                    "title": item.title,
                    "section": item.section,
                    "domain": item.domain,
                    "entity_tags": item.entity_tags,
                    "score": item.score,
                    "key_fields": item.metadata.get("key_fields", {}) if isinstance(item.metadata, dict) else {},
                    "salient_lines": self._select_salient_lines(user_query, item),
                    "text_excerpt": self._truncate_text_preserve_lines(item.text, self.max_chars_per_evidence),
                }
            )

        return (
            "Generate a grounded final answer for the user.\n\n"
            f"User query:\n{user_query}\n\n"
            f"Processor plan:\n{json.dumps(plan_payload, ensure_ascii=False)}\n\n"
            f"Evidence:\n{json.dumps(evidence_payload, ensure_ascii=False, indent=2)}"
        )

    def _coerce_llm_answer(
        self,
        *,
        llm_payload: Dict[str, Any],
        evidence: Sequence[EvidenceItem],
    ) -> AnswerResult:
        try:
            parsed = _AnswererLLMOutput.model_validate(llm_payload or {})
        except ValidationError:
            return self._fallback_answer(
                user_query="",
                evidence=evidence,
            )

        citations = self._build_citations(parsed.citation_ids, evidence)
        if parsed.grounded and not citations and evidence:
            citations = self._build_citations([1], evidence)

        return AnswerResult(
            answer=self._polish_answer_text(parsed.answer) or "I couldn't form a grounded answer from the retrieved evidence.",
            grounded=bool(parsed.grounded),
            confidence=float(parsed.confidence),
            citations=citations,
            used_evidence_count=len(citations),
            fallback_used=False,
        )

    def _fallback_answer(
        self,
        *,
        user_query: str,
        evidence: Sequence[EvidenceItem],
    ) -> AnswerResult:
        """Deterministic fallback when no LLM is available."""
        extreme_price_answer = self._build_price_extreme_answer(user_query=user_query, evidence=evidence)
        if extreme_price_answer is not None:
            return extreme_price_answer

        price_answer = self._build_price_answer(user_query=user_query, evidence=evidence)
        if price_answer is not None:
            return price_answer

        ranked_lines: List[Dict[str, Any]] = []
        query_lc = user_query.lower()
        query_tokens = self._tokenize(user_query)

        for evidence_id, item in enumerate(evidence[: self.max_evidence_items], start=1):
            raw_lines = [self._clean_text(line) for line in item.text.splitlines() if self._clean_text(line)]
            if not raw_lines:
                raw_lines = [self._clean_text(item.text)]

            for raw_line in raw_lines[:20]:
                score = 0.0
                line_lc = raw_line.lower()
                if query_lc and query_lc in line_lc:
                    score += 4.0
                score += sum(1.0 for token in query_tokens if token in line_lc)
                if any(tag.lower() in query_lc for tag in item.entity_tags):
                    score += 1.0
                if score <= 0 and evidence_id > 2:
                    continue

                ranked_lines.append(
                    {
                        "score": score,
                        "evidence_id": evidence_id,
                        "line": self._format_line_for_user(raw_line),
                    }
                )

        if not ranked_lines:
            citations = self._build_citations([1], evidence)
            return AnswerResult(
                answer="I found some potentially relevant evidence, but I couldn't reliably synthesize a grounded answer from it.",
                grounded=False,
                confidence=0.2,
                citations=citations,
                used_evidence_count=len(citations),
                fallback_used=True,
            )

        ranked_lines.sort(key=lambda row: row["score"], reverse=True)
        selected_lines: List[str] = []
        citation_ids: List[int] = []
        seen_lines = set()

        for row in ranked_lines:
            line = row["line"]
            key = line.lower()
            if not line or key in seen_lines:
                continue
            seen_lines.add(key)
            selected_lines.append(line)
            citation_ids.append(int(row["evidence_id"]))
            if len(selected_lines) >= 5:
                break

        citations = self._build_citations(citation_ids, evidence)
        answer = "Based on the retrieved information:\n" + "\n".join(f"- {line}" for line in selected_lines)
        confidence = min(0.85, 0.35 + (0.1 * len(citations)) + (0.04 * len(selected_lines)))

        return AnswerResult(
            answer=answer,
            grounded=True,
            confidence=confidence,
            citations=citations,
            used_evidence_count=len(citations),
            fallback_used=True,
        )

    def _should_prefer_fallback(
        self,
        *,
        user_query: str,
        llm_result: AnswerResult,
        fallback_result: AnswerResult,
    ) -> bool:
        if not fallback_result.grounded or not fallback_result.citations:
            return False

        if llm_result.grounded and llm_result.citations and llm_result.confidence >= 0.35:
            return False

        answer_lc = llm_result.answer.lower()
        if llm_result.used_evidence_count == 0 and fallback_result.used_evidence_count > 0:
            return True
        if not llm_result.grounded and fallback_result.grounded:
            return True
        if llm_result.confidence <= 0.2 < fallback_result.confidence:
            return True
        if self._looks_like_missing_evidence_answer(answer_lc):
            return True
        if self._looks_like_price_extreme_query(user_query):
            return True
        if self._looks_like_price_query(user_query) and "per week" in fallback_result.answer.lower():
            return True
        return False

    def _build_price_extreme_answer(
        self,
        *,
        user_query: str,
        evidence: Sequence[EvidenceItem],
    ) -> Optional[AnswerResult]:
        direction = self._price_extreme_direction(user_query)
        if direction is None or not self._looks_like_price_extreme_query(user_query):
            return None

        candidates = self._collect_price_candidates(evidence)
        if not candidates:
            return None

        best_value = min(item["per_week"] for item in candidates) if direction == "min" else max(
            item["per_week"] for item in candidates
        )
        best_matches = [
            item
            for item in candidates
            if abs(item["per_week"] - best_value) < 1e-9
        ]
        best_matches = self._dedupe_price_candidates(best_matches)
        best_matches.sort(
            key=lambda item: (
                str(item.get("hall_name") or ""),
                str(item.get("room_name") or ""),
            )
        )

        citations = self._build_citations([item["evidence_id"] for item in best_matches], evidence)
        if not citations:
            return None

        comparator = "lowest" if direction == "min" else "highest"
        money = self._format_money(str(best_value))
        years = [str(item["year"]) for item in best_matches if item.get("year")]
        unique_years = sorted({year for year in years if year})

        if len(best_matches) == 1:
            best = best_matches[0]
            subject = best["hall_name"] or "The accommodation"
            room_name = best["room_name"]
            if room_name:
                answer = (
                    f"The {comparator} weekly price I found in the accommodation data is "
                    f"{subject} ({room_name}) at {money} per week"
                )
            else:
                answer = (
                    f"The {comparator} weekly price I found in the accommodation data is "
                    f"{subject} at {money} per week"
                )
        else:
            labels = [self._format_price_candidate_label(item) for item in best_matches]
            answer = (
                f"The {comparator} weekly price I found in the accommodation data is "
                f"{money} per week, shared by {self._join_with_and(labels)}"
            )

        if len(unique_years) == 1:
            answer += f" for {unique_years[0]}"
        answer += "."

        return AnswerResult(
            answer=self._polish_answer_text(answer),
            grounded=True,
            confidence=0.92,
            citations=citations,
            used_evidence_count=len(citations),
            fallback_used=True,
        )

    def _build_price_answer(
        self,
        *,
        user_query: str,
        evidence: Sequence[EvidenceItem],
    ) -> Optional[AnswerResult]:
        if not self._looks_like_price_query(user_query):
            return None

        best_candidate: Optional[Dict[str, Any]] = None
        query_lc = user_query.lower()

        for evidence_id, item in enumerate(evidence[: self.max_evidence_items], start=1):
            fields = self._extract_structured_fields(item.text)
            if not fields:
                continue

            prices = fields.get("prices", {})
            per_week = prices.get("per_week_gbp")
            total_contract = prices.get("total_contract_gbp")
            year = prices.get("year")
            if per_week is None and total_contract is None:
                continue

            matched_name = None
            for tag in item.entity_tags:
                tag_value = self._clean_text(tag)
                if tag_value and tag_value.lower() in query_lc:
                    matched_name = tag_value
                    break

            hall_name = matched_name or fields.get("name")
            room_name = fields.get("room_name")
            relevance = float(item.score or 0.0)
            if matched_name:
                relevance += 5.0
            if hall_name and hall_name.lower() in query_lc:
                relevance += 2.0

            candidate = {
                "evidence_id": evidence_id,
                "hall_name": hall_name,
                "room_name": room_name,
                "per_week": per_week,
                "total_contract": total_contract,
                "year": year,
                "relevance": relevance,
            }
            if best_candidate is None or candidate["relevance"] > best_candidate["relevance"]:
                best_candidate = candidate

        if best_candidate is None:
            return None

        citations = self._build_citations([best_candidate["evidence_id"]], evidence)
        if not citations:
            return None

        subject = best_candidate["hall_name"] or "The accommodation"
        room_name = best_candidate["room_name"]
        if room_name:
            subject = f"{subject} ({room_name})"

        parts = []
        if best_candidate["per_week"] is not None:
            parts.append(f"{subject} is listed at {self._format_money(best_candidate['per_week'])} per week")
        if best_candidate["year"]:
            if parts:
                parts[-1] += f" for {best_candidate['year']}"
            else:
                parts.append(f"The listed academic year is {best_candidate['year']}")
        if best_candidate["total_contract"] is not None:
            parts.append(f"the total contract cost is {self._format_money(best_candidate['total_contract'])}")

        if not parts:
            return None

        answer = parts[0]
        if len(parts) > 1:
            answer += ", and " + parts[1]
        if not answer.endswith("."):
            answer += "."

        return AnswerResult(
            answer=self._polish_answer_text(answer),
            grounded=True,
            confidence=0.86 if best_candidate["per_week"] is not None else 0.74,
            citations=citations,
            used_evidence_count=len(citations),
            fallback_used=True,
        )

    def _build_citations(
        self,
        citation_ids: Sequence[int],
        evidence: Sequence[EvidenceItem],
    ) -> List[AnswerCitation]:
        out: List[AnswerCitation] = []
        seen = set()
        for raw_id in citation_ids:
            try:
                evidence_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if evidence_id < 1 or evidence_id > len(evidence):
                continue
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            item = evidence[evidence_id - 1]
            out.append(
                AnswerCitation(
                    evidence_id=evidence_id,
                    chunk_id=item.chunk_id,
                    source_id=item.source_id,
                    source_type=item.source_type,
                    title=item.title,
                    section=item.section,
                    url=item.url,
                )
            )
        return out

    @staticmethod
    def _normalize_evidence(
        evidence_items: Sequence[EvidenceItem | Dict[str, Any]],
    ) -> List[EvidenceItem]:
        out: List[EvidenceItem] = []
        for item in evidence_items:
            if isinstance(item, EvidenceItem):
                out.append(item)
                continue
            if not isinstance(item, dict):
                continue
            try:
                out.append(EvidenceItem.model_validate(item))
            except ValidationError:
                continue
        return out

    @staticmethod
    def _has_openai_api_key() -> bool:
        return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY"))

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join((text or "").split()).strip()

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        value = AnswererAgent._clean_text(text)
        if len(value) <= limit:
            return value
        return value[: limit - 3].rstrip() + "..."

    @staticmethod
    def _truncate_text_preserve_lines(text: str, limit: int) -> str:
        lines = [line.rstrip() for line in (text or "").splitlines()]
        cleaned_lines = [line for line in lines if line.strip()]
        if not cleaned_lines:
            return AnswererAgent._truncate_text(text, limit)

        out: List[str] = []
        used = 0
        for line in cleaned_lines:
            candidate = line.strip()
            if not candidate:
                continue
            extra = len(candidate) + (1 if out else 0)
            if out and used + extra > limit:
                break
            if not out and len(candidate) > limit:
                return candidate[: limit - 3].rstrip() + "..."
            out.append(candidate)
            used += extra

        value = "\n".join(out)
        if len(value) <= limit:
            return value
        return value[: limit - 3].rstrip() + "..."

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token.lower() for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]*", text or "")]

    @staticmethod
    def _format_line_for_user(line: str) -> str:
        if ":" not in line:
            return line

        path, value = line.split(":", 1)
        path = path.strip()
        value = value.strip()
        if not value:
            return line

        field = re.sub(r"\[\d+\]", "", path)
        field = field.split(".")[-1]
        field = field.replace("_", " ").strip()
        field = field.title() if field else "Field"
        return f"{field}: {value}"

    def _select_salient_lines(
        self,
        user_query: str,
        item: EvidenceItem,
        *,
        max_lines: int = 8,
    ) -> List[str]:
        query_lc = user_query.lower()
        query_tokens = self._tokenize(user_query)
        price_query = self._looks_like_price_query(user_query)
        scored: List[Dict[str, Any]] = []

        raw_lines = [line.strip() for line in item.text.splitlines() if line.strip()]
        for idx, raw_line in enumerate(raw_lines):
            line_lc = raw_line.lower()
            score = 0.0
            if query_lc and query_lc in line_lc:
                score += 5.0
            score += sum(1.0 for token in query_tokens if token in line_lc)
            if any((tag or "").lower() in query_lc for tag in item.entity_tags):
                score += 1.0
            if price_query and any(
                marker in line_lc
                for marker in ("per_week_gbp", "total_contract_gbp", "fee", "fees", "price", "prices", "gbp")
            ):
                score += 4.0
            if score <= 0 and idx >= 3:
                continue
            scored.append(
                {
                    "score": score,
                    "index": idx,
                    "line": self._format_line_for_user(raw_line),
                }
            )

        scored.sort(key=lambda row: (-row["score"], row["index"]))
        out: List[str] = []
        seen = set()
        for row in scored:
            value = row["line"]
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
            if len(out) >= max_lines:
                break
        return out

    @staticmethod
    def _looks_like_missing_evidence_answer(answer_lc: str) -> bool:
        patterns = (
            "does not provide specific",
            "does not provide enough",
            "insufficient evidence",
            "couldn't find enough relevant information",
            "could not find enough relevant information",
            "can't answer that confidently",
            "cannot answer that confidently",
        )
        return any(pattern in answer_lc for pattern in patterns)

    @staticmethod
    def _looks_like_price_query(user_query: str) -> bool:
        query_lc = (user_query or "").lower()
        return any(
            token in query_lc
            for token in (
                "how much",
                "price",
                "prices",
                "cost",
                "costs",
                "fee",
                "fees",
                "per week",
                "weekly",
                "rent",
            )
        )

    @staticmethod
    def _looks_like_price_extreme_query(user_query: str) -> bool:
        query_lc = (user_query or "").lower()
        has_extreme = AnswererAgent._price_extreme_direction(user_query) is not None
        has_price_signal = any(
            token in query_lc
            for token in ("price", "prices", "cost", "costs", "fee", "fees", "rent", "weekly", "per week")
        )
        return has_extreme and has_price_signal

    @staticmethod
    def _price_extreme_direction(user_query: str) -> Optional[str]:
        query_lc = (user_query or "").lower()
        if any(token in query_lc for token in ("cheapest", "lowest", "least expensive", "minimum")):
            return "min"
        if any(token in query_lc for token in ("most expensive", "highest", "maximum", "priciest")):
            return "max"
        return None

    def _collect_price_candidates(self, evidence: Sequence[EvidenceItem]) -> List[Dict[str, Any]]:
        hall_names: Dict[int, str] = {}
        room_names: Dict[tuple[int, int], str] = {}
        years: Dict[tuple[int, int, int], str] = {}

        parsed_by_item: List[Dict[str, Any]] = []
        for evidence_id, item in enumerate(evidence, start=1):
            parsed_lines = self._parse_structured_lines(item.text)
            parsed_by_item.append(
                {
                    "evidence_id": evidence_id,
                    "item": item,
                    "lines": parsed_lines,
                }
            )
            for row in parsed_lines:
                path = row["path"]
                value = row["value"]
                hall_match = re.match(r"^\[(\d+)\]\.name$", path)
                if hall_match:
                    hall_names[int(hall_match.group(1))] = value
                    continue

                room_match = re.match(r"^\[(\d+)\]\.room_types\[(\d+)\]\.name$", path)
                if room_match:
                    hall_idx = int(room_match.group(1))
                    room_idx = int(room_match.group(2))
                    room_names[(hall_idx, room_idx)] = value
                    continue

                year_match = re.match(r"^\[(\d+)\]\.room_types\[(\d+)\]\.prices\[(\d+)\]\.year$", path)
                if year_match:
                    hall_idx = int(year_match.group(1))
                    room_idx = int(year_match.group(2))
                    price_idx = int(year_match.group(3))
                    years[(hall_idx, room_idx, price_idx)] = value

        candidates: List[Dict[str, Any]] = []
        for parsed_item in parsed_by_item:
            evidence_id = parsed_item["evidence_id"]
            item = parsed_item["item"]
            parsed_lines = parsed_item["lines"]
            for row in parsed_lines:
                path = row["path"]
                value = row["value"]
                price_match = re.match(
                    r"^\[(\d+)\]\.room_types\[(\d+)\]\.prices\[(\d+)\]\.per_week_gbp$",
                    path,
                )
                if not price_match:
                    continue

                hall_idx = int(price_match.group(1))
                room_idx = int(price_match.group(2))
                price_idx = int(price_match.group(3))
                per_week = self._coerce_float(value)
                if per_week is None:
                    continue

                hall_name = hall_names.get(hall_idx) or self._infer_hall_name_from_item(item)
                room_name = room_names.get((hall_idx, room_idx))
                year = years.get((hall_idx, room_idx, price_idx))

                candidates.append(
                    {
                        "evidence_id": evidence_id,
                        "hall_idx": hall_idx,
                        "room_idx": room_idx,
                        "price_idx": price_idx,
                        "hall_name": hall_name,
                        "room_name": room_name,
                        "per_week": per_week,
                        "year": year,
                    }
                )
        return candidates

    @staticmethod
    def _parse_structured_lines(text: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            path, value = line.split(":", 1)
            path = path.strip()
            value = value.strip()
            if not path or not value:
                continue
            rows.append({"path": path, "value": value})
        return rows

    @staticmethod
    def _infer_hall_name_from_item(item: EvidenceItem) -> Optional[str]:
        hall_like_tags = []
        for tag in item.entity_tags:
            value = str(tag or "").strip()
            if not value:
                continue
            if any(keyword in value.lower() for keyword in ("court", "hall", "holt")):
                hall_like_tags.append(value)
        return hall_like_tags[0] if hall_like_tags else None

    @staticmethod
    def _dedupe_price_candidates(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for item in candidates:
            key = (
                item.get("hall_idx"),
                item.get("room_idx"),
                item.get("price_idx"),
                item.get("per_week"),
                item.get("hall_name"),
                item.get("room_name"),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    @staticmethod
    def _format_price_candidate_label(candidate: Dict[str, Any]) -> str:
        hall_name = candidate.get("hall_name") or "unknown accommodation"
        room_name = candidate.get("room_name")
        return f"{hall_name} ({room_name})" if room_name else str(hall_name)

    @staticmethod
    def _join_with_and(values: Sequence[str]) -> str:
        parts = [str(value).strip() for value in values if str(value).strip()]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        return f"{', '.join(parts[:-1])}, and {parts[-1]}"

    @staticmethod
    def _extract_structured_fields(text: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {"prices": {}}
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            path, value = line.split(":", 1)
            path = path.strip()
            value = value.strip()
            if not value:
                continue

            field = path.split(".")[-1]
            if field == "name":
                if "room_types" in path:
                    out.setdefault("room_name", value)
                else:
                    out.setdefault("name", value)
            elif field in {"per_week_gbp", "total_contract_gbp", "year"}:
                out["prices"][field] = value
        return out

    @staticmethod
    def _format_money(value: str) -> str:
        amount = str(value or "").strip()
        if not amount:
            return "an unspecified amount"
        normalized = amount.replace("GBP", "").replace("£", "").strip()
        return f"GBP {AnswererAgent._format_numeric_amount(normalized)}"

    @classmethod
    def _polish_answer_text(cls, text: str) -> str:
        value = cls._clean_text(text)
        if not value:
            return value

        value = cls._normalize_currency_amounts(value)
        value = re.sub(
            r",\s*total(?:ing|ling)\s+((?:GBP\s+)?[\d,]+(?:\.\d+)?)\s+for the contract period\.?",
            r", with a total contract cost of \1.",
            value,
            flags=re.IGNORECASE,
        )
        return value

    @staticmethod
    def _normalize_currency_amounts(text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            raw_amount = match.group(1)
            return f"GBP {AnswererAgent._format_numeric_amount(raw_amount)}"

        return re.sub(r"(?:£|GBP\s+)\s?(\d[\d,]*(?:\.\d+)?)", repl, text)

    @staticmethod
    def _format_numeric_amount(amount: str) -> str:
        normalized = str(amount or "").replace(",", "").strip()
        try:
            number = float(normalized)
        except ValueError:
            return normalized

        if "." in normalized:
            return f"{number:,.2f}"
        return f"{int(number):,}"

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            return float(str(value).replace(",", "").strip())
        except (TypeError, ValueError, AttributeError):
            return None
