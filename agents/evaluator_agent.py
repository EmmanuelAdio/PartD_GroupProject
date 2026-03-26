from __future__ import annotations

import json
import os
import re
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

from pydantic import ValidationError

try:
    from schemas.models import (
        AnswerResult,
        EvaluationResult,
        EvaluationVerdict,
        EvidenceItem,
        LLMJudgeResult,
        RetrievalQuery,
        RuleCheckResult,
    )
except ImportError:  # pragma: no cover
    from ..schemas.models import (  # type: ignore[no-redef]
        AnswerResult,
        EvaluationResult,
        EvaluationVerdict,
        EvidenceItem,
        LLMJudgeResult,
        RetrievalQuery,
        RuleCheckResult,
    )

try:
    from services.llm_services import LLMService
except ImportError:  # pragma: no cover
    from ..services.llm_services import LLMService  # type: ignore[no-redef]


class EvaluatorAgent:
    """Verification agent for query relevance, grounding, clarity, and safety.

    This agent is designed to sit after the AnswererAgent in a multi-agent loop:
    - Rule checks run first (fast and deterministic).
    - Optional LLM judge runs second only for ambiguous/borderline cases.
    - Returns structured feedback for orchestration decisions.
    """

    MIN_EVIDENCE_ITEMS = 2
    MIN_TOP_SCORE = 0.45
    MIN_AVG_TOP3_SCORE = 0.30
    MIN_ANSWER_CHARS = 24
    MIN_FOCUS_TOKEN_OVERLAP = 1

    _QUERY_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "could",
        "do",
        "for",
        "from",
        "give",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "please",
        "show",
        "tell",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "to",
        "us",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    }
    _OVERCONFIDENT_PHRASES = (
        "definitely",
        "certainly",
        "guaranteed",
        "always",
        "never",
        "100%",
    )
    _UNCERTAINTY_PHRASES = (
        "may",
        "might",
        "could",
        "likely",
        "based on the retrieved",
        "from the available information",
        "please check official",
        "i do not know",
    )

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        llm_model: str = "gpt-4o-mini",
        max_evidence_items_for_judge: int = 6,
        max_chars_per_evidence: int = 900,
    ) -> None:
        self.llm = llm_service
        if self.llm is None and self._has_openai_api_key():
            self.llm = LLMService(model=llm_model)
        self.max_evidence_items_for_judge = max(1, int(max_evidence_items_for_judge))
        self.max_chars_per_evidence = max(200, int(max_chars_per_evidence))

    def evaluate(
        self,
        *,
        user_query: str,
        retrieval_query: Optional[RetrievalQuery],
        evidence: Sequence[EvidenceItem | Dict[str, Any]],
        draft_answer: AnswerResult | Dict[str, Any],
    ) -> EvaluationResult:
        """Run evaluation and return a structured orchestration-ready verdict."""
        normalized_query = self._clean_text(user_query)
        if not normalized_query:
            raise ValueError("user_query must not be empty.")

        evidence_items = self._normalize_evidence(evidence)
        draft = self._normalize_draft(draft_answer)

        rule_result = self.run_rule_checks(
            user_query=normalized_query,
            retrieval_query=retrieval_query,
            evidence=evidence_items,
            draft_answer=draft,
        )

        llm_result: Optional[LLMJudgeResult] = None
        if self.should_call_llm_judge(
            rule_result=rule_result,
            evidence=evidence_items,
            retrieval_query=retrieval_query,
        ):
            llm_result = self.run_llm_judge(
                user_query=normalized_query,
                retrieval_query=retrieval_query,
                evidence=evidence_items,
                draft_answer=draft,
            )

        return self.decide_verdict(
            user_query=normalized_query,
            retrieval_query=retrieval_query,
            rule_result=rule_result,
            llm_result=llm_result,
        )

    def run_rule_checks(
        self,
        *,
        user_query: str,
        retrieval_query: Optional[RetrievalQuery],
        evidence: Sequence[EvidenceItem],
        draft_answer: AnswerResult,
    ) -> RuleCheckResult:
        """Deterministic first-line checks for obvious failure cases."""
        issues: List[str] = []
        grounded = True
        relevant = True
        clear = True
        safe = True

        answer_text = self._clean_text(draft_answer.answer)
        evidence_scores = [float(item.score or 0.0) for item in evidence]
        top_score = max(evidence_scores) if evidence_scores else 0.0
        avg_top3 = mean(sorted(evidence_scores, reverse=True)[:3]) if evidence_scores else 0.0
        evidence_count = len(evidence)

        no_evidence = evidence_count == 0
        weak_evidence = (
            no_evidence
            or evidence_count < self.MIN_EVIDENCE_ITEMS
            or top_score < self.MIN_TOP_SCORE
            or avg_top3 < self.MIN_AVG_TOP3_SCORE
        )
        ambiguous_query = self._is_ambiguous_query(user_query=user_query, retrieval_query=retrieval_query)

        if no_evidence:
            grounded = False
            relevant = False
            issues.append("no_evidence_returned")
        elif weak_evidence:
            issues.append("weak_retrieval_evidence")

        if len(answer_text) < self.MIN_ANSWER_CHARS or self._looks_vague_answer(answer_text):
            clear = False
            issues.append("answer_too_vague_or_short")

        token_overlap = self._focus_overlap(user_query=user_query, answer_text=answer_text)
        if token_overlap < self.MIN_FOCUS_TOKEN_OVERLAP:
            relevant = False
            issues.append("low_query_answer_overlap")

        focus_issue = self._check_query_focus_alignment(user_query=user_query, answer_text=answer_text)
        if focus_issue:
            relevant = False
            issues.append(focus_issue)

        citations_valid = self._citations_are_valid(draft_answer=draft_answer, evidence=evidence)
        numeric_supported = self._numeric_claims_supported(answer_text=answer_text, evidence=evidence)
        lexical_support = self._answer_evidence_overlap(answer_text=answer_text, evidence=evidence)

        if draft_answer.grounded and not citations_valid:
            grounded = False
            issues.append("missing_or_invalid_citations")
        if draft_answer.grounded and not numeric_supported:
            grounded = False
            issues.append("unsupported_numeric_claims")
        if draft_answer.grounded and lexical_support < 0.05 and evidence_count > 0:
            grounded = False
            issues.append("low_answer_evidence_overlap")

        if weak_evidence and (draft_answer.confidence >= 0.75 or self._has_overconfident_language(answer_text)):
            safe = False
            issues.append("overconfident_with_weak_evidence")
        if weak_evidence and not self._has_uncertainty_language(answer_text) and draft_answer.confidence >= 0.60:
            safe = False
            issues.append("missing_uncertainty_when_evidence_weak")

        if ambiguous_query:
            issues.append("ambiguous_user_query")

        suggested_filters = None
        if weak_evidence or not relevant:
            suggested_filters = self._suggested_filters_from_query(
                user_query=user_query,
                retrieval_query=retrieval_query,
            )

        clarification_question = self._build_clarification_question(
            user_query=user_query,
            retrieval_query=retrieval_query,
        ) if ambiguous_query else None

        if no_evidence:
            suggested_action = "use_safe_fallback"
        elif ambiguous_query and not relevant:
            suggested_action = "ask_user_clarification"
        elif not grounded or not relevant or not clear or not safe:
            suggested_action = "retry_with_adjusted_filters" if weak_evidence else "revise_answer"
        else:
            suggested_action = "accept_answer"

        needs_llm_judge = (
            not no_evidence
            and (weak_evidence or ambiguous_query or (relevant and clear and safe and not grounded))
        )

        notes = (
            f"Evidence count={evidence_count}, top_score={top_score:.3f}, "
            f"avg_top3={avg_top3:.3f}, overlap={token_overlap}, lexical_support={lexical_support:.3f}"
        )

        return RuleCheckResult(
            grounded=grounded,
            relevant=relevant,
            clear=clear,
            safe=safe,
            issues=self._dedupe_str(issues),
            no_evidence=no_evidence,
            weak_evidence=weak_evidence,
            ambiguous_query=ambiguous_query,
            needs_llm_judge=needs_llm_judge,
            suggested_action=suggested_action,
            suggested_filters=suggested_filters,
            clarification_question=clarification_question,
            notes=notes,
            metrics={
                "evidence_count": evidence_count,
                "top_score": top_score,
                "avg_top3_score": avg_top3,
                "focus_token_overlap": token_overlap,
                "lexical_support": lexical_support,
            },
        )

    def should_call_llm_judge(
        self,
        *,
        rule_result: RuleCheckResult,
        evidence: Sequence[EvidenceItem],
        retrieval_query: Optional[RetrievalQuery],
    ) -> bool:
        """Gate expensive second-stage judging to ambiguous/borderline cases."""
        if self.llm is None:
            return False
        if not rule_result.needs_llm_judge:
            return False
        if rule_result.no_evidence:
            return False
        if not evidence:
            return False
        _ = retrieval_query  # reserved for future routing logic
        return True

    def run_llm_judge(
        self,
        *,
        user_query: str,
        retrieval_query: Optional[RetrievalQuery],
        evidence: Sequence[EvidenceItem],
        draft_answer: AnswerResult,
    ) -> Optional[LLMJudgeResult]:
        """Optional structured LLM judge.

        Integration note:
        - This method is intentionally isolated so teams can later swap in
          OpenAI Responses API, LangChain judges, or rubric variants without
          touching rule checks/orchestration policy.
        """
        if self.llm is None:
            return None

        evidence_payload = []
        for idx, item in enumerate(evidence[: self.max_evidence_items_for_judge], start=1):
            evidence_payload.append(
                {
                    "evidence_id": idx,
                    "chunk_id": item.chunk_id,
                    "source_id": item.source_id,
                    "domain": item.domain,
                    "section": item.section,
                    "score": item.score,
                    "text_excerpt": self._truncate_text(item.text, self.max_chars_per_evidence),
                }
            )

        user_prompt = (
            "Evaluate the draft answer quality for a university open-day assistant.\n"
            "Return JSON only.\n\n"
            f"User query:\n{user_query}\n\n"
            f"Retrieval plan:\n{json.dumps(retrieval_query.model_dump() if retrieval_query else None, ensure_ascii=False)}\n\n"
            f"Draft answer:\n{json.dumps(draft_answer.model_dump(), ensure_ascii=False)}\n\n"
            f"Evidence:\n{json.dumps(evidence_payload, ensure_ascii=False, indent=2)}"
        )

        system_prompt = (
            "You are an Evaluator Agent in a multi-agent RAG system.\n"
            "Assess if the draft answer is grounded in evidence, relevant to the query, clear for open-day visitors, "
            "and safe about uncertainty.\n"
            "If evidence is weak, do not allow overconfident claims.\n"
            "Return only JSON with keys: grounded, relevant, clear, safe, issues, suggested_verdict, "
            "suggested_action, suggested_filters, clarification_question, notes.\n"
            "suggested_verdict must be one of: pass, revise, ask_clarification, fallback."
        )

        try:
            payload = self.llm.generate_json(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=350,
            )
            return LLMJudgeResult.model_validate(payload or {})
        except (ValidationError, Exception):
            # Fail-open: if the judge call fails, rule-based evaluation remains authoritative.
            return None

    def decide_verdict(
        self,
        *,
        user_query: str,
        retrieval_query: Optional[RetrievalQuery],
        rule_result: RuleCheckResult,
        llm_result: Optional[LLMJudgeResult],
    ) -> EvaluationResult:
        """Combine rule checks (authoritative) with optional LLM judge output."""
        grounded = bool(rule_result.grounded)
        relevant = bool(rule_result.relevant)
        clear = bool(rule_result.clear)
        safe = bool(rule_result.safe)
        issues = list(rule_result.issues)

        if llm_result is not None:
            if llm_result.grounded is False:
                grounded = False
                issues.append("llm_judge_grounding_concern")
            if llm_result.relevant is False:
                relevant = False
                issues.append("llm_judge_relevance_concern")
            if llm_result.clear is False:
                clear = False
                issues.append("llm_judge_clarity_concern")
            if llm_result.safe is False:
                safe = False
                issues.append("llm_judge_safety_concern")
            issues.extend(llm_result.issues or [])

        verdict: EvaluationVerdict
        if rule_result.no_evidence:
            verdict = "fallback"
        elif rule_result.ambiguous_query and not relevant:
            verdict = "ask_clarification"
        elif not safe and (rule_result.weak_evidence or not grounded):
            verdict = "fallback"
        elif not grounded:
            verdict = "fallback" if rule_result.weak_evidence else "revise"
        elif not relevant or not clear or not safe:
            verdict = "ask_clarification" if rule_result.ambiguous_query else "revise"
        elif rule_result.weak_evidence:
            verdict = "revise"
        else:
            verdict = "pass"

        if llm_result is not None and llm_result.suggested_verdict and not rule_result.no_evidence:
            llm_verdict = llm_result.suggested_verdict
            if llm_verdict in {"fallback", "ask_clarification"}:
                verdict = llm_verdict
            elif llm_verdict == "revise" and verdict == "pass":
                verdict = "revise"
            elif llm_verdict == "pass" and verdict == "revise" and grounded and relevant and clear and safe:
                verdict = "pass"

        suggested_action = rule_result.suggested_action
        if llm_result is not None and llm_result.suggested_action:
            suggested_action = llm_result.suggested_action

        suggested_filters = rule_result.suggested_filters
        if llm_result is not None and llm_result.suggested_filters:
            suggested_filters = llm_result.suggested_filters

        clarification_question = rule_result.clarification_question
        if llm_result is not None and llm_result.clarification_question:
            clarification_question = llm_result.clarification_question
        if verdict == "ask_clarification" and not clarification_question:
            clarification_question = self._build_clarification_question(
                user_query=user_query,
                retrieval_query=retrieval_query,
            )

        notes = rule_result.notes
        if llm_result is not None and llm_result.notes:
            notes = f"{notes} | llm_judge: {llm_result.notes}" if notes else llm_result.notes

        return EvaluationResult(
            verdict=verdict,
            grounded=grounded,
            relevant=relevant,
            clear=clear,
            safe=safe,
            issues=self._dedupe_str(issues),
            suggested_action=suggested_action,
            suggested_filters=suggested_filters,
            clarification_question=clarification_question,
            notes=notes,
        )

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
    def _normalize_draft(draft_answer: AnswerResult | Dict[str, Any]) -> AnswerResult:
        if isinstance(draft_answer, AnswerResult):
            return draft_answer
        if isinstance(draft_answer, dict):
            try:
                return AnswerResult.model_validate(draft_answer)
            except ValidationError:
                pass
        return AnswerResult(
            answer="",
            grounded=False,
            confidence=0.0,
            citations=[],
            used_evidence_count=0,
            fallback_used=True,
        )

    @classmethod
    def _check_query_focus_alignment(cls, *, user_query: str, answer_text: str) -> Optional[str]:
        query_lc = user_query.lower()
        answer_lc = answer_text.lower()

        if cls._is_price_query(query_lc) and not cls._answer_mentions_price(answer_lc):
            return "query_focus_missing_price"
        if cls._is_location_query(query_lc) and not cls._answer_mentions_location(answer_lc):
            return "query_focus_missing_location"
        if cls._is_requirements_query(query_lc) and not cls._answer_mentions_requirements(answer_lc):
            return "query_focus_missing_requirements"
        return None

    @classmethod
    def _suggested_filters_from_query(
        cls,
        *,
        user_query: str,
        retrieval_query: Optional[RetrievalQuery],
    ) -> Optional[Dict[str, Any]]:
        query_lc = user_query.lower()
        domain_hints: List[str] = []
        section_hints: List[str] = []

        if cls._is_price_query(query_lc):
            section_hints.extend(["prices", "fees"])
            domain_hints.extend(["accommodation", "finance"])
        if cls._is_requirements_query(query_lc):
            section_hints.append("entry_requirements")
            domain_hints.extend(["courses", "admissions"])
        if "module" in query_lc:
            section_hints.append("modules")
            domain_hints.append("courses")
        if "facility" in query_lc or "facilities" in query_lc:
            section_hints.append("facilities")
        if "sport" in query_lc:
            domain_hints.append("sports")

        domains = cls._dedupe_str(domain_hints)
        sections = cls._dedupe_str(section_hints)
        out: Dict[str, Any] = {}

        if domains:
            out["domains"] = domains
        if sections:
            out["sections"] = sections

        if retrieval_query is not None:
            if retrieval_query.top_k < 12:
                out["top_k"] = min(12, retrieval_query.top_k + 2)
        elif domains or sections:
            out["top_k"] = 10

        return out or None

    @classmethod
    def _build_clarification_question(
        cls,
        *,
        user_query: str,
        retrieval_query: Optional[RetrievalQuery],
    ) -> str:
        query_lc = user_query.lower()
        has_entity_hint = bool(retrieval_query and retrieval_query.entity_tags)

        if cls._is_price_query(query_lc) and not has_entity_hint:
            return "Could you clarify which accommodation hall or course you mean?"
        if cls._is_requirements_query(query_lc) and not has_entity_hint:
            return "Which specific course are you asking about entry requirements for?"
        if cls._is_location_query(query_lc) and not has_entity_hint:
            return "Which hall, building, or department location are you asking about?"
        return "Could you clarify the exact programme, hall, or topic you want details on?"

    @classmethod
    def _is_ambiguous_query(cls, *, user_query: str, retrieval_query: Optional[RetrievalQuery]) -> bool:
        focus_tokens = cls._focus_query_tokens(user_query)
        query_tokens = cls._tokenize(user_query)
        has_entity_hint = bool(retrieval_query and retrieval_query.entity_tags)

        pronouns = {"it", "this", "that", "there", "they"}
        has_pronoun = any(token in pronouns for token in query_tokens)

        if len(focus_tokens) <= 1 and not has_entity_hint:
            return True
        if has_pronoun and len(focus_tokens) <= 2 and not has_entity_hint:
            return True
        if cls._is_price_query(user_query.lower()) and not has_entity_hint and len(focus_tokens) <= 2:
            return True
        return False

    @classmethod
    def _focus_overlap(cls, *, user_query: str, answer_text: str) -> int:
        focus_tokens = cls._focus_query_tokens(user_query)
        if not focus_tokens:
            return 0
        answer_lc = answer_text.lower()
        return sum(1 for token in focus_tokens if token in answer_lc)

    @staticmethod
    def _citations_are_valid(*, draft_answer: AnswerResult, evidence: Sequence[EvidenceItem]) -> bool:
        if not draft_answer.citations:
            return False
        for citation in draft_answer.citations:
            if citation.evidence_id < 1 or citation.evidence_id > len(evidence):
                return False
        return True

    @classmethod
    def _numeric_claims_supported(cls, *, answer_text: str, evidence: Sequence[EvidenceItem]) -> bool:
        answer_numbers = cls._extract_numbers(answer_text)
        if not answer_numbers:
            return True
        evidence_text = "\n".join(item.text for item in evidence)
        evidence_numbers = cls._extract_numbers(evidence_text)
        evidence_set = {number.replace(",", "") for number in evidence_numbers}
        for value in answer_numbers:
            if value.replace(",", "") not in evidence_set:
                return False
        return True

    @classmethod
    def _answer_evidence_overlap(cls, *, answer_text: str, evidence: Sequence[EvidenceItem]) -> float:
        answer_tokens = {token for token in cls._tokenize(answer_text) if len(token) >= 4}
        if not answer_tokens:
            return 0.0
        evidence_tokens: set[str] = set()
        for item in evidence:
            evidence_tokens.update(token for token in cls._tokenize(item.text) if len(token) >= 4)
        if not evidence_tokens:
            return 0.0
        return len(answer_tokens & evidence_tokens) / float(max(1, len(answer_tokens)))

    @staticmethod
    def _extract_numbers(text: str) -> List[str]:
        return re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", text or "")

    @classmethod
    def _has_overconfident_language(cls, answer_text: str) -> bool:
        text_lc = answer_text.lower()
        return any(phrase in text_lc for phrase in cls._OVERCONFIDENT_PHRASES)

    @classmethod
    def _has_uncertainty_language(cls, answer_text: str) -> bool:
        text_lc = answer_text.lower()
        return any(phrase in text_lc for phrase in cls._UNCERTAINTY_PHRASES)

    @staticmethod
    def _looks_vague_answer(answer_text: str) -> bool:
        value = answer_text.lower()
        patterns = (
            "i am not sure",
            "not enough information",
            "it depends",
            "cannot determine",
            "not certain",
        )
        return any(pattern in value for pattern in patterns)

    @staticmethod
    def _is_price_query(query_lc: str) -> bool:
        return any(
            token in query_lc
            for token in ("price", "cost", "fee", "fees", "rent", "per week", "weekly", "how much")
        )

    @staticmethod
    def _is_location_query(query_lc: str) -> bool:
        return any(token in query_lc for token in ("where", "location", "address", "located", "map"))

    @staticmethod
    def _is_requirements_query(query_lc: str) -> bool:
        return any(
            token in query_lc
            for token in ("entry requirement", "requirements", "ucas", "grades", "qualification")
        )

    @staticmethod
    def _answer_mentions_price(answer_lc: str) -> bool:
        return any(token in answer_lc for token in ("gbp", "£", "per week", "fee", "cost", "price"))

    @staticmethod
    def _answer_mentions_location(answer_lc: str) -> bool:
        return any(token in answer_lc for token in ("located", "address", "campus", "hall", "building"))

    @staticmethod
    def _answer_mentions_requirements(answer_lc: str) -> bool:
        return any(token in answer_lc for token in ("requirement", "ucas", "grade", "qualification", "entry"))

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join((text or "").split()).strip()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token.lower() for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]*", text or "")]

    @classmethod
    def _focus_query_tokens(cls, text: str) -> List[str]:
        out: List[str] = []
        seen = set()
        for token in cls._tokenize(text):
            if len(token) <= 2 or token in cls._QUERY_STOPWORDS:
                continue
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        value = EvaluatorAgent._clean_text(text)
        if len(value) <= limit:
            return value
        return value[: limit - 3].rstrip() + "..."

    @staticmethod
    def _has_openai_api_key() -> bool:
        return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY"))

    @staticmethod
    def _dedupe_str(values: Sequence[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for raw in values:
            value = str(raw or "").strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
        return out
