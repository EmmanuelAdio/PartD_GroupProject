# Answerer Agent Breakdown

This document explains how the `AnswererAgent` works in simple language first, then in a more technical way for teammates.

Imagine we have:

- a child asking a question
- a librarian who goes to find useful pages in books
- a helper who reads those pages and answers nicely

The `RetrieverService` is the librarian.
The `AnswererAgent` is the helper.

So the flow is:

1. The user asks a question.
2. The retriever finds the best chunks of information from the knowledge base.
3. The `AnswererAgent` looks only at those chunks.
4. It tries to write a final answer using that information.
5. It also says which chunk(s) it used, so we can trace where the answer came from.

Very important:

- the answerer is not supposed to make things up
- it should only answer from what was retrieved
- if it does not have enough information, it should say that

So you can think of it like this:

`question -> retrieved evidence -> answerer reads evidence -> final grounded answer`

## Where It Sits In The System

The `AnswererAgent` lives in [`agents/answerer_agent.py`](/Users/dikachiudeh-joseph/Documents/GitHub/PartD_GroupProject/agents/answerer_agent.py).

It is called from [`app/orchestrator.py`](/Users/dikachiudeh-joseph/Documents/GitHub/PartD_GroupProject/app/orchestrator.py) inside `QueryOrchestrator.run(...)`.

High-level flow:

1. `ProcessorAgent` turns the raw user question into a structured retrieval plan.
2. `RetrieverService` uses that plan to fetch the best evidence chunks.
3. `AnswererAgent.answer(...)` turns those evidence chunks into the final answer.
4. The API returns the answer and citations.

## What Goes Into The Answerer

The main entry point is:

`AnswererAgent.answer(user_query, evidence_items, processor_plan=None)`

Inputs:

- `user_query`: the original question from the user
- `evidence_items`: the retrieved chunks from the knowledge base
- `processor_plan`: the structured retrieval plan created earlier by `ProcessorAgent`

The evidence items use the `EvidenceItem` model from [`schemas/models.py`](/Users/dikachiudeh-joseph/Documents/GitHub/PartD_GroupProject/schemas/models.py).

Each evidence item can contain:

- the chunk text
- source information
- entity tags
- section/domain metadata
- retrieval scores
- helper metadata such as `key_fields`

## What Comes Out

The answerer returns an `AnswerResult`.

This includes:

- `answer`: the final text shown to the user
- `grounded`: whether the answer is supported by evidence
- `confidence`: a simple confidence score between 0 and 1
- `citations`: which evidence items were used
- `used_evidence_count`: how many evidence chunks were cited
- `fallback_used`: whether the agent used the deterministic fallback instead of trusting the LLM result

The citation objects are `AnswerCitation` records, also defined in [`schemas/models.py`](/Users/dikachiudeh-joseph/Documents/GitHub/PartD_GroupProject/schemas/models.py).

## Step-By-Step Logic

### 1. Clean And Validate The Question

Inside `answer(...)`, the first thing the agent does is normalize the question text.

Why:

- remove messy spacing
- prevent empty input
- make matching more stable later

If the question is empty, it raises an error.

### 2. Normalize The Retrieved Evidence

The answerer accepts either:

- real `EvidenceItem` objects
- plain dictionaries that can be converted into `EvidenceItem`

This happens in `_normalize_evidence(...)`.

Why:

- keeps the function flexible
- makes orchestration easier
- ensures the rest of the code can work with one consistent type

### 3. Handle The “No Evidence” Case

If no usable evidence exists, the answerer returns a safe response immediately.

That response says it could not find enough relevant information.

Why:

- better than hallucinating
- clearly tells the caller the system had no support for an answer

### 4. Choose LLM Path Or Fallback Path

The class can work in two modes:

- LLM mode
- deterministic fallback mode

If an OpenAI API key exists, it creates an `LLMService` in the constructor.
If not, it uses the fallback logic only.

This is useful because:

- the system still works without an LLM
- you can test locally
- you have a backup when the LLM fails or behaves badly

## The LLM Path

If an LLM is available, the answerer tries to use it first.

### 5. Build A Strict System Prompt

`_build_system_prompt()` tells the model:

- answer only from the supplied evidence
- do not invent facts
- say when evidence is insufficient
- return JSON only
- include citation IDs

This is important because the answerer is not a general chatbot here.
It is a grounded answer generator.

### 6. Build A User Prompt With Structured Evidence

`_build_user_prompt(...)` prepares:

- the user question
- the processor plan
- the evidence payload

For each evidence chunk, it includes:

- `evidence_id`
- source metadata
- domain/section
- entity tags
- retrieval score
- `key_fields`
- `salient_lines`
- `text_excerpt`

This is one of the most important parts of the design.

Instead of throwing one giant block of raw text at the model, the code gives the LLM a cleaner summary of each chunk.

That helps the model notice important facts like:

- `per_week_gbp`
- `total_contract_gbp`
- fees
- entry requirements

### 7. Parse The Model Output Safely

The LLM must return JSON matching `_AnswererLLMOutput`:

- `answer`
- `grounded`
- `confidence`
- `citation_ids`

This validation happens in `_coerce_llm_answer(...)`.

Why:

- protects the pipeline from malformed model output
- gives the orchestrator a predictable structure

### 8. Build Citations

The answerer maps the returned `citation_ids` back onto the evidence list using `_build_citations(...)`.

This creates real `AnswerCitation` objects.

Why:

- makes the answer traceable
- keeps the UI/API able to show sources

### 9. Polish The Final Text

The answer text is cleaned by `_polish_answer_text(...)`.

This helps:

- normalize money formatting
- rewrite awkward phrasing
- keep the output cleaner for users

Example:

- raw LLM wording might be awkward
- polished output becomes more readable and consistent

## The Fallback Path

The fallback path exists for two reasons:

- there is no LLM available
- the LLM answer is bad even though the evidence is good

### 10. Special Handling For Price Questions

`_build_price_answer(...)` is a targeted fallback for pricing queries.

It looks for signals like:

- `how much`
- `price`
- `fees`
- `cost`
- `per_week_gbp`
- `total_contract_gbp`

Then it extracts structured fields from evidence using `_extract_structured_fields(...)`.

If it finds a strong match, it can directly build a clean answer such as:

`Butler Court is listed at GBP 126.68 per week, and the total contract cost is GBP 5,302.27.`

Why this exists:

- price fields are highly structured
- they are easy to extract deterministically
- this avoids the model missing obvious numbers

### 11. Generic Fallback Summarization

If it is not a price question, the fallback still tries to help.

It:

1. splits evidence text into lines
2. scores lines by relevance to the query
3. keeps the best lines
4. formats them into a grounded answer

Relevant helpers:

- `_tokenize(...)`
- `_format_line_for_user(...)`
- `_select_salient_lines(...)`

This is a simple rule-based answering strategy.

It is not as fluent as the LLM, but it is reliable and grounded.

## The “Trust But Verify” Safety Check

One of the most important methods is `_should_prefer_fallback(...)`.

Even when the LLM returns an answer, the code still compares it with the fallback result.

If the LLM does something suspicious, the fallback answer is preferred instead.

Examples of suspicious behavior:

- the LLM says there is not enough evidence
- confidence is extremely low
- no citations are returned
- the question is clearly about price, but the LLM ignored the price line

This is a very practical guardrail.

It means:

- the LLM is given the first chance
- but the system does not blindly trust it

## Key Helper Methods

Here is what the main helper methods do:

- `_normalize_evidence(...)`
  Converts input items into validated `EvidenceItem` objects.

- `_truncate_text_preserve_lines(...)`
  Shortens evidence while keeping line structure, which matters for structured data.

- `_select_salient_lines(...)`
  Picks the most useful lines from a chunk to show the LLM.

- `_looks_like_price_query(...)`
  Detects when the question is probably about money.

- `_extract_structured_fields(...)`
  Pulls out values like hall name, room name, weekly price, total price, and year.

- `_format_money(...)`
  Formats money consistently.

- `_looks_like_missing_evidence_answer(...)`
  Detects “I don’t know” style LLM answers that may be too pessimistic.

## Why This Design Is Good

This design is good because it balances:

- fluency
- grounding
- robustness

The LLM gives natural answers.
The fallback logic gives reliability.
The citations make the result explainable.

So the answerer is not just:

`prompt -> model -> answer`

It is more like:

`evidence -> structured prompt -> LLM answer -> validation -> fallback check -> final grounded answer`

## Limitations To Be Aware Of

This answerer is still not perfect.

Some current limitations:

- it only knows what the retriever gives it
- if retrieval misses the right chunk, the answerer cannot fix that
- some fallback logic is specialized for price-style questions
- generic comparison or synthesis questions may still rely heavily on LLM quality
- confidence is heuristic, not a calibrated probability

## Simple Way To Explain It In A Team Meeting

You can describe it like this:

“After retrieval finishes, the AnswererAgent is the last step in the RAG pipeline. It takes the user’s question and the retrieved evidence chunks, builds a structured prompt for the LLM, validates the LLM’s JSON output, attaches citations, and returns a grounded answer. If the LLM is unavailable or gives a weak answer, the agent falls back to a rule-based grounded answer so we avoid hallucinating.”

## One-Sentence Summary

The `AnswererAgent` is the part of the system that turns retrieved knowledge-base chunks into a final user-facing answer, while trying to stay grounded, cite evidence, and recover safely when the LLM is weak or unavailable.

