# Demo Checklist — Loughborough University Virtual Assistant

## Environment Variables Required

Both variables must be present in `.env` at the project root before starting the backend.

| Variable | Required | Description |
|---|---|---|
| `MONGODB_URI` | Yes | MongoDB Atlas connection string |
| `OPENAI_API_KEY` | Yes | OpenAI API key (gpt-4o-mini + text-embedding-3-small) |

The frontend reads `VITE_API_BASE_URL` from `cs-avatar/.env` (defaults to `http://127.0.0.1:8000`).

---

## Setup Steps

### 1. Backend

```bash
# From the project root
cd c:/Users/Emman/OneDrive/Documents/GitHub/PartD_GroupProject

# Activate virtual environment
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies (first time only)
pip install -r requirements.txt

# Start the API server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Expected startup output:
```
[startup] embedder=openai | openai_key=set | mongodb_uri=set
[startup] Orchestrators ready.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 2. Check data is ingested

Open http://127.0.0.1:8000/health in a browser and confirm:
- `"status": "ok"`
- `"mongo": { "total_chunk_docs": <number > 0> }`
- `"index_health": { "healthy": true }`

If `total_chunk_docs` is 0 or missing, trigger ingestion once:
```bash
curl -X POST http://127.0.0.1:8000/ingest/all
```

### 3. Frontend

```bash
cd cs-avatar
npm install        # first time only
npm run dev        # Vite dev server at http://localhost:5173
```

Open http://localhost:5173 in Chrome (best voice support).

### 4. Debug / Pipeline Inspector (optional)

Open http://localhost:5173**?debug=1** to activate the Pipeline Inspector panel (top-right corner). This shows:
- Processor Agent output (domain, entity tags, rewritten query)
- Retrieval stats (attempt used, chunk count, vector/text mode)
- Evaluator verdict (pass/fallback/ask_clarification) and grounding flags
- Per-stage latency in milliseconds

The backend terminal also logs a one-line summary for every query.

---

## Demo Questions and Expected Behaviour

### Accommodation

| Question | Expected behaviour |
|---|---|
| "What is the cheapest accommodation?" | Deterministic price-extreme answer listing the cheapest hall + weekly price |
| "What is the most expensive accommodation?" | Deterministic price-extreme answer listing the most expensive option |
| "How much does it cost to stay in Elvyn Hall?" | Specific hall price with per-week and total contract amounts |
| "Can you tell me about student accommodation?" | Overview of halls, facilities, contract lengths |
| "What accommodation options are available on campus?" | List of halls with descriptions |

### Courses

| Question | Expected behaviour |
|---|---|
| "What Computer Science courses do you offer?" | List of CS undergraduate courses |
| "What are the entry requirements for Engineering?" | A-level/UCAS point requirements for engineering degrees |
| "What modules are covered in the Business degree?" | Module list from business course data |
| "Do you offer foundation year programmes?" | Foundation course information |

### Facilities and Campus Life

| Question | Expected behaviour |
|---|---|
| "What sports facilities are available?" | Sports centre, pitches, swimming pool etc. |
| "What is the campus like?" | General overview of Loughborough campus |
| "Is parking available on campus?" | Parking / transport answer |

### Finance and Admissions

| Question | Expected behaviour |
|---|---|
| "How much are tuition fees?" | Home/international fee information |
| "What scholarships or bursaries are available?" | Finance/funding answer |
| "How do I apply as an international student?" | Admissions/visa process answer |
| "What support is available for disabled students?" | Disability support services |

### Clarification and Fallback Cases

| Question | Expected behaviour |
|---|---|
| "What does it cost?" | Evaluator triggers ask_clarification: "Could you clarify what you mean — tuition fees, accommodation, or something else?" |
| "Tell me about it." | Ambiguous pronoun → clarification question |
| "What is the weather today?" | No evidence found → safe fallback: "I couldn't verify a reliable answer…" with help URL |

### Canned Responses (instant, no backend call)

| Question | Expected behaviour |
|---|---|
| "Hello" / "Hi" | Greeting response, no API call |
| "Thank you" | Acknowledgement response |
| "Repeat that" | Repeats `lastAnswer` or "I don't have a previous answer" |
| "Bye" | Goodbye message |
| "What can you help with?" | Capability overview |

### Follow-up Context

| Sequence | Expected behaviour |
|---|---|
| Ask about CS → "Tell me more" | Appends "(in relation to: [previous question])" to the query |
| Ask about fees → "What about scholarships?" | Builds on the previous topic |

---

## Known Limitations

- **Voice (TTS/STT):** Uses the browser's Web Speech API. Works best in Chrome. Firefox and Safari have partial or no support. If speech is unavailable, the mic button is automatically disabled and text input still works.
- **Speech quality:** Voice selection prefers "Google UK English Male". If this voice is not installed, the nearest en-GB local voice is used, which may sound different.
- **Atlas Vector Search:** Requires the `kb_vector_index` Atlas index to be in `READY` state. If not ready, the retriever falls back to Python cosine similarity (slower). Check `/health` for index status.
- **Atlas Text Search:** Similarly requires `kb_text_index` to be `READY`. Falls back to MongoDB `$text` index, then regex scan.
- **Accommodation pricing:** The price data reflects the academic year stored in the JSON source files. Always verify against the official Loughborough website for current figures.
- **OpenAI rate limits:** If the OpenAI key is rate-limited, the Processor and Answerer agents fall back to deterministic logic. Answers may be less fluent but will still be grounded.
- **MongoDB Atlas IP allowlist:** The Atlas cluster must have the demo machine's IP address in its allowlist. If the backend fails to start, check Network Access in Atlas.
- **CORS:** The backend allows `http://localhost:5173` and `http://127.0.0.1:5173` by default. If you serve the frontend on a different port, add it to `FRONTEND_ORIGINS` in `.env`.

---

## Fallback / Graceful Degradation Summary

| Failure | System behaviour |
|---|---|
| MongoDB unavailable | `/health` returns `"status": "degraded"`. Query endpoint catches the exception and returns a clear 500 error rather than crashing silently. |
| OpenAI API unavailable | Processor falls back to bare RetrievalQuery. Answerer uses deterministic evidence extraction. Evaluator uses rule-based checks only. |
| No evidence retrieved | Answerer returns safe fallback: "I couldn't verify a reliable answer…" with the official Loughborough URL. |
| Ambiguous query | Evaluator detects ambiguity → orchestrator returns a clarification question. |
| Avatar GLB fails to load | Chat UI remains fully functional. TTS/STT still works without the 3D model. |
| Speech recognition unavailable | Mic button is disabled with a tooltip. Text input remains active. |

---

## Quick Smoke Test (run before the demo)

```bash
# 1. Backend health
curl http://127.0.0.1:8000/health

# 2. Simple query
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the cheapest accommodation?"}'

# 3. Debug query (full pipeline response)
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What Computer Science courses do you offer?", "debug": true}'

# 4. Status (index health)
curl http://127.0.0.1:8000/status
```
