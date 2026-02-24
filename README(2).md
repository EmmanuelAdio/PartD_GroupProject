# ProcessorAgent_LLM (`agents/processor_LLM.py`) — How it Works

## Purpose
`ProcessorAgent_LLM` converts a **raw user question** into a **structured payload** that downstream components (especially the Answerer) can use reliably.

It solves the problem: *“Users speak naturally, but the Answerer needs a predictable structure.”*

**Inputs:** free-text user query  
**Outputs:** a dict containing:
- `intent` (one of `ALLOWED_INTENTS`)
- `domain` (routing category for the Answerer)
- `slots` (extracted entities like course title / hall name / UCAS)
- `requested_fields` (what the user is asking for: modules, fees, etc.)
- `retrieval_query` (debuggable string for future retrieval)
- `_debug` (how decisions were made)

---

## High-level Flow (Architecture)

```mermaid
flowchart TD
    A[User Question] --> B[ProcessorAgent_LLM.process()]
    B --> C[Normalize text\n(lowercase + unicode NFKC)]
    C --> D{OpenAI available?\nOPENAI_API_KEY set}
    D -- Yes --> E[LLM Extraction\n(intent + entities + requested_fields)]
    D -- No --> F[Fallback Intent Classifier\n(utils/llm_intent)]
    E --> G{Valid intent\nin ALLOWED_INTENTS?}
    G -- Yes --> H[Map intent -> domain]
    G -- No --> F
    F --> H[Map intent -> domain]

    H --> I[Build slots from entities\n(+ generic entity list)]
    I --> J{Entity resolution enabled?\nENABLE_ENTITY_RESOLUTION=1\n& Mongo connected}
    J -- Yes --> K[Resolve entity to exact DB doc\n(accommodation / undergraduate_courses)]
    J -- No --> L[Skip resolution]

    K --> M[Build retrieval_query\n(debug string)]
    L --> M

    M --> N[Return structured output\n+ _debug info]
    N --> O[Answerer uses domain + slots\nand resolved_id if present]