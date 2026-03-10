# System Diagram

This project is building a **Loughborough Open Day assistant**: a RAG backend over university data, with a 3D avatar chat UI.

```mermaid
flowchart LR
    U[User]

    subgraph FE["Avatar Frontend (Vite + Three.js)"]
        AV[3D Avatar Scene<br/>GLTF + OrbitControls]
        UI[Chat + Mic UI<br/>SpeechRecognition]
        DUMMY[Current reply path<br/>Local dummy response]
    end

    subgraph API[FastAPI App]
        MAIN[app/main.py<br/>/query /ingest/* /status]
        QO[QueryOrchestrator]
        IO[IngestionOrchestrator]
    end

    subgraph QP[Query Pipeline]
        PA[ProcessorAgent<br/>LLM-first RetrievalQuery planner]
        RS[RetrieverService<br/>Hybrid retrieval + rerank]
    end

    subgraph ING[Ingestion Pipeline]
        IS[IngestionService<br/>normalize -> chunk -> tag -> embed]
        ES[EmbeddingService<br/>OpenAI or deterministic fake]
        LS[LLMService<br/>tagging + planner JSON]
    end

    subgraph DB[MongoDB / Atlas]
        CH[(kb_chuncks)]
        MF[(kb_ingestion_manifest)]
        IDX[Atlas Vector + Search indexes<br/>managed by AtlasIndexManager]
    end

    subgraph DATA[Knowledge Sources]
        JSON[data/*.json<br/>accommodation, courses, admissions, sports]
    end

    OAI[(OpenAI API)]

    U --> UI
    UI --> DUMMY
    UI -->|planned API integration| MAIN
    AV -->|depends on| UI

    MAIN --> QO
    MAIN --> IO

    QO --> PA
    QO --> RS
    PA --> LS
    RS --> CH
    RS --> IDX
    RS --> ES
    ES --> OAI

    IO --> IS
    JSON --> IS
    IS --> ES
    IS --> LS
    IS --> CH
    IO --> MF
    IO --> IDX
    MF --> CH
```

## Current state reflected by code
- Backend RAG pipeline is implemented end-to-end (`app/`, `services/`, `agents/`).
- Query path: `ProcessorAgent` plans structured retrieval, then `RetrieverService` executes hybrid search with fallbacks.
- Ingestion path: JSON sources are chunked, tagged, embedded, and upserted into Mongo with source manifests for incremental runs.
- Avatar frontend currently uses local dummy responses and is **not yet connected** to `POST /query`.

## Runtime Query Sequence

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant AvatarUI as Avatar UI (main.js)
    participant API as FastAPI /query
    participant QO as QueryOrchestrator
    participant PA as ProcessorAgent
    participant LLM as LLMService (OpenAI)
    participant RS as RetrieverService
    participant EMB as EmbeddingService
    participant Mongo as MongoDB/Atlas

    User->>AvatarUI: Ask question (typed or voice)
    AvatarUI->>API: POST /query { query, top_k? }
    API->>QO: run(user_query, top_k_override?)

    QO->>PA: process(user_query)
    alt OpenAI key available
        PA->>LLM: generate_json(planner prompt)
        LLM-->>PA: RetrievalQuery JSON
    else No OpenAI key
        PA-->>PA: Build default RetrievalQuery
    end
    PA-->>QO: RetrievalQuery plan

    QO->>RS: retrieve(plan)
    RS->>EMB: embed_query(query_text)
    EMB-->>RS: query vector

    par Vector channel
        RS->>Mongo: $vectorSearch (with metadata filters)
        alt Vector index unavailable/error
            RS->>Mongo: find() + Python cosine fallback
        end
    and Lexical channel
        RS->>Mongo: Atlas $search (with metadata filters)
        alt Atlas search unavailable/error
            RS->>Mongo: Mongo $text
            alt $text unavailable
                RS->>Mongo: regex/token scan fallback
            end
        end
    end

    RS-->>QO: merged+rereanked EvidenceItem[]
    alt 0 hits
        QO->>RS: retry without domain filter
        alt still 0 hits
            QO->>RS: retry bare query (no filters)
        end
    end

    QO-->>API: { processor_plan, attempts_log, evidence, diagnostics }
    API-->>AvatarUI: JSON response
    AvatarUI-->>User: Render answer/evidence (integration step)
```
