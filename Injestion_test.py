import sys
import os
import json
import hashlib
from pprint import pprint
from typing import Any, Dict, List, Optional, Union

from services.ingestion_service import IngestionService

from services.ingestion_service import IngestionService

# --- Make project root importable ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd()))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Import schemas directly (absolute import from project root) ---
from schemas.models import Document, Chunk, ChunkTags, ChunkRecord

# --- OpenAI ---
import openai
from dotenv import load_dotenv


load_dotenv()  # loads OPENAI_API_KEY from a .env file if present

# If you don't have a .env file, set your key directly here:
# os.environ["OPENAI_API_KEY"] = "sk-..."

print("✓ Imports OK")
print(f"  Project root : {PROJECT_ROOT}")
print(f"  OpenAI key   : {'set' if os.getenv('OPENAI_API_KEY') else 'NOT SET — add to .env or set above'}")


JSON_PATH = os.path.join(PROJECT_ROOT, "data", "accommodation_halls.json")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    accommodation_data = json.load(f)

print(f"✓ Loaded {len(accommodation_data)} halls from {JSON_PATH}\n")
print("=" * 60)
print("First hall entry (full structure):")
print("=" * 60)
pprint(accommodation_data[0], depth=3)



class OpenAIEmbedder:
    """Temporary embedder for local ingestion testing.

    Uses OpenAI's embeddings API in the same spirit as your earlier MVP.
    Requires OPENAI_API_KEY to be set in the environment.
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        if openai.OpenAI is None:
            raise ImportError(
                "The 'openai' package is not installed. Install it with: pip install openai"
            )
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
    

    

def run_json_ingestion_tests(path) -> None:
    """Runs step-by-step tests for the JSON ingestion pipeline.

    Stops before MongoDB upsert and prints the final ChunkRecord payloads.
    """

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    service = IngestionService(embedder=OpenAIEmbedder(), version="test-v1", json_group_size=25)

    print("=" * 80)
    print("TEST 1: LOAD JSON")
    print(f"Loaded file: {path}")
    print(f"Top-level type: {type(data).__name__}")
    print(f"Top-level length: {len(data) if isinstance(data, list) else 'N/A'}")

    print("\n" + "=" * 80)
    print("TEST 2: NORMALIZE JSON -> DOCUMENT")
    doc = service._normalize_json(data=data, source_id="accommodation_halls", title="Accommodation Halls")
    print(f"Document source_id: {doc.source_id}")
    print(f"Document source_type: {doc.source_type}")
    print(f"Document title: {doc.title}")
    print(f"Normalized text preview:\n{doc.text[:1200]}\n")

    print("=" * 80)
    print("TEST 3: SEGMENT DOCUMENT -> CHUNKS")
    chunks = service._segment(doc)
    print(f"Number of chunks created: {len(chunks)}")
    for chunk in chunks[:3]:
        print(f"\nChunk order={chunk.order} id={chunk.chunk_id} section={chunk.section} raw_path={chunk.raw_path}")
        print(chunk.text[:600])
        print("-" * 60)

    print("\n" + "=" * 80)
    print("TEST 4: TAG FIRST 3 CHUNKS")
    for chunk in chunks[:3]:
        tags = service._tag_chunk(doc, chunk)
        print(f"\nChunk {chunk.order} tags:")
        print(tags.model_dump_json(indent=2))

    print("\n" + "=" * 80)
    print("TEST 5: EMBED FIRST 3 CHUNKS")
    sample_texts = [chunk.text for chunk in chunks[:3]]
    sample_embeddings = service.embedder.embed_many(sample_texts)
    for i, emb in enumerate(sample_embeddings):
        print(f"Chunk {i} embedding dimension: {len(emb)}")
        print(f"Embedding preview: {emb[:8]}")

    print("\n" + "=" * 80)
    print("TEST 6: BUILD FINAL RECORDS (PRE-UPSERT OUTPUT)")
    final_records = service.ingest_json(data=data, source_id="accommodation_halls", title="Accommodation Halls")
    print(f"Total final records built: {len(final_records)}")

    preview_count = min(3, len(final_records))
    for i in range(preview_count):
        print(f"\nFinal record {i + 1}/{preview_count}:")
        print(final_records[i].model_dump_json(indent=2)[:4000])

    output_path = path.parent / "accommodation_chunk_records_preview.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in final_records], f, indent=2)

    print("\n" + "=" * 80)
    print("DONE")
    print(f"Saved full pre-upsert record preview to: {output_path}")
    print("These records are the exact shape you would upsert into MongoDB next.")


run_json_ingestion_tests(JSON_PATH)