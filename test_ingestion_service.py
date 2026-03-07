import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, List

from services.embedding_service import EmbeddingService
from services.ingestion_service import IngestionService


class FakeEmbedder:
    """Deterministic local embedder for testing without OpenAI."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def _embed_text(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = list(digest[: self.dim])
        # Map bytes [0,255] -> [-1,1] for embedding-like shape.
        return [((v / 255.0) * 2.0) - 1.0 for v in values]


def discover_json_files(data_dir: Path) -> Iterable[Path]:
    for path in sorted(data_dir.glob("*.json")):
        if path.name.endswith("_chunk_records_preview.json"):
            continue
        yield path


def ingest_file(path: Path, service: IngestionService, output_dir: Path | None) -> None:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    source_id = path.stem
    count, records = service.ingest_json(data=data, source_id=source_id, title=source_id.replace("_", " ").title())

    print(f"\n[{path.name}]")
    print(f"  records: {count}")
    if records:
        print(f"  first chunk id: {records[0].chunk_id}")
        print(f"  first chunk preview: {records[0].text[:120].replace(chr(10), ' ')}")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{path.stem}_chunk_records_preview.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in records], f, indent=2, ensure_ascii=False)
        print(f"  wrote preview: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ingestion against JSON files in data/ using real files.")
    parser.add_argument("--data-dir", default="data", help="Directory containing input JSON files.")
    parser.add_argument("--file", default=None, help="Single JSON file name inside data-dir.")
    parser.add_argument(
        "--embedder",
        choices=["fake", "openai"],
        default="fake",
        help="Embedding backend. 'fake' requires no API key. 'openai' requires OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--write-previews",
        action="store_true",
        help="Write generated chunk records as JSON files under data/chunk_previews/.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    embedder = FakeEmbedder() if args.embedder == "fake" else EmbeddingService()
    service = IngestionService(repo=None, embedder=embedder, version="test-v1")

    output_dir = data_dir / "chunk_previews" if args.write_previews else None

    if args.file:
        path = data_dir / args.file
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")
        ingest_file(path, service, output_dir)
        return

    files = list(discover_json_files(data_dir))
    if not files:
        print(f"No JSON files found in {data_dir}.")
        return

    for path in files:
        ingest_file(path, service, output_dir)


if __name__ == "__main__":
    main()
