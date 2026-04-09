from __future__ import annotations

from services.embedding_service import DeterministicEmbeddingService
from services.ingestion_service import IngestionService


def _build_service(*, json_group_size: int = 30) -> IngestionService:
    return IngestionService(
        repo=None,
        embedder=DeterministicEmbeddingService(dim=8),
        tagger=None,
        version="test-v1",
        json_group_size=json_group_size,
    )


def test_ingest_json_top_level_array_chunks_one_entity_per_item() -> None:
    service = _build_service()
    data = [
        {
            "name": "Butler Court",
            "room_types": [
                {
                    "name": "Standard",
                    "prices": [{"per_week_gbp": 126.68, "total_contract_gbp": 5302.27}],
                }
            ],
        },
        {
            "name": "The Holt",
            "room_types": [
                {
                    "name": "Ensuite",
                    "prices": [{"per_week_gbp": 165.41, "total_contract_gbp": 6900.14}],
                }
            ],
        },
    ]

    count, records = service.ingest_json(data=data, source_id="accommodation_halls")

    assert count == 2
    assert [r.order for r in records] == [0, 1]
    assert all(r.section == "json_fields" for r in records)

    assert "[0].name: Butler Court" in records[0].text
    assert "[0].room_types[0].prices[0].per_week_gbp: 126.68" in records[0].text
    assert "[1].name: The Holt" not in records[0].text

    assert "[1].name: The Holt" in records[1].text
    assert "[1].room_types[0].prices[0].per_week_gbp: 165.41" in records[1].text
    assert "[0].name: Butler Court" not in records[1].text


def test_ingest_json_non_array_keeps_line_group_chunking() -> None:
    service = _build_service(json_group_size=2)
    data = {
        "halls": [
            {"name": "A", "per_week_gbp": 100},
            {"name": "B", "per_week_gbp": 110},
            {"name": "C", "per_week_gbp": 120},
        ]
    }

    count, records = service.ingest_json(data=data, source_id="accommodation_halls")

    # 6 flattened lines grouped by 2 => 3 chunks.
    assert count == 3
    assert [r.order for r in records] == [0, 1, 2]
    assert all(r.section == "json_fields" for r in records)
    assert "halls[0].name: A" in records[0].text


def test_ingest_json_array_skips_empty_items_and_preserves_index_order() -> None:
    service = _build_service()
    data = [
        {"name": "Alpha"},
        {},
        {"name": "Gamma", "room_types": []},
        {"name": "   "},
        "",
        {"name": "Omega"},
    ]

    count, records = service.ingest_json(data=data, source_id="accommodation_halls")

    assert count == 3
    assert [r.order for r in records] == [0, 2, 5]
    assert all(r.section == "json_fields" for r in records)
    assert "[0].name: Alpha" in records[0].text
    assert "[2].name: Gamma" in records[1].text
    assert "[5].name: Omega" in records[2].text
