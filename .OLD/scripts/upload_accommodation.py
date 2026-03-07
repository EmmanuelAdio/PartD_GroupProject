import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne


def main():
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    mongo_uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "open_day_knowledge")
    if not mongo_uri:
        raise RuntimeError("MONGODB_URI not found in .env")

    client = MongoClient(mongo_uri)
    db = client[db_name]
    col = db["accommodation"]

    json_path = project_root / "accommodation_halls.json"  # <-- rename to your file
    if not json_path.exists():
        raise FileNotFoundError(f"Accommodation JSON not found at: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected accommodation_halls.json to contain a JSON list of hall objects.")

    ops = []
    skipped = 0

    for hall in data:
        if not isinstance(hall, dict):
            skipped += 1
            continue

        key = hall.get("official_url") or hall.get("name")
        if not key:
            skipped += 1
            continue

        hall["doc_id"] = key

        ops.append(
            UpdateOne(
                {"doc_id": key},
                {"$set": hall},
                upsert=True
            )
        )

    if ops:
        res = col.bulk_write(ops, ordered=False)
        print("✅ Accommodation upload complete")
        print("DB:", db_name)
        print("Collection:", col.name)
        print("Matched:", res.matched_count)
        print("Modified:", res.modified_count)
        print("Upserted:", len(res.upserted_ids))
        print("Skipped:", skipped)
        print("Total docs now:", col.count_documents({}))
    else:
        print("No valid hall records found.")


if __name__ == "__main__":
    main()