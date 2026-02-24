import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne


def main():
    # 1) Load .env from project root (works even if you run from /scripts)
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    load_dotenv(env_path)

    mongo_uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "open_day_knowledge")

    if not mongo_uri:
        raise RuntimeError("MONGODB_URI not found. Put it in your .env file.")

    # 2) Connect
    client = MongoClient(mongo_uri)
    db = client[db_name]

    # 3) Load JSON file
    json_path = project_root / "all_ug_courses.json"   # <-- put file in project root
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found at: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Your file is expected to be a list of dicts
    if not isinstance(data, list):
        raise ValueError("Expected all_ug_courses.json to contain a JSON list of course objects.")

    # 4) Choose collection
    col = db["undergraduate_courses"]

    # 5) Upsert strategy: avoid duplicates
    # We'll use source_url if present, else course_title as fallback.
    ops = []
    inserted_like = 0
    skipped = 0

    for course in data:
        if not isinstance(course, dict):
            skipped += 1
            continue

        key = course.get("source_url") or course.get("course_title")
        if not key:
            skipped += 1
            continue

        # Store a stable "doc_id" field for debugging / indexing
        course["doc_id"] = key

        ops.append(
            UpdateOne(
                {"doc_id": key},
                {"$set": course},
                upsert=True
            )
        )

    # 6) Bulk write
    if ops:
        result = col.bulk_write(ops, ordered=False)
        print("Upload complete ✅")
        print("DB:", db_name)
        print("Collection:", col.name)
        print("Matched:", result.matched_count)
        print("Modified:", result.modified_count)
        print("Upserted:", len(result.upserted_ids))
        print("Skipped records:", skipped)
        print("Total docs now:", col.count_documents({}))
    else:
        print("No valid course records found to upload.")


if __name__ == "__main__":
    main()