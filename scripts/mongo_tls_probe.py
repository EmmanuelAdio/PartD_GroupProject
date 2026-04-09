from __future__ import annotations

import os
import ssl
import sys
from pathlib import Path

import certifi
import pymongo
from pymongo import MongoClient
from pymongo.errors import PyMongoError

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def _load_project_env() -> None:
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if not env_path.exists():
        return
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path)
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _redact_uri(uri: str) -> str:
    if "://" not in uri or "@" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    if "@" not in rest:
        return uri
    _, tail = rest.split("@", 1)
    return f"{scheme}://<redacted>@{tail}"


def main() -> int:
    _load_project_env()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        print("MONGODB_URI is not set.")
        return 2

    print(f"python_executable: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"openssl: {ssl.OPENSSL_VERSION}")
    print(f"pymongo: {pymongo.version}")
    print(f"certifi: {certifi.where()}")
    print(f"uri: {_redact_uri(uri)}")

    try:
        client = MongoClient(
            uri,
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=10_000,
            connectTimeoutMS=10_000,
            socketTimeoutMS=10_000,
        )
        result = client.admin.command("ping")
        print(f"ping: {result}")
        print("Mongo TLS probe passed.")
        return 0
    except PyMongoError as exc:
        print("Mongo TLS probe failed.")
        print(exc)
        print(
            "Next checks: Atlas IP allowlist, DB user permissions, VPN/proxy SSL interception, "
            "and trying from a different network."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
