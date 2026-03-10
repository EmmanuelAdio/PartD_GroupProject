$ErrorActionPreference = "Stop"

param(
    [Parameter(Mandatory=$false)]
    [string]$MongoDb = "open_day_knowledge",
    [Parameter(Mandatory=$false)]
    [string]$MongoCollection = "kb_chuncks",
    [switch]$UseOpenAI
)

Write-Host ""
Write-Host "=== SAFETY CHECK TESTS (READ-ONLY FOR MONGODB DATA) ==="
Write-Host "This script does NOT run ingestion upload and does NOT delete/update MongoDB chunk records."
Write-Host ""

Write-Host "1) Python compile checks"
python -m compileall services schemas app test_ingestion_service.py test_retrieval_service.py test_index_manager.py

Write-Host ""
Write-Host "2) Local ingestion pipeline dry-run (NO MongoDB writes)"
Write-Host "   - Builds chunks/embeddings in memory only"
python test_ingestion_service.py --embedder fake --tagger heuristic

Write-Host ""
Write-Host "3) Atlas index health check (read-only, NO index creation)"
python test_index_manager.py --embedder fake --mongo-db $MongoDb --mongo-collection $MongoCollection

Write-Host ""
Write-Host "4) Retrieval smoke test against existing MongoDB knowledge base (read-only)"
python test_retrieval_service.py --embedder fake --mongo-db $MongoDb --mongo-collection $MongoCollection

Write-Host ""
Write-Host "5) Retrieval focused query (read-only)"
python test_retrieval_service.py --embedder fake --mongo-db $MongoDb --mongo-collection $MongoCollection --query "UCAS entry requirements" --top-k 6

if ($UseOpenAI) {
    Write-Host ""
    Write-Host "6) OpenAI retrieval check (read-only, no re-ingestion)"
    python test_retrieval_service.py --embedder openai --embedding-model text-embedding-3-small --mongo-db $MongoDb --mongo-collection $MongoCollection --query "UCAS entry requirements" --top-k 6
}

Write-Host ""
Write-Host "=== SAFETY CHECK TESTS COMPLETED ==="
