# Starts the Indian-Acts instance of the API on http://127.0.0.1:8001 (the general app stays on :8000).
#
#   cd "D:\Adaptive rag - skill lab\backend"
#   .\scripts\start_legal.ps1
#
# Everything instance-specific is set here as process environment variables, so the shared .env is never touched.
# API keys are still read from the shared .env (GEMINI_API_KEY is used when present; otherwise OPENAI_API_KEY).
$ErrorActionPreference = "Stop"
$backend = Split-Path -Parent $PSScriptRoot          # scripts\ -> backend\
Set-Location -LiteralPath $backend                    # uvicorn must run from backend\ so that "app" is importable

$dataDir = Join-Path $backend "data_legal"
$env:DATA_DIR = $dataDir
$env:DATABASE_URL = "sqlite+aiosqlite:///" + ((Join-Path $dataDir "app.db") -replace "\\", "/")
$env:DEFAULT_CORPUS = "indian_acts"
$env:INSTANCE_MODE = "legal_acts"
$env:LEGAL_BUDGET_MODE = "true"                       # fixed top-4, short answers, local fast NLI, no retries
$env:LLM_PROVIDER = "auto"                            # Gemini when GEMINI_API_KEY is set
$env:GEMINI_MODEL = "gemini-2.5-flash"               # free tier allows ~20 requests/day PER MODEL; change if exhausted
$env:EMBEDDING_PROVIDER = "local"                     # free local embeddings
$env:SEED_DEMO_CORPUS = "false"
$env:DEMO_CACHE_ENABLED = "true"
$env:DEMO_CACHE_DIR = Join-Path $backend "seed\legal_demo_cache"
$env:DEMO_QUESTIONS_PATH = Join-Path $backend "seed\legal_demo_questions.json"
$env:CORS_ORIGINS = '["http://localhost:5174"]'

& (Join-Path $backend ".venv\Scripts\python.exe") -m uvicorn app.main:app --host 127.0.0.1 --port 8001
