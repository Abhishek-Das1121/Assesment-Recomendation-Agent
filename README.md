# SHL Assessment Recommender

Conversational SHL assessment recommendation engine.
Architecture: retrieval-backed, stateless, grounded, evaluator-safe.

## Stack
- FastAPI (API)
- sentence-transformers / all-MiniLM-L6-v2 (embeddings, CPU-only)
- FAISS (vector search)
- Gemini Flash (structured extraction + response generation)
- Pydantic (schema validation)

## Setup (run in order)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set environment variables
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```
Get a free Gemini API key at: https://aistudio.google.com/

### 3. Scrape the SHL catalog
```bash
python scrape_catalog.py
# Output: catalog.json (~200 entries)
```

### 4. Build the FAISS index
```bash
python build_index.py
# Output: catalog.faiss + metadata.json
# Takes ~2 minutes on CPU (downloads model on first run)
```

### 5. Start the server
```bash
python main.py
# Starts on http://localhost:8000
```

### 6. Test health
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### 7. Test chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hiring a senior Java developer"}]}'
```

### 8. Run evaluation traces
```bash
python test_traces.py
# Runs all sample traces and reports Recall@10
```

## File structure
```
main.py              — FastAPI app, lifespan, endpoints
models.py            — Pydantic schemas (non-negotiable)
agent.py             — Pipeline orchestrator
catalog_engine.py    — Catalog loading + lookup maps
retriever.py         — FAISS semantic search + exact lookup
ranker.py            — Hybrid ranking + filtering + diversification
state_extractor.py   — Gemini call #1: structured JSON extraction
mode_detector.py     — Pure Python mode detection (no LLM)
response_generator.py— Gemini call #2: grounded reply generation
prompts.py           — All prompt templates
validators.py        — Hallucination firewall + schema validation
scrape_catalog.py    — One-time catalog scraper
build_index.py       — One-time FAISS index builder
test_traces.py       — Evaluator trace replay
```

## Architecture principles
- **Retrieval decides. LLM explains. Python controls.**
- Two Gemini calls per request: (1) state extraction temp=0, (2) response generation temp=0.2
- FAISS retrieval on synthesized query (NOT raw conversation text)
- Exact catalog lookup for compare mode (NO semantic retrieval)
- Stateless: full history reconstructed from messages[] every turn
- Hallucination firewall: every URL validated against catalog before response

## Deployment (Render / Railway / Fly.io)
Set environment variables:
- `GEMINI_API_KEY`

The server preloads catalog + FAISS index at startup (~30-60 seconds).
/health returns 200 once ready. Evaluator allows 2 minutes for cold start.
