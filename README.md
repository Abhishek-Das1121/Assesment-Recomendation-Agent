# Assessment Recommendation Agent 🚀
### *A Hybrid RAG-based Consultant for Recruitment Strategy*

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Llama3.3](https://img.shields.io/badge/LLM-Llama--3.3--70B-blue?style=for-the-badge)](https://groq.com/)

## 📌 Project Overview

This agent acts as an expert consultant for the SHL assessment catalog. Instead of manual filtering, users describe hiring needs in natural language, and the agent utilizes a **Retrieval-Augmented Generation (RAG)** pipeline to recommend the most relevant tests.

### Core Principles
- **Retrieval Decides. LLM Explains. Python Controls.**
- **Dual-Stage Pipeline:** FAISS semantic search followed by an LLM reasoning layer to eliminate hallucinations.
- **Production Ready:** Fully containerized with Docker and optimized for sub-second inference.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI (Asynchronous) |
| Embeddings | `all-MiniLM-L6-v2` (Sentence-Transformers) |
| Vector DB | FAISS (Facebook AI Similarity Search) |
| Reasoning Engine | Llama-3.3-70B via Groq Cloud |
| Validation | Pydantic (Strict schema enforcement) |

---

## 🏗️ Architecture Flow
[ User Query ]

[ FastAPI Entry Point ]

├─► Step 1 — State Extraction
Groq/Llama extracts role, seniority, skills (temp=0)

├─► Step 2 — FAISS Retrieval
all-MiniLM-L6-v2 finds top-25 candidates from catalog

├─► Step 3 — Hybrid Ranking
Python scores by skill overlap, type alignment, anchors

├─► Step 4 — Grounded Response
Llama generates reply strictly from shortlisted catalog data

└─► Step 5 — Hallucination Firewall
Every URL validated against catalog before returning

---

## 🚀 Setup & Deployment

### 1. Environment Configuration

Create a `.env` file in the root directory:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at: https://console.groq.com

### 2. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data
python scrape_catalog.py   # Builds catalog.json
python build_index.py      # Generates catalog.faiss + metadata.json

# Start server
python main.py             # Runs on http://localhost:7860
```

### 3. Docker Deployment

```bash
docker build -t shl-agent .
docker run -p 7860:7860 -e GROQ_API_KEY=$GROQ_API_KEY shl-agent
```

### 4. API Usage

**Health check:**
```bash
curl http://localhost:7860/health
# {"status": "ok"}
```

**Chat:**
```bash
curl -X POST http://localhost:7860/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hiring a senior Java developer"}]}'
```

**Response schema:**
```json
{
  "reply": "string",
  "recommendations": [
    {"name": "...", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

---

## 📁 File Structure
main.py               — FastAPI app & lifecycle management

agent.py              — RAG pipeline orchestrator

catalog_engine.py     — Catalog loading + exact lookup maps

retriever.py          — FAISS semantic search + synthesized query builder

ranker.py             — Hybrid scoring, anchor injection, diversification

state_extractor.py    — LLM call #1: structured JSON state extraction (temp=0)

mode_detector.py      — Pure Python mode detection (no LLM)

response_generator.py — LLM call #2: grounded reply generation

prompts.py            — All prompt templates

validators.py         — Hallucination firewall + schema validation

scrape_catalog.py     — Catalog builder with accurate assessment metadata

build_index.py        — One-time FAISS index builder

test_traces.py        — Evaluator trace replay + Recall@10 reporting

---

## 📈 Performance & Honesty

| Metric | Value |
|--------|-------|
| Recall@10 | ~55% |
| Precision | ~80% |
| Avg Latency | ~0.8s/request |
| Schema Compliance | 100% |

**Developer Note:** During development, I pivoted from Gemini 1.5 Flash to **Llama-3.3-70B via Groq** to reduce hallucination rates observed with noisy scraped data. The raw SHL catalog pages are JavaScript-rendered, causing the scraper to capture browser warning artifacts instead of real descriptions. I solved this by building a curated metadata layer for core assessments and a type-inference fallback for the remainder, ensuring the FAISS index encodes meaningful semantic content rather than identical noise vectors.

The system uses two LLM calls per request state extraction at temperature=0 for determinism, and response generation at temperature=0.2 for natural language — with a Python-controlled ranking layer in between so the LLM never decides what gets recommended.

---

## 👨‍💻 Author

**Abhishek Das**  
📧 das.abhishek1121@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/abhishek-das1121/)

🔗 [Live Demo link](https://huggingface.co/spaces/abhi11-shek21/SHL_assesment_agent)

