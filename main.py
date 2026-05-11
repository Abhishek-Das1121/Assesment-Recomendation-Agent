"""
main.py — FastAPI entrypoint with startup preloading.

Preloads at startup:
- CatalogEngine (catalog.json)
- Retriever (FAISS index + sentence-transformer model)

This prevents cold-start latency during the first /chat call.
The evaluator allows 2 minutes for /health to return 200 before timing out.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import ChatRequest, ChatResponse
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload heavy resources at startup."""
    logger.info("Startup: loading CatalogEngine...")
    from catalog_engine import get_catalog_engine
    engine = get_catalog_engine()
    logger.info(f"Startup: catalog loaded ({len(engine.assessments)} assessments)")

    logger.info("Startup: loading Retriever (model + FAISS index)...")
    from retriever import get_retriever
    retriever = get_retriever()
    retriever._ensure_loaded()
    logger.info("Startup: retriever ready")

    yield  # Server is running

    logger.info("Shutdown: cleanup complete")


app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Evaluator polls this until 200 OK before running conversations."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Stateless conversational recommendation endpoint.

    Full messages[] history must be sent on every call.
    Server stores no session state.
    """
    try:
        from agent import process_chat
        return process_chat(request)
    except Exception as e:
        logger.exception(f"Unhandled error in /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False)
