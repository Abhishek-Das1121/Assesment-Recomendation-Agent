"""
models.py — Pydantic schemas for /chat request and response.
These are non-negotiable per the evaluator spec.
"""
from typing import Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]  # empty list when gathering context
    end_of_conversation: bool
