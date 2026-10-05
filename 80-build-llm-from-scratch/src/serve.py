"""Minimal API skeleton for the learner's evaluated capstone artifact."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4096)
    max_new_tokens: int = Field(default=32, ge=1, le=256)


class GenerateResponse(BaseModel):
    text: str
    model_version: str
    latency_ms: float


@dataclass
class Runtime:
    generate: Callable[[str, int], str]
    model_version: str


app = FastAPI(title="Tiny LLM Capstone")


def _not_loaded(prompt: str, max_new_tokens: int) -> str:
    raise RuntimeError("load your evaluated tokenizer/model artifact before serving")


runtime = Runtime(generate=_not_loaded, model_version="not-loaded")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_version": runtime.model_version}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    started = time.perf_counter()
    try:
        text = runtime.generate(request.prompt, request.max_new_tokens)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return GenerateResponse(
        text=text,
        model_version=runtime.model_version,
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )
