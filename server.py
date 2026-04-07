#!/usr/bin/env python3
"""
Prior API Server - FastAPI with real-time SSE streaming.

Usage:
    uvicorn server:app --reload
    # or
    python server.py
"""
import json
import threading
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.graph import build_graph
from core.events import EventEmitter, EventType, emit
from db.vector import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    init_db()
    yield


app = FastAPI(
    title="Prior API",
    description="Research literature analysis with real-time streaming",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    question: str


class AnalysisResponse(BaseModel):
    question: str
    sub_queries: list[str]
    papers_count: int
    claims_count: int
    report: dict | None


def run_analysis_with_events(question: str, emitter: EventEmitter):
    """Run the analysis pipeline, emitting events along the way."""
    try:
        # Set the emitter for this thread so agents can emit events
        EventEmitter.set_current(emitter)

        graph = build_graph(adaptive=True, use_memory=True)
        initial_state = {
            "question": question,
            "sub_queries": [],
            "papers": [],
            "claims": [],
            "report": None,
            # Adaptive loop state
            "iteration": 1,
            "max_iterations": 3,
            "context_summary": None,
            "needs_more_info": False,
            "searched_queries": [],
            # Memory state
            "memory_context": [],
            "compressed_summary": None,
        }

        result = graph.invoke(initial_state)

        # Emit completion with the final report
        report = None
        if result.get("report"):
            try:
                report = json.loads(result["report"])
            except json.JSONDecodeError:
                report = {"raw": result["report"]}

        emitter.emit(
            EventType.COMPLETE,
            question=question,
            sub_queries=result.get("sub_queries", []),
            papers_count=len(result.get("papers", [])),
            claims_count=len(result.get("claims", [])),
            report=report,
        )

    except Exception as e:
        emitter.emit(EventType.ERROR, error=str(e))

    finally:
        EventEmitter.set_current(None)
        emitter.close()


@app.post("/analyze")
async def analyze(request: AnalysisRequest) -> StreamingResponse:
    """
    Run analysis with real-time SSE streaming.

    Returns a stream of Server-Sent Events showing progress,
    ending with the final report.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    emitter = EventEmitter()

    # Run analysis in background thread
    thread = threading.Thread(
        target=run_analysis_with_events,
        args=(request.question, emitter),
        daemon=True,
    )
    thread.start()

    return StreamingResponse(
        emitter.stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/analyze/sync", response_model=AnalysisResponse)
async def analyze_sync(request: AnalysisRequest) -> AnalysisResponse:
    """
    Run analysis synchronously (no streaming).

    Returns the complete result when done.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    graph = build_graph(adaptive=True, use_memory=True)
    initial_state = {
        "question": request.question,
        "sub_queries": [],
        "papers": [],
        "claims": [],
        "report": None,
        # Adaptive loop state
        "iteration": 1,
        "max_iterations": 3,
        "context_summary": None,
        "needs_more_info": False,
        "searched_queries": [],
        # Memory state
        "memory_context": [],
        "compressed_summary": None,
    }

    result = graph.invoke(initial_state)

    report = None
    if result.get("report"):
        try:
            report = json.loads(result["report"])
        except json.JSONDecodeError:
            report = {"raw": result["report"]}

    return AnalysisResponse(
        question=request.question,
        sub_queries=result.get("sub_queries", []),
        papers_count=len(result.get("papers", [])),
        claims_count=len(result.get("claims", [])),
        report=report,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
