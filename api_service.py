"""
api_service.py — FastAPI wrapper around the LangGraph order agent.
Maintains per-session conversation history using session_id.

Run:
    uvicorn api_service:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /order          — place or manage an order
    DELETE /session/{id} — clear a session's history
    GET  /session/{id}   — inspect a session's history (debug)
    GET  /               — health check
"""

import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent.langgraph_agent import process_order

app = FastAPI(title="KFC Order Agent API")

MAX_HISTORY_TURNS = 4   # keep last N turns (1 turn = 1 user + 1 agent message)

# ── In-memory session store ────────────────────────────────────────────────
# key:   session_id (str)
# value: list of {"role": "user"|"assistant", "content": str}
_sessions: dict[str, list] = {}


def _trim(history: list) -> list:
    """Keep only the last MAX_HISTORY_TURNS turns."""
    return history[-(MAX_HISTORY_TURNS * 2):]


# ── Request / Response schemas ─────────────────────────────────────────────

class OrderRequest(BaseModel):
    text:        str
    session_id:  str                  # required — whisper generates UUID per call session
    customer_id: str | None = None


class OrderResponse(BaseModel):
    voice_reply: str
    status:      str
    order_id:    str | None = None
    order_total: float | None = None
    session_id:  str


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/order", response_model=OrderResponse)
def order_endpoint(req: OrderRequest):
    # 1. Load existing history for this session (empty list if new session)
    history = _sessions.get(req.session_id, [])

    # 2. Call the LangGraph agent with history
    result = process_order(
        raw_input=req.text,
        customer_id=req.customer_id,
        history=_trim(history),
    )

    # 3. Append this turn to history and save back
    history.append({"role": "user",      "content": req.text})
    history.append({"role": "assistant", "content": result["voice_reply"]})
    _sessions[req.session_id] = _trim(history)

    return OrderResponse(
        voice_reply=result["voice_reply"],
        status=result["status"],
        order_id=str(result["order_id"]) if result.get("order_id") is not None else None,
        order_total=result.get("order_total"),
        session_id=req.session_id,
    )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Call this when a voice call ends to free memory."""
    if session_id in _sessions:
        del _sessions[session_id]
    return {"cleared": session_id}


@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Debug endpoint — inspect what history is stored for a session."""
    history = _sessions.get(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "turns": len(history) // 2, "history": history}


@app.get("/")
def root():
    return {
        "status":   "ok",
        "message":  "KFC Order Agent API is running",
        "sessions": len(_sessions),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)