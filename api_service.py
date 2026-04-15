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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent.langgraph_agent import process_order, classify_intent_only

app = FastAPI(title="KFC Order Agent API")

# CORS: allow calls from browser clients during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_HISTORY_TURNS = 4   # keep last N turns (1 turn = 1 user + 1 agent message)

# Map intents to short filler keys (client caches .wav for these keys)
FILLER_MAP = {
    "PLACE_ORDER": "processing",
    "GET_MENU": "fetching",
    "CREATE_ITEM": "processing",
    "UPDATE_ITEM": "processing",
    "DELETE_ITEM": "processing",
    "END_ORDER": None,
    "UNKNOWN": None,
    "SMALL_TALK": None
}

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


# Lightweight intent-only request/response for pre-flight intent check
class IntentRequest(BaseModel):
    text:       str
    session_id: str


class IntentResponse(BaseModel):
    intent: str
    filler: str | None = None


# Fast intent endpoint: classify intent and return filler key for client TTS cache
@app.post("/intent", response_model=IntentResponse)
def intent_endpoint(req: IntentRequest):
    history = _sessions.get(req.session_id, [])
    intent = classify_intent_only(req.text, _trim(history))
    filler = FILLER_MAP.get(intent)
    return IntentResponse(intent=intent, filler=filler)


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/order", response_model=OrderResponse)
def order_endpoint(req: OrderRequest):

    history = _sessions.get(req.session_id, [])

    # 🔴 EARLY SMALL TALK CHECK
    intent_result = classify_intent_only(req.text, _trim(history))

    if intent_result in ["UNKNOWN", "SMALL_TALK"]:
        return OrderResponse(
            voice_reply="Hey! Welcome to KFC  What would you like to order?",
            status="small_talk",
            order_id=None,
            order_total=None,
            session_id=req.session_id,
        )

    # ONLY THEN run graph
    result = process_order(
        raw_input=req.text,
        customer_id=req.customer_id,
        history=_trim(history),
    )

    # Append this turn to history and save back
    history.append({"role": "user",      "content": req.text})
    history.append({"role": "assistant", "content": result.get("voice_reply", "")})
    _sessions[req.session_id] = _trim(history)

    return OrderResponse(
        voice_reply=result.get("voice_reply", ""),
        status=result.get("status", "error"),
        order_id=str(result.get("order_id")) if result.get("order_id") is not None else None,
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