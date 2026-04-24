from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import logging
import logging.config

import asyncio
import threading

from agent.langgraph_agent import process_order, classify_intent_only
from agent.mapping_agent import detect_roman_urdu
import agent.mapping_agent as mapping_agent
from api.order_api import get_cached_menu, get_menu_if_cached, start_cache_refresh, register_refresh_callback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
log = logging.getLogger("api")

_sessions = {}
_session_state = {}

def get_state(session_id: str) -> dict:
    """Return per-session mutable state, creating defaults when missing."""
    if session_id not in _session_state:
        _session_state[session_id] = {
            "pending_cart": [],
            "session_orders":[],
            "session_language": "en", # 🔴 NEW: Track session language globally
            "awaiting_confirmation": False,
        }
    return _session_state[session_id]

class OrderRequest(BaseModel):
    text: str
    session_id: str
    customer_id: str | None = None

def _build_embeddings_from(menu_data):
    """Rebuild item embeddings whenever the menu is refreshed."""
    try:
        m = {"single_items": menu_data} if isinstance(menu_data, list) else menu_data
        if m.get("single_items"):
            mapping_agent.invalidate_embeddings_cache()
            mapping_agent.build_item_embeddings(m)
            log.info("startup/refresh: item embeddings built (%d items)", len(m["single_items"]))
    except Exception as e:
        log.exception("_build_embeddings_from: failed: %s", e)


def _wait_and_build_embeddings():
    """Poll until initial menu fetch completes, then pre-build embeddings."""
    for _ in range(30):                      # up to 30 s wait
        data = get_menu_if_cached()
        if data:
            _build_embeddings_from(data)
            return
        threading.Event().wait(1)
    log.warning("startup: embeddings not pre-built — menu unavailable after 30 s")


@app.on_event("startup")
def startup():
    log.info("startup: warming menu cache and starting refresher")
    # Rebuild embeddings automatically on every background menu refresh
    register_refresh_callback(_build_embeddings_from)
    start_cache_refresh()                    # initial fetch fires immediately in background

    # Preload heavy models (embedder) to avoid per-request latency
    try:
        log.info("startup: preloading semantic embedder")
        mapping_agent.get_embedder()
        log.info("startup: embedder loaded")
    except Exception as e:
        log.exception("startup: embedder preload failed: %s", e)

    # Pre-build item embeddings once menu arrives (non-blocking)
    threading.Thread(target=_wait_and_build_embeddings, daemon=True).start()

@app.post("/order")
async def order(req: OrderRequest):
    log.info("/order: session=%s customer=%s text=%s", req.session_id, req.customer_id, req.text)
    history = _sessions.get(req.session_id, [])
    state = get_state(req.session_id)

    if detect_roman_urdu(req.text):
        state["session_language"] = "roman_urdu"

    cached_menu = get_cached_menu()
    if isinstance(cached_menu, list):
        cached_menu = {"single_items": cached_menu}

    result = await asyncio.to_thread(
        process_order,
        raw_input=req.text,
        customer_id=req.customer_id,
        history=history,
        pending_cart=state["pending_cart"],
        session_orders=state["session_orders"],
        reply_language=state["session_language"],
        session_language=state["session_language"],
        cached_menu=cached_menu,
        awaiting_confirmation=state["awaiting_confirmation"],
    )

    history += [
        {"role": "user", "content": req.text},
        {"role": "assistant", "content": result.get("voice_reply", "")},
    ]
    _sessions[req.session_id] = history[-10:]

    state["pending_cart"] = result.get("pending_cart", [])
    state["session_language"] = result.get("reply_language", state["session_language"])
    state["awaiting_confirmation"] = result.get("awaiting_confirmation", False)
    _session_state[req.session_id] = state

    log.info("/order: updated state pending_cart=%s awaiting_confirmation=%s", len(state["pending_cart"]), state["awaiting_confirmation"])

    result.pop("menu", None)
    return result


@app.post("/intent")
def intent(req: OrderRequest):
    return {"intent": classify_intent_only(req.text)}