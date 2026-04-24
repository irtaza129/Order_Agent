"""
main_voice.py — Voice Gateway for Savour Foods Order Agent.

Architecture (Golden Rule):
  Mic → Deepgram STT → text → LangGraph Agent → text → Deepgram TTS → Speaker

Latency strategy:
  1. STT returns text.
  2. classify_intent_only() runs instantly (~0ms, keyword-only).
  3. Filler clip starts playing immediately via sd.play() (non-blocking).
  4. process_order() runs concurrently while filler plays.
  5. sd.stop() cuts filler; real TTS response streams to speaker.

Result: user hears audio within ~100ms of finishing their sentence instead
of waiting 1.5–2s for the full agent + TTS pipeline.
"""

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import agent.mapping_agent as mapping_agent
from agent.langgraph_agent import process_order, classify_intent_only
from api.order_api import get_cached_menu, start_cache_refresh, register_refresh_callback
from voice.deepgram_stt import transcribe_once
from voice.deepgram_tts import speak
import voice.filler as filler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("voice_main")

_MAX_HISTORY = 10
_EXIT_WORDS = frozenset(["quit", "exit", "goodbye", "bye", "khuda hafiz", "alvida"])


_embed_signature: tuple | None = None


def _build_embeddings_from(menu_data) -> None:
    global _embed_signature
    try:
        m = {"single_items": menu_data} if isinstance(menu_data, list) else menu_data
        items = m.get("single_items") or []
        if not items:
            return
        sig = (len(items), id(items))
        if sig == _embed_signature:
            return
        mapping_agent.invalidate_embeddings_cache()
        mapping_agent.build_item_embeddings(m)
        _embed_signature = sig
        log.info("Embeddings rebuilt: %d items", len(items))
    except Exception as e:
        log.exception("Embedding rebuild failed: %s", e)


async def _warmup() -> None:
    """Pre-load all heavy resources so the first turn is fast."""
    log.info("Warmup: loading sentence embedder...")
    mapping_agent.get_embedder()

    log.info("Warmup: starting menu cache refresh...")
    register_refresh_callback(_build_embeddings_from)
    start_cache_refresh()

    # Filler preload runs in parallel with menu fetch (both are network-bound).
    filler_task = asyncio.create_task(asyncio.to_thread(filler.preload))

    for _ in range(15):
        menu = get_cached_menu()
        if menu:
            _build_embeddings_from(menu)
            break
        await asyncio.sleep(1)
    else:
        log.warning("Warmup: menu not ready after 15s — continuing anyway")

    await filler_task
    log.info("Warmup complete — ready.")


async def voice_loop() -> None:
    session_id = str(uuid.uuid4())[:8]
    history: list[dict] = []
    pending_cart: list[dict] = []
    session_language = "en"
    awaiting_confirmation = False

    print("\n" + "=" * 52)
    print("  Savour Foods Voice Agent  (session: %s)" % session_id)
    print("  Speak in English or Roman Urdu")
    print("  Say 'goodbye' or Ctrl+C to exit")
    print("=" * 52 + "\n")

    while True:
        print("Listening...")
        try:
            text = await asyncio.to_thread(transcribe_once)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.error("STT error: %s", e)
            print(f"[STT error — retrying: {e}]")
            await asyncio.sleep(0.5)
            continue

        if not text:
            continue

        t_stt_done = time.perf_counter()
        print(f"You  : {text}")

        # ── Exit ──────────────────────────────────────────────────────────────
        if text.lower().strip() in _EXIT_WORDS:
            farewell = (
                "Khuda hafiz! Apka din acha guzre!"
                if session_language == "roman_urdu"
                else "Goodbye! Have a great day!"
            )
            print(f"Agent: {farewell}")
            await asyncio.to_thread(speak, farewell)
            break

        # Step 1 — instant intent guess, start filler clip immediately.
        quick_intent = classify_intent_only(text)
        filler.play(quick_intent, session_language)

        # Step 2 — full agent processing runs while filler plays.
        cached_menu = get_cached_menu()
        if isinstance(cached_menu, list):
            cached_menu = {"single_items": cached_menu}

        try:
            result = await asyncio.to_thread(
                process_order,
                raw_input=text,
                customer_id=session_id,
                history=history[-_MAX_HISTORY:],
                pending_cart=pending_cart,
                session_orders=[],
                reply_language=session_language,
                session_language=session_language,
                cached_menu=cached_menu or {},
                awaiting_confirmation=awaiting_confirmation,
            )
        except Exception as e:
            filler.stop()
            log.exception("Agent error: %s", e)
            err_reply = "Maaf kijiye, kuch masla ho gaya. Dobara try karein."
            print(f"Agent: {err_reply}")
            await asyncio.to_thread(speak, err_reply)
            continue

        t_agent_done = time.perf_counter()

        # Step 3 — cut filler, stream real response.
        filler.stop()

        reply = result.get("voice_reply", "Maaf kijiye, samajh nahi aaya.")
        print(f"Agent: {reply}")

        pending_cart = result.get("pending_cart", pending_cart)
        session_language = result.get("reply_language", session_language)
        awaiting_confirmation = result.get("awaiting_confirmation", False)

        history.extend([
            {"role": "user", "content": text},
            {"role": "assistant", "content": reply},
        ])
        history = history[-_MAX_HISTORY:]

        await asyncio.to_thread(speak, reply)

        log.info(
            "turn: agent=%.2fs tts-start=%.2fs (intent=%s)",
            t_agent_done - t_stt_done,
            time.perf_counter() - t_agent_done,
            quick_intent,
        )

        if result.get("status") == "success":
            log.info("Order placed — ending voice session %s", session_id)
            break


async def main() -> None:
    await _warmup()
    await voice_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nVoice agent stopped.")
