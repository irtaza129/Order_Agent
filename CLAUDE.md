# Order Agent — Project Context

## What This Is

A voice-first food ordering agent for **Savour Foods Pakistan**. Customers speak (STT) or type in English or Roman Urdu. The agent parses intent, maps items to the menu, manages a cart, and places orders via a backend REST API. Responses are fed to TTS so replies must be plain spoken text — no emojis, bullets, or special characters.

**Stack**: FastAPI · LangGraph · OpenAI (gpt-4o-mini via Responses API) · SentenceTransformer · Python 3.12

---

## Architecture

```
POST /order
  └─ api_service.py           # FastAPI entry; session + language state
       └─ process_order()     # langgraph_agent.py
            └─ LangGraph pipeline:
                 normalize → classify → [map | menu | end] → [cart | clarify | end]
```

### Pipeline Nodes

| Node | File | Purpose |
|------|------|---------|
| `node_normalize` | langgraph_agent.py:81 | Roman Urdu → English (full JSON LLM call) |
| `node_classify` | langgraph_agent.py:101 | Keyword-based intent routing |
| `node_map` | langgraph_agent.py:166 | Semantic + LLM item mapping |
| `node_cart` | langgraph_agent.py:214 | Merge items into cart |
| `node_menu` | langgraph_agent.py:144 | Return full menu text |
| `node_end` | langgraph_agent.py:251 | Checkout, small talk, invalid fallback |
| `node_clarify` | langgraph_agent.py:201 | Return clarification question |

### Intent Labels

- `PLACE_ORDER` → map → cart
- `GET_MENU` → menu
- `ORDER_QUERY` → end (returns cart total)
- `CHECKOUT` → end (confirm → place API order)
- `SMALL_TALK` → end (greeting reply)

---

## Key Files

| File | Role |
|------|------|
| [api_service.py](api_service.py) | FastAPI app, session state, startup hooks |
| [agent/langgraph_agent.py](agent/langgraph_agent.py) | LangGraph graph + all nodes |
| [agent/mapping_agent.py](agent/mapping_agent.py) | Semantic search, LLM mapping, Roman Urdu detection |
| [api/order_api.py](api/order_api.py) | Menu cache, order creation, background refresh |
| [menu/menu.json](menu/menu.json) | Local menu snapshot (fallback / dev reference) |

---

## Roman Urdu Support

Detection is word-level frozenset matching (~1ms, no LLM):
- `detect_roman_urdu()` in mapping_agent.py:100 — returns bool
- `"do"` (English) was intentionally removed from `_ROMAN_URDU_WORDS` because it caused false detection and the LLM would translate "do" as the Urdu number 2.
- `"order"` is also excluded for the same reason.

Two normalization paths:
1. **`normalize_if_urdu()`** (mapping_agent.py:175) — full JSON LLM call, used by `node_normalize` for final state
2. **`fast_normalize()`** (mapping_agent.py:126) — plain text LLM call, used inside semantic search only

The `pre_normalized=True` parameter on `find_similar_items()` skips the second LLM call when the query was already normalized by `node_normalize`. Always pass this from `_build_semantic_context()` to avoid double LLM cost.

---

## Semantic Item Matching

`find_similar_items()` in mapping_agent.py:314:
1. Encodes query with SentenceTransformer `all-MiniLM-L6-v2`
2. Cosine similarity against pre-built item embeddings
3. **Keyword boost**: `+0.08 * len(overlap)` for direct name-word matches — critical for short ambiguous names like "Special Choice"
4. **Context boost**: `+0.3` for pulao items on "ek plate" queries; `+0.25` for chicken piece disambiguation based on cart/history
5. Threshold: `0.2` (lowered from 0.3 — short item names like "Special Choice" score ~0.25-0.35 vs short queries)

Item embeddings are pre-built at startup via `build_item_embeddings()` and cached in `_ITEM_EMBEDDINGS_CACHE`. Call `invalidate_embeddings_cache()` before rebuilding on menu refresh.

---

## Clarification Logic

`should_ask_clarification()` in mapping_agent.py:421:

- **Never fires** if query contains a word from `_EXPLICIT_PULAO_VARIANTS` (`"special"`, `"single choice"`, `"without kabab"`, `"plain"`, `"pulao kabab"`). User named a specific variant — go straight to LLM mapping.
- **Always fires** for `"plate"` / `"ek plate"` queries with pulao candidates (implicit order).
- **Fires on close scores** only when top-2 items are from **different categories** (score gap < 0.15). Same-category near-ties are fine to pass to the LLM.

Early exit at mapping_agent.py:746: if `needs_clarification=True`, returns `NEEDS_CLARIFICATION` without calling the LLM. The LLM is still responsible for mapping when clarification is not needed.

---

## Performance & Startup

### Startup sequence (api_service.py:70)
1. `register_refresh_callback(_build_embeddings_from)` — wire embeddings rebuild to every menu refresh
2. `start_cache_refresh()` — fires initial menu fetch **immediately** in a background thread (not after 300s sleep)
3. `get_embedder()` — preloads 50MB SentenceTransformer model synchronously
4. `_wait_and_build_embeddings()` thread — polls up to 30s for menu, then pre-builds embeddings

### Menu cache (api/order_api.py)
- TTL: 300s (env `MENU_CACHE_TTL`)
- Background thread: fetch immediately on start, then loop with `sleep(TTL)`
- Never caches error payloads
- `get_menu_if_cached()` — non-blocking, returns `None` if not yet populated
- `register_refresh_callback(fn)` — register hooks to run after each successful refresh

### OpenAI calls
- Uses `client.responses.create()` (Responses API), **not** `chat.completions.create()`
- Model: `gpt-4o-mini` (env `OPENAI_MODEL`)
- Persistent `requests.Session` with retry adapter for HTTP calls to backend API

---

## Current Goals / Active Work

1. **Correct order mapping** — Every explicit item name must map to the right SKU without triggering NEEDS_CLARIFICATION. Key edge cases:
   - `"do one special pulao"` — "do" must not trigger Urdu detection
   - `"eik special pulao order hai mera"` — "special" keyword must suppress clarification and keyword boost must surface "Special Choice"
   - `"ek plate kardo"` — implicit, must ask which pulao variant

2. **Correct intent routing** — Checkout confirmation flow, small talk early exit, ORDER_QUERY for bill/total, GET_MENU for menu display.

3. **Roman Urdu support** — Mixed Urdu/English (Hinglish) input must be handled at every stage. Session language locks to `roman_urdu` once detected and all replies in that session use Roman Urdu.

4. **Latency / cold-start** — Embedder and menu must be ready before first request. Target: <500ms for warm requests, <3s for first request after startup.

---

## Menu — Savour Foods Categories

| Category | Notable Items |
|----------|--------------|
| Pulao | Single (645), Single Choice (665), Single Without Kabab (585), Special Choice (845), Pulao (335), Pulao Kabab (445), Chicken Piece Addon (205) |
| Breakfast | Halwa Puri (420), Anda Paratha (260), Chana Plate (220), Omelette Toast (295) |
| Burgers | Zinger (650), Krispo (610), Chicken (510), Double Patty (890) |
| Fried | Krispo Chicken Piece (299), Krispo Chicken 2 Piece (540), Wings 4/6/10 pcs, Fries S/M/L |
| Deals | Lunch (745), Dinner (899), Student (699), Family 1 (2499), Family 2 (3299) |
| Beverages | Soft Drink Can (120), Bottles 500ml/1L/1.5L, Tea (130), Coffee (199) |
| Desserts | Kheer Cup (210), Zarda Cup (195), Falooda (350), Ice Cream |
| Seasonal | Ramzan Deal 1 (899), Ramzan Deal 2 (1499), Eid Family Deal (3599) |
| Sides | Raita Extra (35), Salad Extra (35), Shami Kabab Addon (65) |

---

## Environment

```
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini          # optional override
API_BASE_URL=https://voiceai-hzyb.onrender.com
API_TIMEOUT=60
MENU_CACHE_TTL=300
```

Run server: `uvicorn api_service:app --reload`
Run CLI: `python main.py`
Test intent: `python main.py intent "ek special pulao chahiye"`

---

## Known Pitfalls

- **"do" false detection**: `"do"` was in `_ROMAN_URDU_WORDS` causing "do one special pulao" to be treated as Urdu (translating "do" as 2). Removed. Do not re-add it.
- **Double normalization**: `find_similar_items` calls `fast_normalize` internally. Always pass `pre_normalized=True` when query is already normalized to avoid a second LLM call.
- **Embeddings cache invalidation**: `build_item_embeddings()` is a no-op if cache exists. Call `invalidate_embeddings_cache()` first when rebuilding after a menu refresh.
- **TTS safety**: All voice replies must be plain text. Strip emojis, bullets, markdown. The `_strip_menu_echo()` function in `call_mapping_llm` catches LLM replies that accidentally echo the menu.
- **Responses API**: The codebase uses `client.responses.create()` with `resp.output_text`. Do not switch to `chat.completions.create()` — different response shape.
- **Session state lives in memory**: `_sessions` and `_session_state` in api_service.py are process-local dicts. Restarting the server clears all carts and histories.
