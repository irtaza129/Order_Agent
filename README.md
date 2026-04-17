# Project Structure for LangGraph Food Order Agent

- menu/menu.json: KFC menu representation (single items, deals)
- agent/mapping_agent.py: LLM mapping logic (validation, deal mapping)
- api/order_api.py: API integration for order submission
- main.py: Entry point for backend workflow
- tests/: Sample workflow and test cases

This structure supports modular development and easy iteration.

## Recent Architecture Revamp

- Summary: Major architecture cleanup and pipeline optimizations.
- Implemented a low-latency, in-process menu cache and background refresher to remove cold-call latency.
- Reduced HTTP connection overhead by introducing a persistent requests session.

### Details
- Added in-process caching and TTL handling (`MENU_CACHE`, `MENU_CACHE_TTL`) and warm/refresh helpers in `api/order_api.py`.
- Replaced per-request session setup with a module-level persistent `SESSION` to reuse HTTP connections.
- `get_menu()` now serves the menu from the in-process cache; `warm_menu_cache()` and `start_menu_cache_refresher()` keep it populated and fresh.
- Wired cache warm/refresher into FastAPI startup in `api_service.py` so the cache is warmed at boot and refreshed in the background.
- Fixed mapping LLM call bug (`text` → `input_text`) in `agent/mapping_agent.py`.

### Why
- Dramatically reduces end-to-end latency for common order flows by avoiding repeated remote menu fetches and expensive per-request connection setup.
- Keeps behavior unchanged while improving throughput and responsiveness in production-like loads.

### Notes
- The cache is process-local (in-memory). For multi-worker deployments consider a shared cache (Redis) or centralized invalidation.
- Env var `MENU_CACHE_TTL` controls cache lifetime (default 300s).
- Recommended follow-ups: invalidate in-process cache on menu mutations and consider async/httpx + aioredis refactor for high-concurrency.