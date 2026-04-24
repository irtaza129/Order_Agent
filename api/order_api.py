import os, json, logging, threading, time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = os.getenv("API_BASE_URL", "https://voiceai-hzyb.onrender.com")
TIMEOUT = int(os.getenv("API_TIMEOUT", 60))

log = logging.getLogger("order_api")

# ── SINGLE SESSION (important optimization) ──
_session = requests.Session()
_session.mount("https://", HTTPAdapter(
    max_retries=Retry(total=3, backoff_factor=1)
))

HEADERS = {"accept": "application/json", "Content-Type": "application/json"}

# ── CACHE ─────────────────────────────
MENU_CACHE = {"data": None, "ts": 0}
CACHE_TTL = int(os.getenv("MENU_CACHE_TTL", 300))
LOCK = threading.Lock()


def _request(method, url, **kwargs):
    try:
        log.debug("_request: method=%s url=%s kwargs=%s", method, url, {k: v for k, v in kwargs.items() if k != 'json'})
        r = _session.request(method, url, timeout=TIMEOUT, headers=HEADERS, **kwargs)
        r.raise_for_status()
        data = r.json()
        log.debug("_request: success url=%s status=%s", url, r.status_code)
        return data
    except Exception as e:
        log.exception("_request: failed url=%s error=%s", url, e)
        return {"error": str(e)}


# ── MENU ─────────────────────────────
def get_menu():
    log.info("get_menu: fetching live menu from %s", BASE_URL)
    return _request("GET", f"{BASE_URL}/menu")


def get_cached_menu():
    with LOCK:
        if MENU_CACHE["data"] and time.time() - MENU_CACHE["ts"] < CACHE_TTL:
            log.debug("get_cached_menu: returning cached menu (age=%s)", time.time()-MENU_CACHE["ts"])
            return MENU_CACHE["data"]

    data = get_menu()
    
    # 🔴 FIX: NEVER CACHE AN ERROR PAYLOAD
    if "error" not in data:
        with LOCK:
            MENU_CACHE["data"] = data
            MENU_CACHE["ts"] = time.time()
        log.info("get_cached_menu: cache refreshed successfully.")
    else:
        log.warning("get_cached_menu: API returned error, refusing to update cache.")

    return data


# ── ORDERS ───────────────────────────
def create_order(cart, customer_name="Guest"):
    items = []
    for i in cart:
        try:
            items.append({"menu_item_id": int(i["item_id"]), "quantity": int(i.get("qty", 1))})
        except (ValueError, TypeError):
            log.warning("create_order: skipping item with non-integer item_id=%s", i.get("item_id"))
    payload = {
        "customer_name": customer_name,
        "items": items,
    }
    log.info("create_order: customer=%s items=%s", customer_name, len(cart))
    res = _request("POST", f"{BASE_URL}/orders", json=payload)
    log.debug("create_order: response=%s", res)
    return res


# ── BACKGROUND REFRESH ───────────────
_on_refresh_callbacks: list = []


def register_refresh_callback(fn):
    """Register a callable invoked after each successful menu refresh."""
    _on_refresh_callbacks.append(fn)


def get_menu_if_cached():
    """Return cached menu without blocking HTTP; None if not yet populated."""
    with LOCK:
        return MENU_CACHE["data"]


def start_cache_refresh():
    def _populate(data):
        if "error" not in data:
            with LOCK:
                MENU_CACHE["data"] = data
                MENU_CACHE["ts"] = time.time()
            for cb in _on_refresh_callbacks:
                try:
                    cb(data)
                except Exception as e:
                    log.exception("refresh callback failed: %s", e)
            return True
        return False

    def loop():
        # Initial fetch immediately so first request hits the cache
        try:
            log.info("start_cache_refresh: initial menu fetch")
            _populate(get_menu())
        except Exception as e:
            log.exception("start_cache_refresh: initial fetch failed: %s", e)

        while True:
            time.sleep(CACHE_TTL)
            try:
                log.info("start_cache_refresh: periodic refresh")
                _populate(get_menu())
            except Exception as e:
                log.exception("start_cache_refresh: refresh failed: %s", e)

    threading.Thread(target=loop, daemon=True).start()