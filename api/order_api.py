"""
Order API — api/order_api.py
"""

import os
import json
import logging

import requests
from dotenv import load_dotenv
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = os.getenv("API_BASE_URL", "https://voiceai-hzyb.onrender.com")
TIMEOUT  = int(os.getenv("API_TIMEOUT", 60))
HEADERS  = {"accept": "application/json", "Content-Type": "application/json"}

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ORDER_API] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("order_api")


def _session():
    session = requests.Session()
    retry   = Retry(total=3, backoff_factor=2, status_forcelist=[502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _handle(response, label: str) -> dict:
    log.info("← %s | status=%s | body=%s", label, response.status_code, response.text[:300])
    try:
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        return {"error": str(e), "status_code": response.status_code, "body": response.text}
    except Exception as e:
        return {"error": str(e)}


def get_menu():
    url = f"{BASE_URL}/menu"
    log.info("→ GET %s", url)
    resp = _session().get(url, headers=HEADERS, timeout=TIMEOUT)
    return _handle(resp, "GET /menu")


def create_menu_item(name, category, price, description="", available=True):
    url     = f"{BASE_URL}/menu"
    payload = {"name": name, "category": category,
               "description": description, "price": price, "available": available}
    log.info("→ POST %s | payload=%s", url, json.dumps(payload))
    resp = _session().post(url, json=payload, headers=HEADERS, timeout=TIMEOUT)
    return _handle(resp, "POST /menu")


def update_menu_item(item_id, name, category, price, description="", available=True):
    url     = f"{BASE_URL}/menu/{item_id}"
    payload = {"name": name, "category": category,
               "description": description, "price": price, "available": available}
    log.info("→ PUT %s | payload=%s", url, json.dumps(payload))
    resp = _session().put(url, json=payload, headers=HEADERS, timeout=TIMEOUT)
    return _handle(resp, f"PUT /menu/{item_id}")


def delete_menu_item(item_id):
    url = f"{BASE_URL}/menu/{item_id}"
    log.info("→ DELETE %s", url)
    resp = _session().delete(url, headers=HEADERS, timeout=TIMEOUT)
    return _handle(resp, f"DELETE /menu/{item_id}")


def create_order(cart, customer_name: str = "Guest"):
    """
    POST /orders
    Transforms internal cart into API payload:
        {"customer_name": str, "items": [{"menu_item_id": int, "quantity": int}]}
    """
    url   = f"{BASE_URL}/orders"
    items = [
        {
            "menu_item_id": int(item["item_id"]),
            "quantity":     int(item.get("qty", 1)),
        }
        for item in cart
    ]
    payload = {"customer_name": customer_name, "items": items}

    log.info("→ POST %s | payload=%s", url, json.dumps(payload))
    resp   = _session().post(url, json=payload, headers=HEADERS, timeout=TIMEOUT)
    result = _handle(resp, "POST /orders")
    log.info("Order placed in DB: %s", json.dumps(result))
    return result