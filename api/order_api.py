"""
Order API — api/order_api.py
"""

import os
import uuid

import requests
from dotenv import load_dotenv
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = os.getenv("API_BASE_URL", "https://voiceai-hzyb.onrender.com")
TIMEOUT  = int(os.getenv("API_TIMEOUT", 60))
HEADERS  = {"accept": "application/json", "Content-Type": "application/json"}


def _session():
    session = requests.Session()
    retry   = Retry(total=3, backoff_factor=2, status_forcelist=[502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _handle(response):
    try:
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        return {"error": str(e), "status_code": response.status_code, "body": response.text}
    except Exception as e:
        return {"error": str(e)}


def get_menu():
    resp = _session().get(f"{BASE_URL}/menu", headers=HEADERS, timeout=TIMEOUT)
    return _handle(resp)


def create_menu_item(name, category, price, description="", available=True):
    payload = {"name": name, "category": category,
               "description": description, "price": price, "available": available}
    resp = _session().post(f"{BASE_URL}/menu", json=payload, headers=HEADERS, timeout=TIMEOUT)
    return _handle(resp)


def update_menu_item(item_id, name, category, price, description="", available=True):
    payload = {"name": name, "category": category,
               "description": description, "price": price, "available": available}
    resp = _session().put(f"{BASE_URL}/menu/{item_id}", json=payload, headers=HEADERS, timeout=TIMEOUT)
    return _handle(resp)


def delete_menu_item(item_id):
    resp = _session().delete(f"{BASE_URL}/menu/{item_id}", headers=HEADERS, timeout=TIMEOUT)
    return _handle(resp)


def create_order(cart):
    order_total = sum(item.get("total_price", 0) for item in cart)
    order_id    = f"ORD-{uuid.uuid4().hex[:8].upper()}"
    return {"order_id": order_id, "status": "created",
            "cart": cart, "order_total": order_total}