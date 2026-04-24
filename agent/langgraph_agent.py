import json
import os
import logging
from pathlib import Path
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from openai import OpenAI

from agent.mapping_agent import normalize_if_urdu, call_mapping_llm, detect_roman_urdu
from api.order_api import create_order

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

log = logging.getLogger("langgraph")
log.setLevel(logging.INFO)

_oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────
# HARD SAFETY LAYERS
# ─────────────────────────────

GREETINGS = {"salam", "assalam", "hello", "hi", "hey", "good morning"}
FOOD_SIGNAL = {"burger", "zinger", "fries", "drink", "chicken", "deal", "pulao", "combo", "meal", "rice", "piece"}

# Words that reveal intent from raw text — skip normalize LLM for these
_SKIP_NORM_CHECKOUT = frozenset([
    "bas", "done", "checkout", "finish", "yehi", "itna",
    "that's all", "thats all", "nothing else", "nothing more",
    "aur kuch nahi", "aur nahi", "sirf yehi",
])
_SKIP_NORM_MENU = frozenset(["menu", "dikha", "dikhao"])
_SKIP_NORM_CONFIRM = frozenset([
    "haan", "han", "yes", "ok", "ji", "jee", "theek", "confirm",
    "kardo", "proceed",
])

def _can_skip_normalize(raw: str) -> bool:
    """True when intent is clear from raw text and LLM translation would be wasted."""
    words = set(raw.lower().split())
    return bool(
        words & _SKIP_NORM_CHECKOUT
        or words & _SKIP_NORM_MENU
        or words & _SKIP_NORM_CONFIRM
        or words & GREETINGS
    )

def is_greeting(text: str) -> bool:
    return any(g in text.lower() for g in GREETINGS)

def is_food(text: str) -> bool:
    return any(f in text.lower() for f in FOOD_SIGNAL)

def is_confirmation(text: str) -> bool:
    t = text.lower().strip()
    # Check exact match first
    if t in {"yes", "y", "confirm", "ok", "haan", "han", "proceed", "done", "theek", "ji", "jee", "kardo"}:
        return True
    # Check if a strong confirmation phrase exists inside the sentence
    if any(w in t for w in["theek hai", "haan kardo", "yes please", "ok kardo"]):
        return True
    return False

# ─────────────────────────────
# STATE
# ─────────────────────────────

class OrderState(TypedDict):
    raw_input: str
    normalized_input: str
    reply_language: str
    session_language: str
    history: list
    pending_cart: list
    session_orders: list
    menu: dict
    intent: str
    mapping_result: dict
    api_response: dict
    voice_reply: str
    status: str
    awaiting_confirmation: bool


# ─────────────────────────────
# HELPERS
# ─────────────────────────────

def _cart_total(cart) -> float:
    return round(sum(i.get("total_price", 0) for i in cart), 2)

def _cart_summary(cart) -> str:
    return ", ".join(f"{i.get('name')} x{i.get('qty', 1)}" for i in cart) if cart else "empty"


# ─────────────────────────────
# NODE 1 — NORMALIZE
# ─────────────────────────────

def node_normalize(state: OrderState) -> OrderState:
    raw = state["raw_input"]
    session_lang = state.get("session_language", "en")

    if _can_skip_normalize(raw):
        # Intent is clear from raw text — no LLM needed, saves ~500ms
        was_urdu = detect_roman_urdu(raw)
        if session_lang == "en" and was_urdu:
            session_lang = "roman_urdu"
        return {
            **state,
            "normalized_input": raw,
            "reply_language": session_lang,
            "session_language": session_lang,
        }

    normalized, was_urdu = normalize_if_urdu(raw)
    if session_lang == "en" and was_urdu:
        session_lang = "roman_urdu"

    return {
        **state,
        "normalized_input": normalized,
        "reply_language": session_lang,
        "session_language": session_lang,
    }


# ─────────────────────────────
# NODE 2 — INTENT ROUTER
# ─────────────────────────────

def node_classify(state: OrderState) -> OrderState:
    text = state["normalized_input"].lower()
    awaiting = state.get("awaiting_confirmation", False)

    log.info("[intent] classifying: %s", text)

    # 1. Handle Active Checkout Confirmation State
    if awaiting:
        if is_confirmation(text):
            return {**state, "intent": "CHECKOUT"}
        else:
            # User said something else (e.g. asked for menu, added another item).
            # We clear the confirmation lock and evaluate their prompt normally.
            state["awaiting_confirmation"] = False

    # 2. End Order / Checkout intents
    # Also check raw_input so Roman Urdu stop-words survive normalization
    raw = state.get("raw_input", "").lower()
    _checkout_en = ("done", "that's all", "thats all", "that is all", "nothing else",
                    "nothing more", "that's it", "thats it", "checkout", "finish",
                    "place order", "aur kuch nahi")
    _checkout_ur = ("bas", "yehi hai", "bas yehi", "bas itna", "nahi chahiye aur",
                    "aur nahi", "sirf yehi", "itna hi")
    if (any(w in text for w in _checkout_en)
            or any(w in raw for w in _checkout_ur)
            or any(w in text for w in _checkout_ur)):
        return {**state, "intent": "CHECKOUT"}

    # 3. Menu queries
    if "menu" in text or "dikha" in text:
        return {**state, "intent": "GET_MENU"}

    if "bill" in text or "total" in text:
        return {**state, "intent": "ORDER_QUERY"}

    # 4. Ordering Signals
    ordering_signals = ("single", "choice", "pulao", "burger", "deal", "add", "want", "order", "kardo", "chahiye", "plate", "rice", "piece")
    if any(sig in text for sig in ordering_signals) or is_food(text):
        return {**state, "intent": "PLACE_ORDER"}

    # 5. Greetings / Small talk
    if is_greeting(text):
        return {**state, "intent": "SMALL_TALK"}

    # Default to order if unsure
    return {**state, "intent": "PLACE_ORDER"}


# ─────────────────────────────
# NODE 3 — MENU 
# ─────────────────────────────

def node_menu(state: OrderState) -> OrderState:
    menu = state.get("menu", {})
    items = menu.get("single_items",[])
    lang = state.get("reply_language", "en")
    
    if not items:
        reply = "Menu abhi available nahi hai." if lang == "roman_urdu" else "The menu is currently unavailable."
    else:
        # Full menu construction (Removed the [:10] limitation)
        item_list = ", ".join([f"{i.get('name')} (PKR {i.get('price')})" for i in items])
        if lang == "roman_urdu":
            reply = f"Hamare menu mein yeh items hain: {item_list}. Aap kya order karna pasand karenge?"
        else:
            reply = f"Our menu includes: {item_list}. What would you like to order?"
            
    return {**state, "status": "handled", "voice_reply": reply}


# ─────────────────────────────
# NODE 4 — MAP ORDER
# ─────────────────────────────

def node_map(state: OrderState) -> OrderState:
    menu = state.get("menu") or {}
    if isinstance(menu, list):
        menu = {"single_items": menu}

    try:
        result = call_mapping_llm(
            raw_order=state.get("normalized_input", ""),
            menu=menu,
            history=state.get("history",[]),
            reply_language=state.get("reply_language", "en"),
        )
        status = result.get("status", "VALID")
        
        # Sanity check fallback
        if not result.get("final_cart") and status == "VALID":
            status = "INVALID"

        # Early return for clarification
        if status == "NEEDS_CLARIFICATION":
            return {
                **state,
                "mapping_result": result,
                "status": "NEEDS_CLARIFICATION",
                "voice_reply": result.get("clarification_question"),
            }

    except Exception as e:
        log.exception("[node_map] mapping agent failed: %s", e)
        result = {"status": "INVALID", "final_cart":[]}
        status = "INVALID"

    return {**state, "mapping_result": result, "status": status}


def node_clarify(state: OrderState) -> OrderState:
    semantic = state.get("mapping_result", {})
    return {
        **state,
        "voice_reply": semantic.get("clarification_question", "Which item would you like?"),
        "status": "NEEDS_CLARIFICATION",
    }


# ─────────────────────────────
# NODE 5 — CART
# ─────────────────────────────

def node_cart(state: OrderState) -> OrderState:
    cart = list(state.get("pending_cart",[]))
    mapped_items = state["mapping_result"].get("final_cart",[])

    for item in mapped_items:
        item_name = item.get("name", "").lower().strip()
        existing = next(
            (i for i in cart
             if i["item_id"] == item["item_id"]
             or i.get("name", "").lower().strip() == item_name),
            None
        )
        if existing:
            existing["qty"] += item.get("qty", 1)
            existing["total_price"] = existing["unit_price"] * existing["qty"]
        else:
            cart.append(item)

    lang = state.get("reply_language", "en")
    llm_reply = state["mapping_result"].get("reply")

    # If the Mapping agent provided a conversational reply, use it.
    if llm_reply:
        reply = llm_reply
    else:
        # Fallback dynamic text
        if lang == "roman_urdu":
            reply = f"Maine item add kar diya hai. Total bill {_cart_total(cart)} rupay hai. Aur kuch chahiye?"
        else:
            reply = f"I have added that to your cart. Total is PKR {_cart_total(cart)}. Anything else?"

    return {
        **state,
        "pending_cart": cart,
        "voice_reply": reply,
        "status": "cart_updated",
    }


# ─────────────────────────────
# NODE 6 — END FLOW (CHECKOUT & REJECTS)
# ─────────────────────────────

def node_end(state: OrderState) -> OrderState:
    cart = state.get("pending_cart",[])
    intent = state.get("intent")
    awaiting = state.get("awaiting_confirmation", False)
    lang = state.get("reply_language", "en")

    # 1. INVALID MAPPING FALLBACK
    if state.get("status") == "INVALID":
        reply = "Maaf kijiye, mujhe samajh nahi aaya. Menu dekhna chahenge?" if lang == "roman_urdu" else "Sorry, I couldn't map that to our menu. Would you like to hear the menu?"
        return {**state, "voice_reply": reply, "status": "invalid"}

    # 2. SMALL TALK
    if intent == "SMALL_TALK":
        reply = "Wa alaikum salam! Kya order karna hai?" if lang == "roman_urdu" else "Hello! What would you like to order?"
        return {**state, "voice_reply": reply, "status": "small_talk"}

    # 3. ORDER QUERIES
    if intent == "ORDER_QUERY":
        reply = f"Aapka cart: {_cart_summary(cart)}. Total: {_cart_total(cart)}." if lang == "roman_urdu" else f"Cart: {_cart_summary(cart)} | Total: {_cart_total(cart)}"
        return {**state, "status": "handled", "voice_reply": reply}

    # 4. CHECKOUT LOGIC
    if intent == "CHECKOUT":
        if not cart:
            reply = "Aapka cart khali hai. Kya order karna hai?" if lang == "roman_urdu" else "Your cart is empty. What would you like to order?"
            return {**state, "status": "no_action", "voice_reply": reply, "awaiting_confirmation": False}

        if not awaiting:
            # Step 1 of Checkout: Ask for confirmation
            reply = f"Kya aap apna order confirm karte hain? {_cart_summary(cart)}. Total bill {_cart_total(cart)} rupay hai." if lang == "roman_urdu" else f"Confirm order: {_cart_summary(cart)}? Total: {_cart_total(cart)}."
            return {**state, "voice_reply": reply, "status": "awaiting_confirmation", "awaiting_confirmation": True}
        else:
            # Step 2 of Checkout: Confirmed, place API request!
            api = create_order(cart)
            order_id = api.get("id") or api.get("order_id", "")
            total = _cart_total(cart)
            if lang == "roman_urdu":
                success_reply = (
                    f"Shukriya! Aapka order place ho gaya. "
                    f"Aapka order ID {order_id} hai aur total bill {total} rupay hai. "
                    f"Apka din acha guzre!"
                )
            else:
                success_reply = (
                    f"Thank you! Your order has been placed. "
                    f"Your order ID is {order_id} and total is PKR {total}. "
                    f"Have a great day!"
                )
            return {
                **state,
                "pending_cart": [],
                "awaiting_confirmation": False,
                "api_response": api,
                "voice_reply": success_reply,
                "status": "success",
            }

    return {**state, "status": "handled"}


# ─────────────────────────────
# GRAPH
# ─────────────────────────────

def build_graph():
    g = StateGraph(OrderState)

    g.add_node("normalize", node_normalize)
    g.add_node("classify", node_classify)
    g.add_node("menu", node_menu)
    g.add_node("map", node_map)
    g.add_node("cart", node_cart)
    g.add_node("end", node_end)
    g.add_node("clarify", node_clarify)
    
    g.set_entry_point("normalize")
    g.add_edge("normalize", "classify")

    # Intent Routing
    g.add_conditional_edges("classify", lambda s: s["intent"], {
        "PLACE_ORDER": "map",
        "GET_MENU": "menu",
        "ORDER_QUERY": "end",
        "CHECKOUT": "end",
        "SMALL_TALK": "end",
    })

    # Mapping Result Routing
    g.add_conditional_edges("map", lambda s: s["status"], {
        "NEEDS_CLARIFICATION": "clarify",
        "VALID": "cart",
        "INVALID": "end",
    })
    
    # Terminal edges
    g.add_edge("menu", END)
    g.add_edge("cart", END)
    g.add_edge("end", END)
    g.add_edge("clarify", END)

    return g.compile()


order_graph = build_graph()


# ─────────────────────────────
# ENTRY
# ─────────────────────────────

def process_order(**kwargs):
    log.info("[process_order] Processing input: %s", kwargs.get('raw_input'))

    raw_history = kwargs.get("history", []) or []
    safe_history = raw_history[-10:]

    # Run LangGraph Execution
    raw_result = order_graph.invoke({
        "raw_input": kwargs["raw_input"],
        "normalized_input": kwargs["raw_input"],
        "reply_language": kwargs.get("reply_language", "en"),
        "session_language": kwargs.get("session_language", "en"),
        "history": safe_history,
        "pending_cart": kwargs.get("pending_cart", []),
        "session_orders": kwargs.get("session_orders",[]),
        "menu": kwargs.get("cached_menu", {}),
        "intent": "",
        "mapping_result": {},                           
        "api_response": {},
        "voice_reply": "",
        "status": "pending",
        "awaiting_confirmation": kwargs.get("awaiting_confirmation", False),
    })
    
    # 🔴 FIX: Construct a strict, bounded response dictionary.
    # This guarantees that 'menu', 'history', and 'mapping_result' are securely stripped out 
    # and NEVER returned in the API payload or leaked between sessions.
    clean_result = {
        "voice_reply": raw_result.get("voice_reply", ""),
        "status": raw_result.get("status", "error"),
        "intent": raw_result.get("intent", "UNKNOWN"),
        "pending_cart": raw_result.get("pending_cart",[]),
        "reply_language": raw_result.get("reply_language", "en"),
        "awaiting_confirmation": raw_result.get("awaiting_confirmation", False),
        "api_response": raw_result.get("api_response", {})
    }

    log.info("[process_order] Cleaned Result: %s", clean_result)
    return clean_result


def classify_intent_only(text: str, history=None) -> str:
    """Fast keyword-only intent classifier (~0ms). Used for filler audio selection."""
    t = text.lower()
    words = set(t.split())
    if words & {"bas", "done", "checkout", "yehi", "itna", "finish"} or "that's all" in t or "thats all" in t:
        return "CHECKOUT"
    if words & {"menu", "dikha", "dikhao"}:
        return "GET_MENU"
    if words & {"bill", "total", "kitna"}:
        return "ORDER_QUERY"
    if is_greeting(t):
        return "SMALL_TALK"
    return "PLACE_ORDER"