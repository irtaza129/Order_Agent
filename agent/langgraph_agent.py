"""
KFC Order & Menu Management Agent — agent/langgraph_agent.py

Graph flow:
    load_menu → classify_intent → [router] → place_order_map → [router] → place_order_exec   → END
                                                                        → ask_clarification   → END
                                                                        → handle_unknown      → END
                                           → get_menu                                         → END
                                           → create_item                                      → END
                                           → update_item                                      → END
                                           → delete_item                                      → END
                                           → end_order                                        → END
                                           → handle_unknown                                   → END
"""

import json
import os
from pathlib import Path
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from api.order_api import (
    create_menu_item,
    create_order,
    delete_menu_item,
    get_menu,
    update_menu_item,
)
from agent.mapping_agent import call_mapping_llm

_oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── State ─────────────────────────────────────────────────────────────────────

class OrderState(TypedDict):
    raw_input:      str
    customer_id:    Optional[str]
    history:        list        # rolling window of past turns
    menu:           dict
    intent:         str
    intent_payload: dict
    mapping_result: dict
    api_response:   dict
    voice_reply:    str
    status:         str


# ── History helper ────────────────────────────────────────────────────────────

def _history_block(history: list) -> str:
    """
    Converts history list into a compact text block injected into prompts.
    e.g.
      [Conversation so far]
      User: I want original recipe chicken
      Agent: Got it! Added Original Recipe Chicken (3 pcs). Order ORD-123. Anything else?
    """
    if not history:
        return ""
    lines = ["[Conversation so far]"]
    for msg in history:
        role = "User" if msg["role"] == "user" else "Agent"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) + "\n\n"


# ── Intent prompt ─────────────────────────────────────────────────────────────

_INTENT_PROMPT = """\
You are the intent classifier for a KFC AI ordering system.
Classify the user message and return json. Output ONLY valid JSON, no markdown.

Intents: PLACE_ORDER | GET_MENU | CREATE_ITEM | UPDATE_ITEM | DELETE_ITEM | END_ORDER | UNKNOWN

Output schema:
{
  "intent": "<intent>",
  "payload": {
    // PLACE_ORDER  -> {}
    // GET_MENU     -> {}
    // CREATE_ITEM  -> {"name": str, "category": str, "price": float, "description": str, "available": true}
    // UPDATE_ITEM  -> {"item_id": int, "name": str, "category": str, "price": float, "description": str, "available": bool}
    // DELETE_ITEM  -> {"item_id": int}
    // END_ORDER    -> {}
    // UNKNOWN      -> {}
  }
}

Rules:
- Default category to "main course", description to "", available to true when not stated.
- Extract item_id as integer from phrases like "item 8". If not found set -1.
- PLACE_ORDER is the DEFAULT for any food or drink request, even vague ones like
  "that corn thing", "the potato one", "something spicy", "that burger".
  When in doubt, choose PLACE_ORDER — the mapping agent will handle fuzzy matching.
- Use the conversation history to resolve references like "repeat my order",
  "add the same again", "make that two" — look at what was previously ordered.
- END_ORDER when the customer signals they have finished ordering and don't want
  anything else. Look for the intent behind the words, not just keywords — if the
  customer is indicating they are satisfied and wrapping up the interaction, that is
  END_ORDER. Examples: "thats all", "that's it", "thank you that'd be it",
  "i'm good", "nothing else", "no that's everything", "all done", "bye".
- UNKNOWN only for truly unrelated requests like "play music" or "what's the weather".
"""


def classify_intent_only(text: str, history: list | None = None) -> str:
    """Fast intent-only classifier used by the /intent endpoint.
    Returns the intent string (e.g. PLACE_ORDER, GET_MENU, UNKNOWN).
    This performs a single, minimal LLM call and returns quickly.
    """
    input_with_history = _history_block(history or []) + f"Classify this message and return json: {text}"
    resp = _oai.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        instructions=_INTENT_PROMPT,
        input=input_with_history,
        text={"format": {"type": "json_object"}},
    )
    try:
        parsed = json.loads(resp.output_text)
        return parsed.get("intent", "UNKNOWN")
    except Exception:
        return "UNKNOWN"


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_load_menu(state: OrderState) -> OrderState:
    print("[Node: load_menu] Fetching live menu ... (may take ~30s on cold start)")
    try:
        result = get_menu()
        if "error" in result:
            print(f"[Node: load_menu] API error: {result['error']} - using empty menu")
            menu = {"single_items": [], "deals": []}
        else:
            items = result if isinstance(result, list) else result.get("items", [])
            menu  = {"single_items": items, "deals": []}
            print(f"[Node: load_menu] {len(items)} items loaded")
    except Exception as e:
        print(f"[Node: load_menu] Could not reach server ({e}) - using empty menu")
        menu = {"single_items": [], "deals": []}
    return {**state, "menu": menu, "status": "pending"}


def node_classify_intent(state: OrderState) -> OrderState:
    print(f"[Node: intent] '{state['raw_input']}'")

    # Prepend history so the classifier understands references like "repeat my order"
    input_with_history = (
        _history_block(state["history"])
        + f"Classify this message and return json: {state['raw_input']}"
    )

    resp = _oai.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        instructions=_INTENT_PROMPT,
        input=input_with_history,
        text={"format": {"type": "json_object"}},
    )
    parsed = json.loads(resp.output_text)
    intent = parsed.get("intent", "UNKNOWN")
    print(f"[Node: intent] → {intent}")
    return {**state, "intent": intent, "intent_payload": parsed.get("payload", {})}


def node_place_order_map(state: OrderState) -> OrderState:
    print("[Node: place_order_map] Mapping order ...")
    mapping_result = call_mapping_llm(
        raw_order=state["raw_input"],
        menu=state["menu"],
        history=state["history"],       # ← pass history to mapping agent
    )
    needs_clarification = mapping_result.get("needs_clarification", False)
    print(f"[Node: place_order_map] valid={mapping_result['is_valid']}  "
          f"needs_clarification={needs_clarification}")
    return {**state, "mapping_result": mapping_result}


def node_ask_clarification(state: OrderState) -> OrderState:
    question = state["mapping_result"].get(
        "clarification_question",
        "Could you clarify your order? We have a few options available."
    )
    print(f"[Node: ask_clarification] → '{question}'")
    return {**state, "api_response": {}, "voice_reply": question, "status": "clarification"}


def node_place_order_exec(state: OrderState) -> OrderState:
    mapping = state["mapping_result"]
    if not mapping.get("is_valid"):
        return {**state, "api_response": {}, "status": "error",
                "voice_reply": mapping.get("invalid_reason", "Sorry, I couldn't process that order.")}
    cart         = mapping["final_cart"]
    api_response = create_order(cart, customer_name=state.get("customer_id") or "Guest")
    item_names   = ", ".join(i["name"] for i in cart)
    savings      = mapping.get("savings", 0)
    savings_text = f" You saved PKR {savings} with a deal!" if savings > 0 else ""
    # real API returns "id", fallback to "order_id" for compatibility
    order_id     = api_response.get("id") or api_response.get("order_id", "N/A")
    voice_reply  = (f"Got it! I've added {item_names} to your order.{savings_text} "
                    f"Your order ID is {order_id}. Anything else?")
    print(f"[Node: place_order_exec] Order created: {order_id}")
    api_response["order_id"] = order_id  # normalise key for downstream use
    return {**state, "api_response": api_response, "voice_reply": voice_reply, "status": "success"}


def node_get_menu(state: OrderState) -> OrderState:
    items = state["menu"].get("single_items", [])
    if not items:
        voice_reply = "The menu is currently empty."
    else:
        lines       = [f"{i.get('name')} — PKR {i.get('price', '?')}" for i in items]
        voice_reply = "Here's our menu:\n" + "\n".join(f"  • {l}" for l in lines)
    print(f"[Node: get_menu] Returning {len(items)} items")
    return {**state, "api_response": state["menu"], "voice_reply": voice_reply, "status": "success"}


def node_create_item(state: OrderState) -> OrderState:
    p = state["intent_payload"]
    print(f"[Node: create_item] {p}")
    api_response = create_menu_item(
        name=p.get("name", "Unnamed"), category=p.get("category", "main course"),
        price=float(p.get("price", 0)), description=p.get("description", ""),
        available=p.get("available", True),
    )
    if "error" in api_response:
        return {**state, "api_response": api_response, "status": "error",
                "voice_reply": f"Sorry, I couldn't add that item: {api_response['error']}"}
    return {**state, "api_response": api_response, "status": "success",
            "voice_reply": f"Done! '{p.get('name')}' has been added to the menu."}


def node_update_item(state: OrderState) -> OrderState:
    p       = state["intent_payload"]
    item_id = p.get("item_id", -1)
    print(f"[Node: update_item] item_id={item_id}")
    if item_id == -1:
        return {**state, "api_response": {}, "status": "error",
                "voice_reply": "I need the item ID to update it. Which item number?"}
    api_response = update_menu_item(
        item_id=item_id, name=p.get("name", ""), category=p.get("category", "main course"),
        price=float(p.get("price", 0)), description=p.get("description", ""),
        available=p.get("available", True),
    )
    if "error" in api_response:
        return {**state, "api_response": api_response, "status": "error",
                "voice_reply": f"Couldn't update item {item_id}: {api_response['error']}"}
    return {**state, "api_response": api_response, "status": "success",
            "voice_reply": f"Item {item_id} has been updated successfully."}


def node_delete_item(state: OrderState) -> OrderState:
    item_id = state["intent_payload"].get("item_id", -1)
    print(f"[Node: delete_item] item_id={item_id}")
    if item_id == -1:
        return {**state, "api_response": {}, "status": "error",
                "voice_reply": "I need the item ID to delete it. Which item number?"}
    api_response = delete_menu_item(item_id)
    if "error" in api_response:
        return {**state, "api_response": api_response, "status": "error",
                "voice_reply": f"Couldn't delete item {item_id}: {api_response['error']}"}
    return {**state, "api_response": api_response, "status": "success",
            "voice_reply": f"Item {item_id} has been removed from the menu."}


def node_handle_unknown(state: OrderState) -> OrderState:
    return {**state, "api_response": {}, "status": "error",
            "voice_reply": ("Sorry, I didn't understand that. You can place an order, "
                            "view the menu, or manage menu items.")}


def node_end_order(state: OrderState) -> OrderState:
    return {**state, "api_response": {}, "status": "success",
            "voice_reply": "Thank you for your order! Have a great meal. Goodbye!"}


# ── Routers ───────────────────────────────────────────────────────────────────

def route_by_intent(state: OrderState) -> str:
    return {
        "PLACE_ORDER": "place_order_map",
        "GET_MENU":    "get_menu",
        "CREATE_ITEM": "create_item",
        "UPDATE_ITEM": "update_item",
        "DELETE_ITEM": "delete_item",
        "END_ORDER":   "end_order",
    }.get(state["intent"], "handle_unknown")


def route_after_mapping(state: OrderState) -> str:
    mapping = state["mapping_result"]
    if mapping.get("needs_clarification"):
        return "ask_clarification"
    if mapping.get("is_valid"):
        return "place_order_exec"
    return "handle_unknown"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_order_graph():
    g = StateGraph(OrderState)

    g.add_node("load_menu",          node_load_menu)
    g.add_node("classify_intent",    node_classify_intent)
    g.add_node("place_order_map",    node_place_order_map)
    g.add_node("place_order_exec",   node_place_order_exec)
    g.add_node("ask_clarification",  node_ask_clarification)
    g.add_node("get_menu",           node_get_menu)
    g.add_node("create_item",        node_create_item)
    g.add_node("update_item",        node_update_item)
    g.add_node("delete_item",        node_delete_item)
    g.add_node("handle_unknown",     node_handle_unknown)
    g.add_node("end_order",          node_end_order)

    g.set_entry_point("load_menu")
    g.add_edge("load_menu", "classify_intent")

    g.add_conditional_edges("classify_intent", route_by_intent, {
        "place_order_map": "place_order_map",
        "get_menu":        "get_menu",
        "create_item":     "create_item",
        "update_item":     "update_item",
        "delete_item":     "delete_item",
        "handle_unknown":  "handle_unknown",
        "end_order":       "end_order",
    })

    g.add_conditional_edges("place_order_map", route_after_mapping, {
        "place_order_exec":  "place_order_exec",
        "ask_clarification": "ask_clarification",
        "handle_unknown":    "handle_unknown",
    })

    for terminal in ("place_order_exec", "ask_clarification", "get_menu",
                     "create_item", "update_item", "delete_item",
                     "handle_unknown", "end_order"):
        g.add_edge(terminal, END)

    return g.compile()


order_graph = build_order_graph()


# ── Public entry-point ────────────────────────────────────────────────────────

def process_order(
    raw_input:   str,
    customer_id: Optional[str] = None,
    history:     list          = None,
) -> dict:
    final = order_graph.invoke({
        "raw_input":      raw_input,
        "customer_id":    customer_id,
        "history":        history or [],
        "menu":           {},
        "intent":         "",
        "intent_payload": {},
        "mapping_result": {},
        "api_response":   {},
        "voice_reply":    "",
        "status":         "pending",
    })
    return {
        "status":      final["status"],
        "voice_reply": final["voice_reply"],
        "order_id":    final.get("api_response", {}).get("order_id"),
        "order_total": final.get("api_response", {}).get("order_total"),
    }   