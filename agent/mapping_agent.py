"""
Mapping Agent — agent/mapping_agent.py
Uses OpenAI Responses API with gpt-5-mini.
"""

import json
import os
from typing import Any
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_INSTRUCTIONS = """\
You are an intelligent order mapping system for KFC.
Map the customer's raw order to exact menu SKUs.

RULES:
1. VALIDATION — Flag anything not on the menu as invalid. Do NOT guess or silently substitute.

2. CLARIFICATION — If the request is ambiguous and could match multiple items,
   set needs_clarification to true and ask a clear spoken question.
   Cases that MUST trigger clarification:
   - "drink" / "something to drink" → multiple drinks exist (Pepsi, Mountain Dew, 7UP)
     Ask: "Sure! We have Pepsi, Mountain Dew, and 7UP. Which would you like?"
   - "chicken" / "3 piece chicken" → Original Recipe vs Extra Crispy exist
     Ask: "We have Original Recipe Chicken and Extra Crispy Chicken. Which would you prefer?"
   - Size is missing when both regular and large exist for that item.

3. HISTORY — Use the conversation history to resolve references like:
   - "repeat my order" → look at previous Agent replies to find what was ordered, add same items
   - "make that two" → double the quantity of the last ordered item
   - "add the same again" → duplicate the previous order items

4. DEAL MAPPING — If the customer names a deal (e.g. "mighty combo"), map to it directly.
   If individual items form a deal, combine them and note savings.

5. ALIASES — Resolve informal names: "zinger" → Zinger Burger, "wings" → Hot Wings.

6. QUANTITY — Respect quantities ("2 zingers" → qty: 2).

7. PRICING — Always populate unit_price and total_price from the menu.

Return ONLY valid JSON, no markdown. Use ONE of these schemas:

Clarification needed:
{
  "is_valid": false,
  "needs_clarification": true,
  "clarification_question": "Spoken question to ask the customer",
  "invalid_reason": "",
  "invalid_items": [],
  "final_cart": [],
  "order_total": 0,
  "savings": 0
}

Valid order:
{
  "is_valid": true,
  "needs_clarification": false,
  "clarification_question": "",
  "invalid_reason": "",
  "invalid_items": [],
  "final_cart": [
    {"item_id": "1", "name": "Zinger Burger", "qty": 1, "unit_price": 7.99, "total_price": 7.99}
  ],
  "order_total": 7.99,
  "savings": 0
}

Invalid item:
{
  "is_valid": false,
  "needs_clarification": false,
  "clarification_question": "",
  "invalid_reason": "We don't have beef burgers. We have Zinger, Tower, and Fillet burgers.",
  "invalid_items": ["beef burger"],
  "final_cart": [],
  "order_total": 0,
  "savings": 0
}
"""


def _history_block(history: list) -> str:
    if not history:
        return ""
    lines = ["[Conversation so far]"]
    for msg in history:
        role = "User" if msg["role"] == "user" else "Agent"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) + "\n\n"


def call_mapping_llm(
    raw_order: str | list,
    menu:      dict[str, Any],
    history:   list = None,
) -> dict[str, Any]:
    if isinstance(raw_order, list):
        raw_order = ", ".join(str(i) for i in raw_order)

    instructions_with_menu = (
        _INSTRUCTIONS
        + f"\n\nCurrent menu (JSON):\n{json.dumps(menu, indent=2)}"
    )

    # History + current order as input
    input_text = (
        _history_block(history or [])
        + f"Map this customer order to json: {raw_order}"
    )

    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        instructions=instructions_with_menu,
        input=input_text,
        text={"format": {"type": "json_object"}},
    )

    return json.loads(response.output_text)