"""
Mapping Agent — agent/mapping_agent.py
"""

import json
import os
from typing import Any
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_SYSTEM_PROMPT_TEMPLATE = """\
You are an intelligent order mapping system for KFC.
Map the customer's raw order to exact menu SKUs.

Menu (JSON):
MENU_JSON_PLACEHOLDER

RULES:
1. VALIDATION — Flag anything not on the menu as invalid. Do NOT guess or silently substitute.

2. CLARIFICATION — If the customer's request is ambiguous and could match multiple items,
   you MUST ask for clarification instead of guessing. Set needs_clarification to true
   and provide a clear spoken question in clarification_question.
   
   Ambiguous cases you MUST clarify:
   - "drink" or "something to drink" → multiple drinks exist (Pepsi, Mountain Dew, 7UP, sizes)
     Ask: "Sure! We have Pepsi, Mountain Dew, and 7UP. Which would you like, and would you like large or regular?"
   - "chicken" or "3 piece chicken" → multiple chicken items exist (Original Recipe, Extra Crispy)
     Ask: "We have Original Recipe Chicken and Extra Crispy Chicken. Which would you prefer?"
   - Any item name that partially matches 2+ menu items with meaningfully different options.
   - Size is missing when both regular and large exist for that item.

3. DEAL MAPPING — If the customer names a deal (e.g. "mighty combo"), map to the deal ID directly.
   If individual items together form a deal, combine them into the deal and note savings.

4. ALIASES — Resolve informal names using context:
   - "mighty combo" → map to the Mighty combo deal item directly
   - "zinger" → Zinger Burger
   - "wings" → Hot Wings

5. QUANTITY — Respect quantities ("2 zingers" → qty: 2).

6. PRICING — Always populate unit_price and total_price from the menu.

Return ONLY valid JSON, no markdown. Use ONE of these two schemas:

If clarification is needed:
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

If order is clear and valid:
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

If item is invalid (not on menu at all):
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


def call_mapping_llm(raw_order: str | list, menu: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_order, list):
        raw_order = ", ".join(str(i) for i in raw_order)

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.replace(
        "MENU_JSON_PLACEHOLDER", json.dumps(menu, indent=2)
    )

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Customer order: {raw_order}"},
        ],
    )
    return json.loads(resp.choices[0].message.content)