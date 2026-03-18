"""
main.py — project root
Run from Order_Agent/: python main.py
"""

import os
import sys

from agent.langgraph_agent import process_order

if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY is not set.\n")


def _print_result(result: dict) -> None:
    print(f"  Status  : {result['status'].upper()}")
    if result.get("order_id"):
        print(f"  Order ID: {result['order_id']}")
    if result.get("order_total"):
        print(f"  Total   : PKR {result['order_total']}")
    print()


def run_cli() -> None:
    print("=" * 60)
    print("  KFC AI Agent  —  Orders & Menu Management")
    print("=" * 60)
    print("  Order  : 'I want 1 mighty combo'")
    print("  Menu   : 'Show me the menu'")
    print("  Add    : 'Add Spicy Wings to menu for 350 PKR'")
    print("  Update : 'Update item 3 — set price to 800'")
    print("  Delete : 'Remove item 5 from the menu'")
    print("  Type 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAgent: Goodbye!")
            break
        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print("Agent: Goodbye!")
            break

        result = process_order(raw_input=raw, customer_id="cli_user")
        print(f"\nAgent: {result['voice_reply']}")

        # If agent needs clarification, collect answer and re-submit as a
        # combined message so the mapping agent has full context
        while result["status"] == "clarification":
            try:
                followup = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not followup:
                continue
            # Re-submit original request + clarification answer together
            clarification_context = (
                f"Agent asked: {result['voice_reply']} "
                f"Customer answered: {followup}"
            )
# → "Agent asked: It seems like you meant the Mighty combo. Would you like to order that?
#    Customer answered: yes"
# mapping LLM now has full context — knows exactly what was being confirmed
            result   = process_order(raw_input=clarification_context, customer_id="cli_user")
            print(f"\nAgent: {result['voice_reply']}")

        _print_result(result)


def run_demo() -> None:
    cases = [
        "I want 1 mighty combo",
        "Give me a drink",                   # should ask: Pepsi, MD, or 7UP?
        "I want 3 piece chicken",            # should ask: Original or Extra Crispy?
        "I want a beef burger",              # invalid
        "Show me the menu",
        "Add Spicy Wings to the menu for 350 PKR",
        "Update item 3 — change the price to 800 PKR",
        "Delete item 5 from the menu",
        "thats all thank you",
    ]
    print("=" * 60)
    print("  KFC Agent — Demo Mode")
    print("=" * 60 + "\n")
    for case in cases:
        print(f"Customer : {case}")
        result = process_order(raw_input=case)
        print(f"Agent    : {result['voice_reply']}")
        _print_result(result)
        print("-" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo()
    else:
        run_cli()