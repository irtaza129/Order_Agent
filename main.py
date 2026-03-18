"""
main.py — project root
Run from Order_Agent/: python main.py
"""

import os
import sys

from agent.langgraph_agent import process_order

if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY is not set.\n")

MAX_HISTORY = 4  # number of turns (1 turn = 1 user msg + 1 agent reply) to keep


def _trim_history(history: list) -> list:
    """Keep only the last MAX_HISTORY turns (each turn = 2 entries)."""
    return history[-(MAX_HISTORY * 2):]


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

    history = []   # list of {"role": "user"|"assistant", "content": str}

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

        result = process_order(
            raw_input=raw,
            customer_id="cli_user",
            history=_trim_history(history),
        )
        print(f"\nAgent: {result['voice_reply']}")

        # Clarification loop — agent asks a follow-up question
        while result["status"] == "clarification":
            # Log the clarification exchange into history
            history.append({"role": "user",      "content": raw})
            history.append({"role": "assistant",  "content": result["voice_reply"]})
            history = _trim_history(history)

            try:
                followup = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not followup:
                continue

            raw = followup   # update raw so history is logged correctly below
            clarification_context = (
                f"Agent asked: {result['voice_reply']} "
                f"Customer answered: {followup}"
            )
            result = process_order(
                raw_input=clarification_context,
                customer_id="cli_user",
                history=_trim_history(history),
            )
            print(f"\nAgent: {result['voice_reply']}")

        # Log this turn into history
        history.append({"role": "user",      "content": raw})
        history.append({"role": "assistant",  "content": result["voice_reply"]})
        history = _trim_history(history)

        _print_result(result)


def run_demo() -> None:
    cases = [
        "I want 1 original recipe chicken",
        "add one 7up large and repeat my order",   # tests history
        "Show me the menu",
        "thats all thank you",
    ]
    print("=" * 60)
    print("  KFC Agent — Demo Mode")
    print("=" * 60 + "\n")

    history = []
    for case in cases:
        print(f"Customer : {case}")
        result = process_order(raw_input=case, history=_trim_history(history))
        print(f"Agent    : {result['voice_reply']}")
        history.append({"role": "user",     "content": case})
        history.append({"role": "assistant", "content": result["voice_reply"]})
        history = _trim_history(history)
        _print_result(result)
        print("-" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo()
    else:
        run_cli()