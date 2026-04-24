"""
Test Suite for Mapping Agent V3
Tests Pakistani ordering pattern recognition and context-aware disambiguation.
"""

import json
import os
from dotenv import load_dotenv

load_dotenv()

from agent.mapping_agent import (
    call_mapping_llm,
    find_similar_items,
    fast_normalize,
    detect_roman_urdu,
)

# Full Savour Foods menu subset
SAVOUR_MENU = {
    "single_items": [
        # Pulao items
        {"id": 1, "name": "Single", "category": "Pulao", 
         "description": "Chicken pulao rice served with one chicken piece two shami kababs salad and raita", 
         "price": 645, "available": True},
        {"id": 2, "name": "Single Choice", "category": "Pulao",
         "description": "Chicken pulao with selected chicken piece plus kababs salad and raita",
         "price": 665, "available": True},
        {"id": 4, "name": "Special Choice", "category": "Pulao",
         "description": "Large serving pulao with two chicken pieces two kababs salad and raita",
         "price": 845, "available": True},
        {"id": 5, "name": "Pulao", "category": "Pulao",
         "description": "Plain savour style rice platter served with salad and raita",
         "price": 335, "available": True},
        {"id": 7, "name": "Chicken Piece Addon", "category": "Pulao",
         "description": "Extra steamed chicken piece add on for any meal",
         "price": 205, "available": True},
        
        # Fried chicken
        {"id": 24, "name": "Krispo Chicken Piece", "category": "Fried",
         "description": "Single crispy fried chicken piece",
         "price": 299, "available": True},
        {"id": 25, "name": "Krispo Chicken 2 Piece", "category": "Fried",
         "description": "Two crispy fried chicken pieces combo",
         "price": 540, "available": True},
        
        # Burgers
        {"id": 16, "name": "Zinger Burger", "category": "Burgers",
         "description": "Spicy zinger style chicken fillet burger",
         "price": 650, "available": True},
        {"id": 11, "name": "Krispo Burger", "category": "Burgers",
         "description": "Crispy chicken fillet burger with sauce and lettuce",
         "price": 610, "available": True},
        
        # Sides
        {"id": 27, "name": "French Fries Medium", "category": "Fried",
         "description": "Medium portion crispy fries",
         "price": 275, "available": True},
        
        # Beverages
        {"id": 53, "name": "Soft Drink Can Pepsi", "category": "Beverages",
         "description": "Chilled Pepsi can",
         "price": 120, "available": True},
        {"id": 51, "name": "Tea Cup", "category": "Beverages",
         "description": "Fresh hot tea cup",
         "price": 130, "available": True},
        
        # Breakfast
        {"id": 70, "name": "Halwa Puri", "category": "Breakfast",
         "description": "Traditional breakfast with halwa puri and chana",
         "price": 420, "available": True},
    ]
}


def test_fast_normalize():
    print("\n" + "="*70)
    print("TEST 1: Fast Normalization (Roman Urdu → English for Semantic)")
    print("="*70)
    
    test_cases = [
        "ek plate kardo",
        "leg piece dedo",
        "do pulao aur chana",
        "mujhe zinger burger chahiye",
        "one zinger burger",  # Already English
    ]
    
    for text in test_cases:
        normalized = fast_normalize(text)
        is_urdu = detect_roman_urdu(text)
        print(f"{'[UR]' if is_urdu else '[EN]'} '{text}'")
        print(f"     → '{normalized}'")
        print()


def test_context_aware_plate():
    print("\n" + "="*70)
    print("TEST 2: 'ek plate kardo' Context Detection")
    print("="*70)
    
    # Scenario 1: User says "ek plate" with no context
    print("\n[Scenario 1: No context - should ask which pulao]")
    print("User: 'ek plate kardo'")
    
    matches = find_similar_items(
        "ek plate kardo",
        SAVOUR_MENU,
        context={'history': [], 'pending_cart': [], 'last_category': None},
        top_k=5
    )
    
    print("\nSemantic matches:")
    for score, item in matches:
        print(f"  {score:.3f} - {item['name']} [{item['category']}] PKR {item['price']}")
    
    # The system should boost Pulao category items


def test_context_aware_chicken_piece():
    print("\n" + "="*70)
    print("TEST 3: 'leg piece dedo' Context Disambiguation")
    print("="*70)
    
    # Scenario 1: No pulao context - should suggest Krispo Chicken
    print("\n[Scenario 1: No pulao context]")
    print("User: 'leg piece dedo'")
    
    matches_no_context = find_similar_items(
        "leg piece dedo",
        SAVOUR_MENU,
        context={'history': [], 'pending_cart': [], 'last_category': None},
        top_k=5
    )
    
    print("\nSemantic matches (no context):")
    for score, item in matches_no_context:
        print(f"  {score:.3f} - {item['name']} [{item['category']}] PKR {item['price']}")
    
    # Scenario 2: User has pulao in cart - should suggest addon
    print("\n[Scenario 2: Pulao in cart]")
    print("Cart: [Single Pulao]")
    print("User: 'leg piece dedo'")
    
    matches_with_context = find_similar_items(
        "leg piece dedo",
        SAVOUR_MENU,
        context={
            'history': [],
            'pending_cart': [{'name': 'Single', 'category': 'Pulao'}],
            'last_category': 'Pulao'
        },
        top_k=5
    )
    
    print("\nSemantic matches (with pulao context):")
    for score, item in matches_with_context:
        print(f"  {score:.3f} - {item['name']} [{item['category']}] PKR {item['price']}")
    
    # Scenario 3: Pulao mentioned in recent history
    print("\n[Scenario 3: Pulao in recent history]")
    print("History: User ordered pulao 2 turns ago")
    print("User: 'aur ek piece bhi dedo'")
    
    matches_history_context = find_similar_items(
        "aur ek piece bhi dedo",
        SAVOUR_MENU,
        context={
            'history': [
                {'role': 'user', 'content': 'ek pulao chahiye'},
                {'role': 'assistant', 'content': 'Which pulao would you like?'},
                {'role': 'user', 'content': 'Single'},
                {'role': 'assistant', 'content': 'Added Single pulao to cart'},
            ],
            'pending_cart': [{'name': 'Single', 'category': 'Pulao'}],
            'last_category': 'Pulao'
        },
        top_k=5
    )
    
    print("\nSemantic matches (with history context):")
    for score, item in matches_history_context:
        print(f"  {score:.3f} - {item['name']} [{item['category']}] PKR {item['price']}")


def test_full_mapping_scenarios():
    print("\n" + "="*70)
    print("TEST 4: Full Mapping with Pakistani Ordering Patterns")
    print("="*70)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping - OPENAI_API_KEY not set")
        return
    
    scenarios = [
        {
            "name": "Implicit pulao order",
            "order": "ek plate kardo",
            "history": [],
            "expected": "Should ask which pulao variant"
        },
        {
            "name": "Chicken piece without context",
            "order": "leg piece dedo",
            "history": [],
            "expected": "Should map to Krispo Chicken Piece (PKR 299)"
        },
        {
            "name": "Chicken piece with pulao context",
            "order": "ek leg piece bhi dedo",
            "history": [
                {'role': 'user', 'content': 'ek single pulao'},
                {'role': 'assistant', 'content': 'Added Single pulao PKR 645'},
            ],
            "expected": "Should map to Chicken Piece Addon (PKR 205)"
        },
        {
            "name": "Mixed Urdu-English order",
            "order": "do pulao aur ek zinger burger",
            "history": [],
            "expected": "Should ask which pulao variant and map zinger"
        },
        {
            "name": "Continuation order",
            "order": "aur fries bhi",
            "history": [
                {'role': 'user', 'content': 'zinger burger'},
                {'role': 'assistant', 'content': 'Added Zinger Burger PKR 650'},
            ],
            "expected": "Should add fries to cart"
        },
    ]
    
    for scenario in scenarios:
        print(f"\n{'─'*70}")
        print(f"Scenario: {scenario['name']}")
        print(f"Order: '{scenario['order']}'")
        if scenario['history']:
            print(f"History: {len(scenario['history'])} turns")
        print(f"Expected: {scenario['expected']}")
        print()
        
        try:
            result = call_mapping_llm(
                raw_order=scenario['order'],
                menu=SAVOUR_MENU,
                history=scenario['history'],
                reply_language="en"
            )
            
            print(f"Status: {result.get('status')}")
            print(f"Reply: {result.get('reply')}")
            
            if result.get('clarification_question'):
                print(f"❓ Clarification: {result['clarification_question']}")
            
            if result.get('final_cart'):
                print("\n📦 Cart:")
                for item in result['final_cart']:
                    print(f"   - {item['name']} x{item['qty']} @ PKR {item['unit_price']} = PKR {item['total_price']}")
            
            if result.get('deal_suggestion'):
                print(f"\n💡 Deal: {result['deal_suggestion']}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()


def test_semantic_similarity_edge_cases():
    print("\n" + "="*70)
    print("TEST 5: Edge Cases - Close Semantic Scores")
    print("="*70)
    
    edge_cases = [
        ("crispy chicken", "Should match both Krispo Burger and Krispo Chicken"),
        ("rice meal", "Should match pulao items"),
        ("breakfast plate", "Should match Halwa Puri"),
        ("ek drink", "Should ask which drink"),
    ]
    
    for query, expected in edge_cases:
        print(f"\n Query: '{query}'")
        print(f" Expected: {expected}")
        
        matches = find_similar_items(query, SAVOUR_MENU, top_k=3)
        
        if len(matches) >= 2:
            top_score = matches[0][0]
            second_score = matches[1][0]
            score_diff = top_score - second_score
            
            print(f" Top match: {matches[0][1]['name']} (score: {top_score:.3f})")
            print(f" 2nd match: {matches[1][1]['name']} (score: {second_score:.3f})")
            print(f" Score diff: {score_diff:.3f}")
            
            if score_diff < 0.15:
                print(" ⚠️  Ambiguous - should ask for clarification")
            else:
                print(" ✓ Clear match")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MAPPING AGENT V3 - CONTEXT-AWARE TEST SUITE")
    print("="*70)
    
    test_fast_normalize()
    test_context_aware_plate()
    test_context_aware_chicken_piece()
    test_semantic_similarity_edge_cases()
    
    # LLM tests (require API key)
    test_full_mapping_scenarios()
    
    print("\n" + "="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)