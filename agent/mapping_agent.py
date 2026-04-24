"""
Mapping Agent V3 — agent/mapping_agent.py
Context-aware semantic matching for Pakistani ordering patterns.

Key features:
- Normalizes Roman Urdu to English BEFORE semantic matching
- Understands implicit orders: "ek plate" → pulao context
- Context-aware disambiguation: "leg piece" after pulao → pulao addon vs standalone
- Smart clarification when semantic scores are close
"""

import json
import os
import re
from typing import Any, Optional, List, Tuple
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

# ─────────────────────────────────────────────────────────────────
# CLIENT INIT
# ─────────────────────────────────────────────────────────────────
_client = None
_embedder = None

def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing in environment")
        _client = OpenAI(api_key=api_key)
    return _client


def get_embedder():
    """Load sentence transformer for semantic matching (50MB, one-time)"""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder


# Logger
log = logging.getLogger("mapping_agent")


# ─────────────────────────────────────────────────────────────────
# ROMAN URDU DETECTION (expanded)
# ─────────────────────────────────────────────────────────────────
_ROMAN_URDU_WORDS = frozenset([
    # Greetings
    "assalam","aoa","salam","salaam","kaise","kya","haal","theek","achi",
    
    # Requests & actions
    "dena","dedo","de","chahiye","lena","dijiye","krdo","kardo","krdain",
    "karein","lagao","laga","mujhe","mujhey","humein","hamein","humko",
    "hamko","aur","bhi","waisa","wesa","vesa",
    
    # Negations & confirmations
    "nahi","nai","nahin","bilkul","zaroor","han","haan","theek","thik",
    
    # Numbers (omit "do"=2 — conflicts with English "do")
    "ek","teen","char","panch","chhe","saat","aath","nau","das",
    "gyara","bara","tera","choda","chaudah","pandara","sola","satra","athara","unnis","bees",
    
    # Food general
    "khana","peena","pina","garam","thanda","taza","fresh","plate","plait",
    
    # Savour Foods specific
    "pulao","pulaw","pulav","biryani","chana","channa","halwa","halua","puri","poori",
    "paratha","pratha","naan","nan","roti","anda","omlet","omelette","nashta",
    "breakfast","kabab","kebab","kabob","shami","shaami","raita","raaita",
    "salad","kheer","keer","zarda","zardaa","falooda","faluda","lassi","lasi",
    
    # Chicken parts (CRITICAL for context)
    "piece","pis","leg","breast","tangri","drumstick","thigh","wing","boti",
    
    # Common items
    "burger","burgar","zinger","zingar","krispo","crispo","chicken","chikan",
    "wings","fries","frize","nuggets","nugget","drink","dring",
    
    # Beverages
    "pepsi","pesi","7up","sevenup","mirinda","mirenda","chai","tea","coffee",
    "doodh","milk","pani","water",
    
    # Endings
    "bas","shukriya","shukria","meherbani","alvida","bye","khuda","hafiz",
    
    # State words
    "ho","gaya","gai","gya","hua","wahi","wahe","dobara","phir","repeat","same",
    
    # Queries
    "kuch","kitna","kitni","kaisay","kese","total","bill","menu","dikhao",
    "dikha","batao","bata","hai","hain","he","main","mei","me",
])


def detect_roman_urdu(text: str) -> bool:
    """Word-level Urdu detection (no LLM, 1ms latency)"""
    words = text.lower().split()
    res = any(w in _ROMAN_URDU_WORDS for w in words)
    log.debug("detect_roman_urdu: text=%s -> %s", text, res)
    return res


# ─────────────────────────────────────────────────────────────────
# FAST NORMALIZER (Roman Urdu → English for semantic matching)
# ─────────────────────────────────────────────────────────────────
_FAST_NORMALIZER_PROMPT = """\
Translate ONLY Roman Urdu/Hinglish to plain English for a food ordering system.
Keep English words as-is. Return ONLY the translation, no JSON wrapper.

Examples:
"ek plate kardo" → "one plate"
"do pulao aur chana" → "two pulao and chana"
"leg piece dedo" → "leg piece"
"mujhe zinger burger chahiye" → "I want zinger burger"
"bas itna" → "that's all"

Keep food names in English. Just translate grammar/structure.
"""


def fast_normalize(text: str) -> str:
    """
    Quick normalization for semantic matching.
    Only translates if Roman Urdu detected.
    Returns English text for embedding.
    """
    if not detect_roman_urdu(text):
        return text
    
    try:
        log.info("fast_normalize: normalizing text='%s'", text)
        client = get_client()
        resp = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            instructions=_FAST_NORMALIZER_PROMPT,
            input=text,
            max_tokens=100,
        )
        
        normalized = resp.output_text.strip() if hasattr(resp, "output_text") else text
        # Remove quotes if LLM added them
        normalized = normalized.strip('"\'')
        log.debug("fast_normalize: normalized='%s'", normalized)
        return normalized
        
    except Exception as e:
        print(f"[FastNormalizer] failed ({e}) — using original")
        return text


# ─────────────────────────────────────────────────────────────────
# FULL NORMALIZER (for final LLM call - with JSON)
# ─────────────────────────────────────────────────────────────────
_NORMALIZER_PROMPT = """\
You translate Roman Urdu to English for a Savour Foods ordering system.
Savour Foods serves: pulao, breakfast items (halwa puri, anda paratha), 
burgers, fried chicken, deals, beverages, and traditional desserts.

Return ONLY valid JSON, no markdown.
Schema: {"normalized_text": "<english>"}

Examples:
"mujhe ek pulao aur do chana plate chahiye" → {"normalized_text": "I want one pulao and two chana plates"}
"zinger burger dedo aur fries bhi" → {"normalized_text": "Give me zinger burger and also fries"}
"ek plate kardo" → {"normalized_text": "Give me one plate"}
"leg piece dedo" → {"normalized_text": "Give me leg piece"}
"""


def normalize_if_urdu(text: str) -> tuple[str, bool]:
    """Full normalization with JSON response for final mapping. Results are cached."""
    if not detect_roman_urdu(text):
        return text, False

    cache_key = text.strip().lower()
    if cache_key in _NORMALIZE_CACHE:
        log.debug("normalize_if_urdu: cache hit for '%s'", cache_key)
        return _NORMALIZE_CACHE[cache_key]

    try:
        client = get_client()
        resp = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            instructions=_NORMALIZER_PROMPT,
            input=f"Return JSON translation for: {text}",
            text={"format": {"type": "json_object"}},
        )

        raw = resp.output_text if hasattr(resp, "output_text") else str(resp)
        normalized = json.loads(raw).get("normalized_text", text)
        result = (normalized, True)
        if len(_NORMALIZE_CACHE) < _CACHE_MAX:
            _NORMALIZE_CACHE[cache_key] = result
        return result

    except Exception as e:
        log.warning("normalize_if_urdu: failed (%s) — using raw text", e)
        return text, False


# ─────────────────────────────────────────────────────────────────
# STT PHONETIC CORRECTIONS
# ─────────────────────────────────────────────────────────────────
_PHONETIC_MAP = {
    # Common STT errors
    "gurber": "burger", "burgar": "burger",
    "zenger": "zinger", "zingar": "zinger",
    "frize": "fries", "frise": "fries",
    "nugget": "nuggets",
    "chiken": "chicken", "checken": "chicken", "chikan": "chicken",
    
    # Drinks
    "pesi": "pepsi", "7 up": "7up", "sevenup": "7up",
    "mirenda": "mirinda",
    
    # Savour Foods specific
    "pulaw": "pulao", "pulau": "pulao", "pulav": "pulao",
    "channa": "chana", "channay": "chana",
    "halua": "halwa",
    "poori": "puri", "puree": "puri",
    "pratha": "paratha",
    "kabob": "kabab", "kebab": "kabab",
    "shaami": "shami",
    "raaita": "raita",
    "keer": "kheer",
    "zardaa": "zarda",
    "faluda": "falooda",
    "lasi": "lassi",
    "omlet": "omelette", "omlete": "omelette",
    "crispo": "krispo", "chryspo": "krispo",
    
    # Chicken parts
    "pis": "piece", "peice": "piece",
    "leg": "leg piece", "tangri": "leg piece",
}


def apply_phonetic_correction(text: str) -> str:
    """Fix common STT errors"""
    text_lower = text.lower()
    for wrong, correct in _PHONETIC_MAP.items():
        text_lower = re.sub(r'\b' + wrong + r'\b', correct, text_lower)
    return text_lower


# ─────────────────────────────────────────────────────────────────
# QUANTITY EXTRACTION
# ─────────────────────────────────────────────────────────────────
_URDU_NUMBERS = {
    "ek": 1, "do": 2, "teen": 3, "char": 4, "panch": 5,
    "chhe": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10,
    "gyara": 11, "bara": 12, "tera": 13, "choda": 14, "chaudah": 14,
    "pandara": 15, "sola": 16, "satra": 17, "athara": 18, "unnis": 19, "bees": 20,
}


def extract_quantity(text: str) -> tuple[int, str]:
    """Extract quantity from text"""
    text = text.lower().strip()
    
    # Check for Urdu numbers
    for urdu_num, value in _URDU_NUMBERS.items():
        if text.startswith(urdu_num + " "):
            return value, text[len(urdu_num):].strip()
    
    # Check for English numbers
    match = re.match(r'^(\d+)\s+(.+)$', text)
    if match:
        return int(match.group(1)), match.group(2)
    
    # Check for words
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    for word, value in word_to_num.items():
        if text.startswith(word + " "):
            return value, text[len(word):].strip()
    
    return 1, text


# ─────────────────────────────────────────────────────────────────
# IN-MEMORY CACHES
# ─────────────────────────────────────────────────────────────────
_NORMALIZE_CACHE: dict[str, tuple[str, bool]] = {}
_QUERY_EMB_CACHE: dict[str, "np.ndarray"] = {}
_CACHE_MAX = 1024

# ─────────────────────────────────────────────────────────────────
# CONTEXT-AWARE SEMANTIC MATCHING
# ─────────────────────────────────────────────────────────────────
_ITEM_EMBEDDINGS_CACHE = None


def invalidate_embeddings_cache():
    global _ITEM_EMBEDDINGS_CACHE
    _ITEM_EMBEDDINGS_CACHE = None


def build_item_embeddings(menu: dict) -> dict:
    """Pre-compute embeddings for all menu items"""
    global _ITEM_EMBEDDINGS_CACHE
    
    if _ITEM_EMBEDDINGS_CACHE is not None:
        return _ITEM_EMBEDDINGS_CACHE
    
    embedder = get_embedder()
    items = menu.get("single_items", [])
    
    embeddings = {}
    for item in items:
        # Create searchable text: name + category + description
        search_text = f"{item['name']} {item['category']} {item['description']}"
        embeddings[item['id']] = {
            'embedding': embedder.encode(search_text.lower()),
            'item': item
        }
    
    _ITEM_EMBEDDINGS_CACHE = embeddings
    return embeddings


def find_similar_items(
    query: str,
    menu: dict,
    context: Optional[dict] = None,
    top_k: int = 5,
    threshold: float = 0.4,
    pre_normalized: bool = False,
) -> List[Tuple[float, dict]]:
    """
    Find menu items semantically similar to query.

    Args:
        query: Search text (will be normalized if Roman Urdu, unless pre_normalized=True)
        menu: Full menu dict
        context: Optional context dict with 'history', 'pending_cart', 'last_category'
        top_k: Return top K matches
        threshold: Minimum similarity score
        pre_normalized: Skip LLM normalization (query already in English)

    Returns:
        List of (score, item) tuples sorted by score descending
    """
    normalized_query = query if pre_normalized else fast_normalize(query)

    embedder = get_embedder()
    item_embeddings = build_item_embeddings(menu)

    emb_key = normalized_query.lower().strip()
    if emb_key in _QUERY_EMB_CACHE:
        query_emb = _QUERY_EMB_CACHE[emb_key]
    else:
        query_emb = embedder.encode(emb_key)
        if len(_QUERY_EMB_CACHE) < _CACHE_MAX:
            _QUERY_EMB_CACHE[emb_key] = query_emb
    
    scores = []
    for item_id, data in item_embeddings.items():
        item = data['item']
        
        # Base similarity score
        score = np.dot(data['embedding'], query_emb) / (
            np.linalg.norm(data['embedding']) * np.linalg.norm(query_emb)
        )

        # Keyword boost: reward direct name-word overlap with query
        query_words = set(normalized_query.lower().split())
        item_name_words = set(item['name'].lower().split())
        overlap = query_words & item_name_words
        if overlap:
            score += 0.08 * len(overlap)

        # ──────────────────────────────────────────────────────────
        # CONTEXT BOOSTING (Pakistani ordering patterns)
        # ──────────────────────────────────────────────────────────
        if context:
            # Pattern 1: "ek plate" after ordering pulao → boost pulao items
            if "plate" in normalized_query.lower() and len(normalized_query.split()) <= 3:
                if item['category'] == 'Pulao':
                    score += 0.3  # Strong boost
                    
            # Pattern 2: "leg piece" / "chicken piece" disambiguation
            if any(word in normalized_query.lower() for word in ["piece", "leg", "breast", "tangri"]):
                history = context.get('history', [])
                pending_cart = context.get('pending_cart', [])
                
                # Check if user has pulao in cart or recent history
                has_pulao_context = False
                
                if pending_cart:
                    has_pulao_context = any(
                        item_dict.get('category') == 'Pulao' 
                        for item_dict in pending_cart
                    )
                
                if not has_pulao_context and history:
                    # Check last 3 messages for pulao mention
                    recent = ' '.join([msg.get('content', '') for msg in history[-3:]]).lower()
                    has_pulao_context = 'pulao' in recent or 'plate' in recent
                
                # Boost accordingly
                if has_pulao_context:
                    # User likely wants pulao addon
                    if 'Chicken Piece Addon' in item['name'] or item['category'] == 'Pulao':
                        score += 0.25
                else:
                    # User likely wants standalone fried chicken
                    if item['category'] == 'Fried' and 'Krispo' in item['name']:
                        score += 0.25
            
            # Pattern 3: Category continuity
            last_category = context.get('last_category')
            if last_category and item['category'] == last_category:
                score += 0.1  # Small boost for same category
        
        if score >= threshold:
            scores.append((score, item))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    return scores[:top_k]


# Keywords that uniquely identify a pulao variant — no need to ask which one
_EXPLICIT_PULAO_VARIANTS = frozenset([
    "special", "special choice",
    "single choice",
    "without kabab",
    "plain pulao", "plain",
    "pulao kabab",
])


def should_ask_clarification(matches, query: str, score_threshold: float = 0.15) -> bool:
    if not matches or len(matches) < 2:
        return False

    q = query.lower()

    # User named a specific variant — let the LLM map it, no clarification needed
    if any(v in q for v in _EXPLICIT_PULAO_VARIANTS):
        return False

    top_score = matches[0][0]
    second_score = matches[1][0]

    # ─────────────────────────────
    # HARD RULE: implicit "plate" with no variant specified
    # ─────────────────────────────
    is_plate_query = "plate" in q or "ek plate" in q

    if is_plate_query:
        has_pulao = any("Pulao" in item["category"] for _, item in matches)
        if has_pulao:
            return True

    # ─────────────────────────────
    # SOFT AMBIGUITY RULE: close scores across different categories
    # ─────────────────────────────
    if (top_score - second_score) < score_threshold:
        top_cat = matches[0][1].get("category", "")
        second_cat = matches[1][1].get("category", "")
        # Only ask when items are from genuinely different categories
        return top_cat != second_cat

    return False


# ─────────────────────────────────────────────────────────────────
# MAPPING PROMPT (Savour Foods optimized)
# ─────────────────────────────────────────────────────────────────
_MAPPING_INSTRUCTIONS = """\
You are a food order mapper for Savour Foods Pakistan.
Savour Foods specializes in: Traditional Pakistani pulao, breakfast items 
(halwa puri, anda paratha, chana), burgers (Zinger, Krispo), fried chicken, 
deals, beverages, and desserts (kheer, zarda, falooda).

Map the customer's order to exact menu SKUs.

CRITICAL TTS RULE:
All text will be spoken by TTS. Use ONLY plain spoken English or Roman Urdu.
- NO emojis, bullets, dashes, special characters
- Natural conversational tone

STT CORRECTIONS (fix silently):
- "gurber"→burger, "zenger"→zinger, "pulaw"→pulao
- "channa"→chana, "halua"→halwa, "omlet"→omelette
- "pesi"→pepsi, "crispo"→krispo
- Match phonetically. Mark INVALID only if no plausible match exists.

PAKISTANI ORDERING PATTERNS (CRITICAL):

1. "ek plate" / "do plate" context:
   - If user says just "ek plate kardo", they mean pulao
   - ASK which pulao variant: Single (PKR 645), Single Choice (PKR 665), 
     Special Choice (PKR 845), or plain Pulao (PKR 335)
   - Clarification: "Which pulao would you like? Single, Single Choice, Special Choice, or plain Pulao?"

2. "leg piece" / "chicken piece" disambiguation:
   - IF user has pulao in cart OR mentioned pulao in last 2 turns:
     → They want "Chicken Piece Addon" (PKR 205) for their pulao
   - IF no pulao context:
     → They want standalone "Krispo Chicken Piece" (PKR 299)
   - ASK only if genuinely unclear: "Do you want this as an addon to your pulao 
     or as a separate fried chicken piece?"

3. Implicit additions:
   - "aur ek" / "one more" → repeat last item
   - "wahi" / "same" → repeat last item

SAVOUR FOODS MENU KNOWLEDGE:
- Pulao variants:
  * Single (PKR 645): 1 chicken + 2 kababs + salad + raita
  * Single Choice (PKR 665): Selected chicken + kababs + salad + raita
  * Single Without Kabab (PKR 585): 1 chicken + salad + raita, no kabab
  * Special Choice (PKR 845): 2 chickens + 2 kababs + salad + raita
  * Plain Pulao (PKR 335): Just rice + salad + raita
  * Pulao Kabab (PKR 445): Plain pulao + 2 kababs

- Breakfast: Halwa Puri (PKR 420), Anda Paratha (PKR 260), Omelette Toast (PKR 295)
- Burgers: Zinger (PKR 650), Krispo (PKR 610), Chicken (PKR 510)
- Fried: Krispo Wings, Nuggets, Fries
- Deals: Family deals, student deals, lunch/dinner combos

DEAL SUGGESTIONS:
Check if adding one item unlocks a deal that saves money.
Example: "If you add a drink, you get our lunch deal for PKR 745 instead of PKR 785."

CLARIFICATION RULES:
Ask ONLY when:
1. "ek plate" without specifying which pulao variant
2. "soft drink" without size/flavor
3. Ambiguous chicken piece order (see rule 2 above)
4. Semantic scores very close between different categories

DO NOT ask if context is clear from history or cart.

OUTPUT SCHEMA:
{
  "status": "VALID" | "NEEDS_CLARIFICATION" | "INVALID",
  "final_cart": [
    {
      "item_id": "1",
      "name": "Single",
      "qty": 1,
      "unit_price": 645,
      "total_price": 645
    }
  ],
  "clarification_question": "Which pulao would you like? Single, Single Choice, Special Choice, or plain Pulao?",
  "reply": "Added Single pulao to your cart. Anything else?",
  "deal_suggestion": ""
}
DO NOT repeat menu items in response.
DO NOT output full menu.
Only use provided candidates.
Return ONLY valid JSON.
CRITICAL: NEVER output the full menu or repeat menu items in your reply. Only return `final_cart` and `reply` as specified in the schema.
"""


def _history_block(history: list) -> str:
    """Format conversation history for context"""
    if not history:
        return ""
    lines = ["[Recent conversation]"]
    for msg in history[-6:]:  # Last 3 turns
        role = "Customer" if msg["role"] == "user" else "Agent"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) + "\n\n"


def _build_semantic_context(
    query: str, 
    menu: dict, 
    history: list,
    pending_cart: list,
) -> dict:
    """
    Build semantic match context with BOTH:
    - structured candidates (for LLM reasoning)
    - human-readable hints (for fallback compatibility)
    """

    context = {
        "history": history,
        "pending_cart": pending_cart,
        "last_category": None
    }

    if pending_cart:
        context["last_category"] = pending_cart[-1].get("category")

    matches = find_similar_items(
        query,
        menu,
        context=context,
        top_k=5,
        threshold=0.2,
        pre_normalized=True,
    )

    if not matches:
      return {
        "structured": [],
        "matches": [],
        "needs_clarification": False,
        "debug_text": ""
    }

    # ─────────────────────────────────────────────
    # 1. STRUCTURED CANDIDATES (NEW - IMPORTANT)
    # ─────────────────────────────────────────────
    structured = []
    for score, item in matches:
        structured.append({
            "id": item.get("id"),
            "name": item.get("name"),
            "category": item.get("category"),
            "price": item.get("price"),
            "score": round(float(score), 4)
        })

    # ─────────────────────────────────────────────
    # 2. HUMAN READABLE CONTEXT (OLD BEHAVIOR)
    # ─────────────────────────────────────────────
    lines = ["[SEMANTIC_CONTEXT]"]

    lines.append("\nCANDIDATES_JSON:")
    lines.append(json.dumps(structured, ensure_ascii=False))

    lines.append("\nREADABLE_MATCHES:")
    for score, item in matches:
        lines.append(
            f"- {item['name']} (PKR {item['price']}) "
            f"[{item['category']}] score={score:.2f}"
        )

    # ─────────────────────────────────────────────
    # 3. DISAMBIGUATION SIGNAL
    # ─────────────────────────────────────────────
    if should_ask_clarification(matches,query, score_threshold=0.15):
        lines.append(
            "\nCLARIFICATION_FLAG: true "
            "(multiple similar items detected)"
        )
    else:
        lines.append("\nCLARIFICATION_FLAG: false")

    return {
    "structured": structured,
    "matches": matches,
    "needs_clarification": should_ask_clarification(matches,query, score_threshold=0.15),
    "debug_text": "\n".join(lines)  # optional fallback for LLM visibility
}

def _fix_item_ids(result: dict, menu: dict) -> dict:
    """Ensure all items have correct item_id from menu"""
    items = menu.get("single_items", [])
    name_to_item = {i["name"].lower(): i for i in items}
    
    for cart_item in result.get("final_cart", []):
        if not cart_item.get("item_id") or cart_item["item_id"] == "":
            name = cart_item.get("name", "").lower()
            
            # Exact match
            if name in name_to_item:
                menu_item = name_to_item[name]
                cart_item["item_id"] = str(menu_item["id"])
                cart_item["unit_price"] = menu_item["price"]
                cart_item["total_price"] = menu_item["price"] * cart_item.get("qty", 1)
            else:
                # Fuzzy match
                for menu_name, menu_item in name_to_item.items():
                    if name in menu_name or menu_name in name:
                        cart_item["item_id"] = str(menu_item["id"])
                        cart_item["unit_price"] = menu_item["price"]
                        cart_item["total_price"] = menu_item["price"] * cart_item.get("qty", 1)
                        break
                
                # Last resort: semantic match
                if not cart_item.get("item_id"):
                    similar = find_similar_items(name, menu, top_k=1, threshold=0.5)
                    if similar:
                        score, best_match = similar[0]
                        cart_item["item_id"] = str(best_match["id"])
                        cart_item["name"] = best_match["name"]
                        cart_item["unit_price"] = best_match["price"]
                        cart_item["total_price"] = best_match["price"] * cart_item.get("qty", 1)
    
    return result


# ─────────────────────────────────────────────────────────────────
# MAIN MAPPING FUNCTION
# ─────────────────────────────────────────────────────────────────
def call_mapping_llm(
    raw_order: str | list,
    menu: dict[str, Any],
    history: list = None,
    reply_language: str = "en",
) -> dict[str, Any]:

    log.info(
        "call_mapping_llm: entry raw_order=%s reply_language=%s history_len=%s menu_items=%s",
        raw_order,
        reply_language,
        len(history or []),
        len(menu.get("single_items", []))
    )

    # ─────────────────────────────
    # Normalize input type
    # ─────────────────────────────
    if isinstance(raw_order, list):
        raw_order = ", ".join(str(i) for i in raw_order)

    # ─────────────────────────────
    # Pending cart (context)
    # ─────────────────────────────
    pending_cart = []

    if history:
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                if "cart" in msg.get("content", "").lower():
                    break

    # ─────────────────────────────
    # Phonetic correction
    # ─────────────────────────────
    corrected_order = apply_phonetic_correction(raw_order)
    log.debug("corrected_order=%s", corrected_order)

    # ─────────────────────────────
    # SEMANTIC CONTEXT (NEW STRUCTURE)
    # ─────────────────────────────
    semantic_context = _build_semantic_context(
        corrected_order,
        menu,
        history or [],
        pending_cart
    )

    structured = semantic_context.get("structured", [])
    matches = semantic_context.get("matches", [])
    needs_clarification = semantic_context.get("needs_clarification", False)

    log.debug(
        "semantic_context: structured=%s matches=%s needs_clarification=%s",
        structured,
        len(matches),
        needs_clarification
    )

    # ─────────────────────────────
    # 🚨 EARLY EXIT: clarification logic
    # ─────────────────────────────
    if needs_clarification:
        log.info("EARLY EXIT: NEEDS_CLARIFICATION triggered")

        return {
            "raw_input": raw_order,
            "normalized_input": corrected_order,
            "status": "NEEDS_CLARIFICATION",
            "clarification_question": (
                "Which pulao would you like? "
                "Single, Single Choice, Special Choice, or plain Pulao?"
            ),
            "pending_cart": pending_cart,
            "reply_language": reply_language,
            "semantic_candidates": structured
        }

    # ─────────────────────────────
    # Language handling
    # ─────────────────────────────
    lang_note = (
        "\nREPLY LANGUAGE: Write in Roman Urdu (readable by TTS)."
        if reply_language == "roman_urdu"
        else "\nREPLY LANGUAGE: Plain spoken English."
    )

    # ─────────────────────────────
    # Build LLM context (NO MENU SPAM FIX)
    # ─────────────────────────────
    instructions = (
        _MAPPING_INSTRUCTIONS
        + lang_note
        + "\n\n[SEMANTIC_CANDIDATES]\n"
        + json.dumps(structured[:5], indent=2)
    )

    input_text = (
        _history_block(history or [])
        + f"Customer said: {corrected_order}\n"
        + "Map this order to menu items and return json:"
    )

    # ─────────────────────────────
    # LLM CALL
    # ─────────────────────────────
    client = get_client()
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        instructions=instructions,
        input=input_text,
        text={"format": {"type": "json_object"}},
    )

    raw = response.output_text if hasattr(response, "output_text") else str(response)
    result = json.loads(raw)

    log.debug("LLM result=%s", result)

    # ─────────────────────────────
    # Post-processing & sanitization (remove menu echoes)
    # ─────────────────────────────
    # Remove any leaked menu or debug fields
    for k in ("menu", "full_menu", "semantic_debug", "semantic_context", "debug_text"):
        result.pop(k, None)

    # Strip menu-like echoes from the human reply
    def _strip_menu_echo(text: str) -> str:
        if not text:
            return text
        banned = ["menu", "available items", "here are", "single_items", "full menu"]
        low = text.lower()
        if any(b in low for b in banned):
            return "I have added your item to the cart."
        return text

    result["reply"] = _strip_menu_echo(result.get("reply", ""))

    result["reply_language"] = reply_language
    result = _fix_item_ids(result, menu)

    log.info(
        "call_mapping_llm: returning status=%s",
        result.get("status")
    )

    return result

# ─────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────
__all__ = [
    'call_mapping_llm',
    'normalize_if_urdu',
    'detect_roman_urdu',
    'find_similar_items',
    'apply_phonetic_correction',
    'fast_normalize',
    'build_item_embeddings',
    'invalidate_embeddings_cache',
]