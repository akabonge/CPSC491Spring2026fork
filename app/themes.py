import re
from typing import Dict, List, Tuple

THEME_KEYWORDS: Dict[str, List[str]] = {
    "service_friendliness": ["friendly", "kind", "welcoming", "rude", "unfriendly", "helpful", "attentive", "professional"],
    "speed_wait_time": ["wait", "waiting", "took", "slow", "quick", "fast", "minutes", "long time", "40 minutes"],
    "food_quality": ["delicious", "dry", "bland", "flavor", "fresh", "greasy", "burnt", "soggy", "juicy", "crispy"],
    "portion_value": ["portion", "huge", "big", "price", "expensive", "worth", "steep", "overpriced", "reasonable"],
    "cleanliness_atmosphere": ["clean", "dirty", "bright", "music", "vibe", "atmosphere", "decor"],
    "order_accuracy": ["wrong", "missing", "forgot", "incomplete", "not what was ordered"],
    "health_safety": ["food poisoning", "sick", "ill", "safety", "sanitation"],
    "sides_sauces": ["sauce", "ranch", "mac", "coleslaw", "slaw", "fries", "pickles", "onion", "waffles"],
    "spice_level": ["mild", "medium", "hot", "burn", "spicy", "heat", "pepper", "fire"],
}

# Precompile regex patterns once for speed + correctness.
# - For single words: word boundaries
# - For phrases: match phrase with flexible whitespace
_PATTERNS: Dict[str, List[Tuple[str, re.Pattern]]] = {}
for theme, kws in THEME_KEYWORDS.items():
    compiled: List[Tuple[str, re.Pattern]] = []
    for kw in kws:
        kw_norm = kw.strip().lower()
        if " " in kw_norm:
            # Phrase: allow multiple spaces/newlines between words
            parts = [re.escape(p) for p in kw_norm.split()]
            pat = re.compile(r"(?i)\b" + r"\s+".join(parts) + r"\b")
        else:
            pat = re.compile(r"(?i)\b" + re.escape(kw_norm) + r"\b")
        compiled.append((kw_norm, pat))
    _PATTERNS[theme] = compiled

def detect_themes(text: str) -> List[str]:
    """
    Returns list of theme keys matched in text (deduped, stable order by THEME_KEYWORDS).
    """
    themes, _ = detect_themes_with_hits(text)
    return themes

def detect_themes_with_hits(text: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Returns (themes, theme_hits) where theme_hits maps theme -> matched keywords/phrases.
    This makes results auditable + great for eval.
    """
    t = (text or "").lower()
    out: List[str] = []
    hits: Dict[str, List[str]] = {}

    for theme in THEME_KEYWORDS.keys():
        matched: List[str] = []
        for kw_norm, pat in _PATTERNS[theme]:
            if pat.search(t):
                matched.append(kw_norm)
        if matched:
            out.append(theme)
            hits[theme] = matched

    return out, hits
