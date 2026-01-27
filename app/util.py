# app/util.py
import re
from typing import Any, Dict, Optional, Tuple
import dateparser

URL_RE = re.compile(r"https?://\S+")
JUNK_HINT_RE = re.compile(r"\b(wanderlog|facebook|youtube|tiktok|read more|report review)\b", re.I)

# Common owner-response headings that sometimes get appended to customer text exports
OWNER_BLOCK_RE = re.compile(
    r"(?is)\b(?:owner|management)\s*(?:response|reply)\b\s*[:\-]\s*(.+)$"
)

# Another common pattern: "Response from the owner: ..."
OWNER_BLOCK2_RE = re.compile(
    r"(?is)\bresponse\s+from\s+the\s+owner\b\s*[:\-]\s*(.+)$"
)

# Heuristic "signature" patterns (brand + sign-off lines)
BRAND_HINT_RE = re.compile(
    r"(?i)\bcrimson\s+coward\b|\bcrimson\s+coward\s+team\b|—\s*crimson\s+coward|-+\s*crimson\s+coward"
)

THANKS_RE = re.compile(r"(?i)\bthank(s| you)\b")
APOLOGY_RE = re.compile(r"(?i)\b(sorry|apologize|apologies)\b")
INVITE_BACK_RE = re.compile(r"(?i)\b(visit|come back|next time|hope to see you)\b")
# Heuristic: owner replies often appear as a "tail" appended to customer text without a heading.
# We'll detect an owner-like tail in the last N characters and split it off.
OWNER_TAIL_CUE_RE = re.compile(
    r"(?i)\b("
    r"thank(s| you)|"
    r"apologize|apologies|sorry|"
    r"we (appreciate|take|strive|hope|look forward)|"
    r"please (reach out|contact)|"
    r"come back|visit again|next time|hope to see you"
    r")\b"
)

def split_owner_response_tail(text: str, tail_window: int = 450, min_owner_len: int = 40) -> Tuple[str, Optional[str]]:
    """
    Deterministic tail-splitting:
    If the last `tail_window` chars contains strong owner-reply cues AND brand hint OR greeting/apology,
    split that tail as owner_text.
    Returns (customer_text, owner_text_or_none).
    """
    t = clean_text(text)
    if not t:
        return "", None
    if len(t) < min_owner_len + 20:
        return t, None

    tail = t[-tail_window:]
    # Require cues in tail; then confirm with stronger owner signature (brand OR starts with greeting OR apology/thanks)
    if not OWNER_TAIL_CUE_RE.search(tail):
        return t, None

    # Find a reasonable split point: first cue match inside tail
    m = OWNER_TAIL_CUE_RE.search(tail)
    if not m:
        return t, None

    split_at = max(0, len(t) - tail_window + m.start())

    owner_text = clean_text(t[split_at:])
    customer_text = clean_text(t[:split_at])

    # Confirm owner_text looks like an owner response (use your existing detector)
    if len(owner_text) >= min_owner_len and is_owner_response_text(owner_text):
        # If customer part becomes too short, keep original as customer (avoid false positives)
        if len(customer_text) < 20:
            return t, None
        return customer_text, owner_text

    return t, None


def clean_text(s: str) -> str:
    s = s or ""
    s = s.replace("\u200b", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def looks_like_junk(text: str) -> bool:
    if not text:
        return True
    if URL_RE.search(text) and JUNK_HINT_RE.search(text):
        return True
    if "https://" in text and len(text) < 120:
        return True
    return False


def split_owner_response(text: str) -> Tuple[str, Optional[str]]:
    """
    If a review text includes an embedded owner reply (often appended by exports),
    split it out so there is NO overlap between customer text and owner reply.
    Returns (customer_text, owner_text_or_none).
    """
    t = clean_text(text)
    if not t:
        return "", None

    m = OWNER_BLOCK_RE.search(t)
    if not m:
        m = OWNER_BLOCK2_RE.search(t)

    if not m:
        return t, None

    owner_text = clean_text(m.group(1))
    customer_text = clean_text(t[: m.start()].strip())

    # If customer_text ended up empty (rare), keep original text as customer and drop owner
    if not customer_text:
        return t, None

    # If owner_text is too short, ignore it (avoid false splits)
    if len(owner_text) < 15:
        return t, None

    return customer_text, owner_text


def is_owner_response_text(text: str) -> bool:
    """
    Robust owner-response detection for standalone texts.
    Used when the record is itself an owner reply (or the split-out owner part).
    """
    t = clean_text(text).lower()
    if not t:
        return False

    # Strong signals
    if "owner response" in t or "management response" in t:
        return True
    if BRAND_HINT_RE.search(t) and (THANKS_RE.search(t) or APOLOGY_RE.search(t) or INVITE_BACK_RE.search(t)):
        return True

    # Weaker heuristic (keep conservative to avoid misclassifying customers)
    starts_reply = t.startswith(("hi ", "hello ", "hey "))
    mentions_brand = bool(BRAND_HINT_RE.search(t))
    return bool(starts_reply and mentions_brand)


def parse_date_to_iso(date_str: str) -> Optional[str]:
    if not date_str:
        return None
    dt = dateparser.parse(date_str, settings={"PREFER_DATES_FROM": "past"})
    if not dt:
        return None
    return dt.date().isoformat()


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _extract_owner_from_raw(raw: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Best-effort extraction of owner response text/date from raw schemas (separate fields),
    so we can store it without mixing it into customer text.
    """
    OWNER_TEXT_KEYS = [
        "responseFromOwnerText",
        "response_from_owner_text",
        "ownerResponseText",
        "owner_response_text",
    ]
    OWNER_DATE_KEYS = [
        "responseFromOwnerDate",
        "response_from_owner_date",
        "ownerResponseDate",
        "owner_response_date",
    ]

    def first_str(keys):
        for k in keys:
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                return clean_text(v)
        return None

    owner_text = first_str(OWNER_TEXT_KEYS)
    owner_date = first_str(OWNER_DATE_KEYS)

    # nested objects
    if owner_text is None:
        for k in ("ownerResponse", "owner_response", "responseFromOwner", "response_from_owner"):
            v = raw.get(k)
            if isinstance(v, dict):
                for tk in ("text", "comment", "response", "body", "content"):
                    tv = v.get(tk)
                    if isinstance(tv, str) and tv.strip():
                        owner_text = clean_text(tv)
                        break
                if owner_date is None:
                    for dk in ("date", "date_raw", "publishedAt", "updateTime", "createdAt"):
                        dv = v.get(dk)
                        if isinstance(dv, str) and dv.strip():
                            owner_date = clean_text(dv)
                            break
            if owner_text:
                break

    return owner_text, owner_date


def normalize_review(
    raw: Dict[str, Any],
    idx: int,
    business_name: str,
    business_location: str,
    source: str
) -> Optional[Dict[str, Any]]:
    author = clean_text(str(raw.get("author", ""))) or "Unknown"
    rating = safe_int(raw.get("rating", 0), 0)

    raw_text = clean_text(str(raw.get("text", "")))
    date_raw = clean_text(str(raw.get("date", "")))

    if looks_like_junk(raw_text):
        return None

    # 1) Split embedded owner blocks from customer text to guarantee no overlap
    customer_text, embedded_owner_text = split_owner_response(raw_text)

# 1b) If no heading-based split happened, try tail-based split (owner reply appended without label)
    if not embedded_owner_text:
                customer_text, embedded_owner_text = split_owner_response_tail(customer_text)
    # 1c) If what remains still looks like an owner response, drop (this row isn't a customer review)
    if is_owner_response_text(customer_text):
        return None

    # 2) Extract owner response if present as separate raw field(s)
    field_owner_text, field_owner_date = _extract_owner_from_raw(raw)

    # Prefer separate-field owner response, else embedded split
    owner_text = field_owner_text or embedded_owner_text
    owner_date_raw = field_owner_date

    out: Dict[str, Any] = {
        "review_id": f"r{idx:05d}",
        "author": author,
        "rating": rating,
        "text": customer_text,
        "date_raw": date_raw,
        "date": parse_date_to_iso(date_raw),
        "is_owner_response": False,  # this record is customer-facing by definition
        "business_name": business_name,
        "business_location": business_location,
        "source": source,
    }

    if owner_text:
        out["owner_response"] = {
            "text": owner_text,
            "date_raw": owner_date_raw,
            "date": parse_date_to_iso(owner_date_raw) if owner_date_raw else None,
        }

    return out
