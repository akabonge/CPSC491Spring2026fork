# scripts/01_clean_reviews.py
import json
import os
import re
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from app.util import normalize_review

load_dotenv()


OWNER_START_RE = re.compile(r"^\s*(hi|hello|hey)\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")


def looks_like_owner_response(text: str, business_name: str) -> bool:
    """
    Deterministic heuristic to detect owner/management replies that are stored
    as separate records in scraped JSON.

    We keep this intentionally simple + robust:
    - Starts with Hi/Hello/Hey
    - AND contains at least one "owner-ish" signal (business name, team, thanks, contact info, etc.)
    """
    if not text:
        return False

    t = text.strip()

    # Quick reject: extremely short texts are unlikely to be owner replies
    if len(t) < 20:
        return False

    starts_like_reply = bool(OWNER_START_RE.match(t))

    owner_signals = 0
    lower = t.lower()

    if business_name and business_name.lower() in lower:
        owner_signals += 2  # strong signal

    if "team" in lower:
        owner_signals += 1

    if "we appreciate" in lower or "we're glad" in lower or "thanks for" in lower or "thank you" in lower:
        owner_signals += 1

    if EMAIL_RE.search(t) or PHONE_RE.search(t):
        owner_signals += 2  # very strong signal

    if "we take" in lower and ("seriously" in lower or "standards" in lower or "protocol" in lower):
        owner_signals += 1

    # Decision rule: must look like a reply AND have enough owner signals
    return starts_like_reply and owner_signals >= 2


def main(in_path: str, out_path: str) -> None:
    business_name = os.getenv("BUSINESS_NAME", "Unknown Business")
    business_location = os.getenv("BUSINESS_LOCATION", "Unknown Location")
    source = os.getenv("REVIEW_SOURCE", "google")

    with open(in_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cleaned = []
    owner_added = 0

    last_customer_review_id: Optional[str] = None

    for i, r in enumerate(raw):
        nr = normalize_review(r, i, business_name, business_location, source)
        if not nr:
            continue

        # If normalize_review provided nested owner_response, keep your original behavior:
        owner_resp = nr.pop("owner_response", None)

        text = (nr.get("text") or "").strip()

        # Detect "standalone" owner replies in raw list (common in your JSON)
        if looks_like_owner_response(text, business_name):
            # Convert this record into a canonical owner-response record
            parent_id = last_customer_review_id

            # If we can link to a parent, create stable owner id using parent id
            review_id = nr.get("review_id") or f"r{i:05d}"
            if parent_id:
                review_id = f"{parent_id}-owner"

            cleaned.append({
                "review_id": review_id,
                "parent_review_id": parent_id,
                "author": business_name or "Owner",
                "rating": None,
                "text": text,
                "date_raw": nr.get("date_raw") or "",
                "date": nr.get("date"),
                "is_owner_response": True,
                "business_name": business_name,
                "business_location": business_location,
                "source": source,
            })
            owner_added += 1
            continue

        # Otherwise treat as customer review
        nr["is_owner_response"] = bool(nr.get("is_owner_response", False))
        cleaned.append(nr)
        last_customer_review_id = nr["review_id"]

        # Separate owner record (optional) if nested exists
        if owner_resp and owner_resp.get("text"):
            cleaned.append({
                "review_id": f"{nr['review_id']}-owner",
                "parent_review_id": nr["review_id"],
                "author": business_name or "Owner",
                "rating": None,
                "text": owner_resp["text"],
                "date_raw": owner_resp.get("date_raw") or "",
                "date": owner_resp.get("date"),
                "is_owner_response": True,
                "business_name": business_name,
                "business_location": business_location,
                "source": source,
            })
            owner_added += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    n_owner = sum(1 for x in cleaned if x.get("is_owner_response"))
    print(f"Loaded raw: {len(raw)}")
    print(f"Kept cleaned: {len(cleaned)}")
    print(f"Owner responses added as separate records: {owner_added}")
    print(f"Owner responses flagged (total in cleaned): {n_owner}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()
    main(args.inp, args.out)
