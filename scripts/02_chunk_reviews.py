# scripts/02_chunk_reviews.py
import json
from typing import List, Dict, Any
import tiktoken
from dotenv import load_dotenv

from app.themes import detect_themes_with_hits
from app.sentiment import score_sentiment
from app.util import is_owner_response_text  # <-- NEW

load_dotenv()


def chunk_tokens(text: str, max_tokens: int, overlap: int, enc) -> List[str]:
    toks = enc.encode(text or "")
    chunks: List[str] = []
    start = 0
    while start < len(toks):
        end = min(start + max_tokens, len(toks))
        chunk = enc.decode(toks[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(toks):
            break
        start = max(0, end - overlap)
    return chunks


def main(
    in_path: str,
    out_path: str,
    max_tokens: int = 300,
    overlap: int = 50,
    include_owner: bool = False,
) -> None:
    with open(in_path, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    enc = tiktoken.get_encoding("cl100k_base")

    n_chunks = 0
    n_skipped_owner_records = 0
    n_skipped_ownerish_chunks = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for r in reviews:
            # If the whole record is flagged as owner response and we're excluding owners, skip early.
            record_is_owner = bool(r.get("is_owner_response", False))
            if (not include_owner) and record_is_owner:
                n_skipped_owner_records += 1
                continue

            text = (r.get("text") or "").strip()
            if not text:
                continue

            chunks = chunk_tokens(text, max_tokens=max_tokens, overlap=overlap, enc=enc)

            for j, ch in enumerate(chunks):
                # Deterministic owner-ish detection at chunk level (defense-in-depth)
                text_is_ownerish = is_owner_response_text(ch)
                is_owner = record_is_owner or text_is_ownerish

                # If excluding owners, skip any chunk that is owner/owner-ish
                if (not include_owner) and is_owner:
                    n_skipped_ownerish_chunks += 1
                    continue

                themes, theme_hits = detect_themes_with_hits(ch)
                sent = score_sentiment(ch)

                record: Dict[str, Any] = {
                    "id": f'{r["review_id"]}-c{j:02d}',
                    "review_id": r["review_id"],
                    "chunk_id": j,
                    "text": ch,
                    "token_count": len(enc.encode(ch)),
                    "char_count": len(ch),

                    "rating": r.get("rating"),
                    "date": r.get("date"),
                    "date_raw": r.get("date_raw"),
                    "author": r.get("author"),

                    # Hardened owner labeling
                    "is_owner_response": is_owner,
                    "is_owner_response_text_detected": text_is_ownerish,

                    "parent_review_id": r.get("parent_review_id"),
                    "business_name": r.get("business_name"),
                    "business_location": r.get("business_location"),
                    "source": r.get("source"),

                    "themes": themes,
                    "theme_hits": theme_hits,         # auditable
                    "sentiment": sent,                # includes label
                    "sentiment_label": sent["label"], # convenience
                }

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_chunks += 1

    print(f"Chunks written: {n_chunks} -> {out_path}")
    if not include_owner:
        print(f"Owner response records skipped (flagged): {n_skipped_owner_records}")
        print(f"Owner-ish chunks skipped (detected): {n_skipped_ownerish_chunks}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--max_tokens", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--include_owner", action="store_true", help="Include owner responses in chunk output")
    args = ap.parse_args()
    main(args.inp, args.out, args.max_tokens, args.overlap, include_owner=args.include_owner)
