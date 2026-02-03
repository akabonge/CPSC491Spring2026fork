# scripts/eval.py
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple

from app.rag import generate_grounded_response

SAMPLE_QUERIES = [
    "Summarize the main customer complaints and praises",
    "What themes come up about service speed?",
    "Any consistent issues with pricing or value?",
    "Any consistent issues with cleanliness or food safety?",
]

REQUIRED_TOP_LEVEL_KEYS = [
    "answer_summary",
    "top_themes",
    "overall_sentiment",
    "recurring_issues",
    "sms_draft",
    "ops_recommendations",
]

def assert_sms(out: Dict[str, Any]) -> List[str]:
    errors = []
    sms = out.get("sms_draft") or {}
    msgs = sms.get("messages") or []
    if not isinstance(msgs, list):
        return ["sms_draft.messages not a list"]
    for i, m in enumerate(msgs):
        if not isinstance(m, str):
            errors.append(f"sms message {i} not a string")
        elif len(m) > 160:
            errors.append(f"sms message {i} too long ({len(m)} chars)")
    return errors

def assert_grounding_ids(out: Dict[str, Any]) -> List[str]:
    errors = []
    dbg = out.get("debug") or {}
    allowed = set(dbg.get("allowed_chunk_ids") or [])
    if not allowed:
        # If debug not included, we can’t validate this rule.
        return []

    referenced = set()

    for th in out.get("top_themes", []) or []:
        for e in (th.get("evidence") or []):
            cid = e.get("chunk_id")
            if cid:
                referenced.add(cid)

    for it in out.get("recurring_issues", []) or []:
        for cid in (it.get("evidence_chunk_ids") or []):
            referenced.add(cid)

    for it in out.get("isolated_issues", []) or []:
        for cid in (it.get("evidence_chunk_ids") or []):
            referenced.add(cid)

    for it in out.get("ops_recommendations", []) or []:
        for cid in (it.get("grounding_chunk_ids") or []):
            referenced.add(cid)

    bad = sorted([cid for cid in referenced if cid not in allowed])
    if bad:
        errors.append("Referenced chunk_ids not in allowed_chunk_ids:\n" + "\n".join(bad))
    return errors

def assert_schema_and_shape(out: Dict[str, Any]) -> List[str]:
    errors = []
    if not isinstance(out, dict):
        return ["Output is not a dict"]

    for k in REQUIRED_TOP_LEVEL_KEYS:
        if k not in out:
            errors.append(f"Missing top-level key: {k}")

    themes = out.get("top_themes", [])
    if not isinstance(themes, list):
        errors.append("top_themes not a list")
    else:
        if len(themes) < 3 or len(themes) > 5:
            errors.append(f"top_themes length should be 3–5, got {len(themes)}")

    recs = out.get("ops_recommendations", [])
    if isinstance(recs, list) and (len(recs) < 2 or len(recs) > 3):
        errors.append(f"ops_recommendations length should be 2–3, got {len(recs)}")

    osent = out.get("overall_sentiment") or {}
    if not isinstance(osent, dict) or "label" not in osent or "rationale" not in osent:
        errors.append("overall_sentiment missing label/rationale")

    return errors

def run_one(q: str, top_k: int, min_recurring_reviews: int) -> Tuple[Dict[str, Any], List[str]]:
    out = generate_grounded_response(
        query=q,
        top_k=top_k,
        min_recurring_reviews=min_recurring_reviews,
        include_debug=True,
    )

    errs: List[str] = []
    errs += assert_schema_and_shape(out)
    errs += assert_sms(out)
    errs += assert_grounding_ids(out)
    return out, errs

def main(top_k: int, min_recurring_reviews: int, out_path: str) -> None:
    all_fails = 0
    stamp = datetime.now().isoformat(timespec="seconds")

    results = []
    for q in SAMPLE_QUERIES:
        out, errs = run_one(q, top_k, min_recurring_reviews)

        results.append({
            "timestamp": stamp,
            "query": q,
            "passed": not bool(errs),
            "errors": errs,
            "output": out,
        })

        if errs:
            all_fails += 1
            print("\n" + "=" * 80)
            print("QUERY:", q)
            print("FAILED CHECKS:")
            for e in errs:
                print("-", e)
        else:
            print(f"[PASS] {q}")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote eval results -> {out_path}")

    if all_fails:
        raise SystemExit(f"\nEvaluation finished with {all_fails} failing queries.")
    print("\nAll checks passed.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=25)
    ap.add_argument("--min_recurring_reviews", type=int, default=2)
    ap.add_argument("--out", default="data/eval_results.jsonl")
    args = ap.parse_args()
    main(args.top_k, args.min_recurring_reviews, args.out)
