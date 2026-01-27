# scripts/03_index_pinecone.py
import json
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from tqdm import tqdm

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

EMBED_MODEL_DEFAULT = "text-embedding-3-small"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def list_index_names(pc: Pinecone) -> List[str]:
    li = pc.list_indexes()
    if hasattr(li, "names"):
        return list(li.names())
    if isinstance(li, dict) and "indexes" in li:
        return [x.get("name") for x in li["indexes"]]
    if isinstance(li, list):
        names = []
        for x in li:
            if isinstance(x, dict) and "name" in x:
                names.append(x["name"])
            elif hasattr(x, "name"):
                names.append(x.name)
        return names
    return []


def ensure_index(pc: Pinecone, index_name: str, dimension: int, cloud: str, region: str) -> None:
    existing = set(list_index_names(pc))
    if index_name in existing:
        return
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )


def batch(iterable, n: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def reset_index(index) -> None:
    # Deterministic rebuild: remove stale vectors from prior runs
    # Default namespace is "__default__" in serverless; delete_all clears it.
    index.delete(delete_all=True)


def main(chunks_path: str, reset: bool = False, namespace: str = "") -> None:
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "reviews-index")
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")
    embed_model = os.getenv("OPENAI_EMBED_MODEL", EMBED_MODEL_DEFAULT)

    if not openai_key or not pinecone_key:
        raise RuntimeError("Missing OPENAI_API_KEY or PINECONE_API_KEY in .env")

    client = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)

    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise RuntimeError("chunks.jsonl is empty. Run scripts/02_chunk_reviews.py first.")

    # Infer dimension from first embedding
    first_text = chunks[0]["text"]
    first_emb = client.embeddings.create(model=embed_model, input=[first_text]).data[0].embedding
    dim = len(first_emb)

    ensure_index(pc, index_name, dim, cloud, region)
    index = pc.Index(index_name)

    if reset:
        print("Resetting index (delete_all=True) to remove stale vectors...")
        reset_index(index)

    # quick dataset stats (deterministic)
    n_owner = sum(1 for c in chunks if bool(c.get("is_owner_response", False)))
    print(f"Loaded chunks: {len(chunks)} | owner_response chunks: {n_owner}")

    BATCH = 100
    for b in tqdm(list(batch(chunks, BATCH)), desc="Upserting"):
        texts = [x["text"] for x in b]
        emb_resp = client.embeddings.create(model=embed_model, input=texts).data

        vectors = []
        for item, emb_obj in zip(b, emb_resp):
            is_owner = bool(item.get("is_owner_response", False))

            vectors.append({
                "id": item["id"],
                "values": emb_obj.embedding,
                "metadata": {
                    "review_id": item.get("review_id"),
                    "chunk_id": item.get("chunk_id"),
                    "rating": item.get("rating"),
                    "date": item.get("date") or "",
                    "author": item.get("author"),

                    # IMPORTANT: ensure true boolean is stored
                    "is_owner_response": is_owner,
                    "is_owner_response_type": type(is_owner).__name__,  # sanity

                    "business_location": item.get("business_location"),
                    "source": item.get("source"),
                    "themes": item.get("themes") or [],
                    "compound": float((item.get("sentiment") or {}).get("compound", 0.0)),
                    "sentiment_label": (item.get("sentiment") or {}).get("label") or item.get("sentiment_label"),
                    "text": item.get("text") or "",
                }
            })

        # Namespace optional; leave empty to use default
        if namespace:
            index.upsert(vectors=vectors, namespace=namespace)
        else:
            index.upsert(vectors=vectors)

    stats = index.describe_index_stats()
    print("Done. Index stats:")
    print(stats)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--reset", action="store_true", help="Delete all vectors before upsert (recommended for rebuilds).")
    ap.add_argument("--namespace", default="", help="Optional Pinecone namespace (default: __default__).")
    args = ap.parse_args()
    main(args.chunks, reset=args.reset, namespace=args.namespace)
