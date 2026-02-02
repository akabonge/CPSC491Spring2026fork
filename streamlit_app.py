"""
Streamlit UI for Customer Review Insights Bot (Crimson Coward)

Usage:
  streamlit run streamlit_app.py
"""
import os
import json
import streamlit as st
from dotenv import load_dotenv

# Your existing RAG function
from app.rag import generate_grounded_response

load_dotenv()

st.set_page_config(
    page_title="Customer Review Insights Bot",
    page_icon="🍗",
    layout="wide",
)

# Optional: show Pinecone stats in sidebar (safe if pinecone installed)
try:
    from pinecone import Pinecone
except Exception:
    Pinecone = None


def _safe_index_stats() -> dict | None:
    """Try to read Pinecone index stats (optional)."""
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX")
    if not pinecone_key or not index_name or Pinecone is None:
        return None
    try:
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        if isinstance(stats, dict):
            return stats
        return getattr(stats, "to_dict", lambda: None)()
    except Exception:
        return None


def render_sources_from_output(out: dict) -> None:
    """
    Render evidence chunk IDs from out's theme evidence + recurring issues + ops recs.
    This keeps the UI clean while still showing grounding.
    """
    # Collect chunk IDs from multiple places
    chunk_ids = set()

    for th in out.get("top_themes", []) or []:
        for e in th.get("evidence", []) or []:
            cid = e.get("chunk_id")
            if cid:
                chunk_ids.add(str(cid))

    for it in out.get("recurring_issues", []) or []:
        for cid in it.get("evidence_chunk_ids", []) or []:
            chunk_ids.add(str(cid))

    for r in out.get("ops_recommendations", []) or []:
        for cid in r.get("grounding_chunk_ids", []) or []:
            chunk_ids.add(str(cid))

    if not chunk_ids:
        st.caption("Sources: (none returned)")
        return

    st.caption("Sources (chunk IDs)")
    st.code("\n".join(sorted(chunk_ids)))


def render_response_clean(out: dict, show_details: bool, show_debug: bool) -> None:
    """Render assistant response in a clean way (demo-friendly)."""
    # Main
    summary = out.get("answer_summary", "")
    if summary:
        st.markdown(summary)
    else:
        st.markdown("_No summary returned._")

    # Expandable details
    if show_details:
        with st.expander("Details (themes, issues, SMS, ops)", expanded=False):
            # Overall sentiment
            osent = out.get("overall_sentiment") or {}
            if osent:
                st.markdown(f"**Overall sentiment:** `{osent.get('label', 'mixed')}`")
                if osent.get("rationale"):
                    st.write(osent["rationale"])

            # Themes
            themes = out.get("top_themes", []) or []
            if themes:
                st.markdown("### Themes")
                for th in themes:
                    theme_name = th.get("theme", "Theme")
                    sent = th.get("sentiment", "mixed")
                    with st.expander(f"{theme_name} • sentiment: {sent}", expanded=False):
                        evidence = th.get("evidence", []) or []
                        if not evidence:
                            st.write("_No evidence provided._")
                        else:
                            for e in evidence:
                                cid = e.get("chunk_id", "")
                                excerpt = e.get("excerpt", "")
                                st.markdown(f"- `{cid}` — {excerpt}")

            # Issues
            issues = out.get("recurring_issues", []) or []
            if issues:
                st.markdown("### Recurring issues")
                for i, it in enumerate(issues, 1):
                    issue = it.get("issue", "")
                    ids = it.get("evidence_chunk_ids", []) or []
                    ucnt = it.get("unique_review_count")
                    st.markdown(f"**{i}. {issue}**")
                    if ucnt is not None:
                        st.caption(f"Unique review count: {ucnt}")
                    if ids:
                        st.code("\n".join(ids))

            # SMS
            sms = out.get("sms_draft") or {}
            msgs = sms.get("messages", []) or []
            counts = sms.get("character_counts", []) or []
            if msgs:
                st.markdown("### Draft SMS")
                for i, m in enumerate(msgs, 1):
                    c = counts[i - 1] if (i - 1) < len(counts) else len(m)
                    st.markdown(f"**Message {i}** ({c} chars)")
                    st.code(m)

            # Ops recommendations
            recs = out.get("ops_recommendations", []) or []
            if recs:
                st.markdown("### Ops recommendations")
                for i, r in enumerate(recs, 1):
                    rec = r.get("recommendation", "")
                    ids = r.get("grounding_chunk_ids", []) or []
                    st.markdown(f"**{i}. {rec}**")
                    if ids:
                        st.caption("Grounding chunk IDs")
                        st.code("\n".join(ids))

            # Sources
            st.divider()
            render_sources_from_output(out)

    # Debug (hidden unless toggled)
    if show_debug:
        with st.expander("Debug JSON", expanded=False):
            st.json(out.get("debug", {}))
        with st.expander("Raw output JSON", expanded=False):
            st.code(json.dumps(out, ensure_ascii=False, indent=2), language="json")


def main():
    st.title("🍗 Customer Review Insights Bot")
    st.markdown("Ask questions about your review dataset. The bot retrieves relevant chunks and generates grounded insights.")

    # Sidebar (like your example app)
    with st.sidebar:
        st.header("📊 System Stats")

        pinecone_index_name = os.getenv("PINECONE_INDEX", "(not set)")
        st.success(f"Pinecone index: {pinecone_index_name}")

        stats = _safe_index_stats()
        if stats and isinstance(stats, dict):
            total = stats.get("total_vector_count")
            dim = stats.get("dimension")
            if total is not None:
                st.info(f"Vectors: {total:,}")
            if dim is not None:
                st.info(f"Dimension: {dim}")

        st.divider()
        st.header("⚙️ Controls")
        top_k = st.slider("top_k", min_value=3, max_value=30, value=8, step=1)
        min_recurring = st.slider("min_recurring_reviews", min_value=1, max_value=6, value=2, step=1)

        st.divider()
        st.header("🧾 Display")
        show_details = st.checkbox("Show details (themes/issues/SMS/ops)", value=True)
        show_debug = st.checkbox("Show debug/raw JSON", value=False)

        st.divider()
        st.caption("Environment required:")
        st.code(
            "OPENAI_API_KEY\nPINECONE_API_KEY\nPINECONE_INDEX\n"
            "OPENAI_MODEL (optional)\nOPENAI_EMBED_MODEL (optional)",
            language="text",
        )

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render existing chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and isinstance(msg.get("payload"), dict):
                render_response_clean(msg["payload"], show_details=show_details, show_debug=show_debug)
            else:
                st.markdown(msg.get("content", ""))

    # Chat input
    prompt = st.chat_input("Ask about the reviews (e.g., 'What themes come up about service speed?')")
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving + generating grounded response..."):
                try:
                    out = generate_grounded_response(
                        query=prompt,
                        top_k=int(top_k),
                        min_recurring_reviews=int(min_recurring),
                        include_debug=bool(show_debug),
                    )
                except TypeError:
                    # If your generate_grounded_response doesn't support these args yet,
                    # fall back to the older signature.
                    out = generate_grounded_response(prompt, top_k=int(top_k))
                except Exception as e:
                    st.error(f"Error: {e}")
                    out = {"answer_summary": "An error occurred while generating the response."}

            if not isinstance(out, dict):
                st.error("Unexpected output (expected dict).")
                st.write(out)
                return

            render_response_clean(out, show_details=show_details, show_debug=show_debug)

        # Store assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": "", "payload": out}
        )


if __name__ == "__main__":
    main()
