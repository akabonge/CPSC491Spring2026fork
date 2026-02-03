import json
import streamlit as st
from app.rag import generate_grounded_response

st.set_page_config(page_title="🍗 Customer Review Insights Bot", page_icon="🍗", layout="wide")

# -----------------------------
# Sidebar controls (minimal)
# -----------------------------
with st.sidebar:
    st.header("Controls")
    top_k = st.slider("Retrieved chunks (top_k)", min_value=3, max_value=50, value=12, step=1)
    min_recurring = st.slider("Min recurring reviews", min_value=1, max_value=8, value=2, step=1)
    include_debug = st.checkbox("Show debug", value=False)

    st.divider()
    with st.expander("Environment required", expanded=False):
        st.code(
            "OPENAI_API_KEY\nPINECONE_API_KEY\nPINECONE_INDEX\n"
            "OPENAI_MODEL (optional)\nOPENAI_EMBED_MODEL (optional)\n",
            language="text"
        )

# -----------------------------
# Title
# -----------------------------
st.title("🍗 Customer Review Insights Bot")
st.caption("Ask questions about your review dataset. The bot retrieves relevant chunks and generates grounded insights.")

# -----------------------------
# Chat session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me something like: **“Is the food here good?”** or **“What themes come up about service speed?”**"
        }
    ]

# -----------------------------
# Render chat history
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# Helper render (UI only)
# -----------------------------
def render_assistant_payload(out: dict):
    """
    UI-only renderer. Backend logic stays in app/rag.py.
    """
    answer = out.get("answer_summary", "").strip()
    osent = out.get("overall_sentiment") or {}
    label = osent.get("label", "mixed")
    rationale = osent.get("rationale", "")

    # Primary answer (clean)
    if answer:
        st.markdown(answer)
    else:
        st.markdown("_No answer_summary returned._")

    st.markdown(f"\n**Overall sentiment:** `{label}`")
    if rationale:
        st.caption(rationale)

    # Everything else behind expander (no clutter)
    with st.expander("Details (themes, issues, SMS, ops)", expanded=False):
        # Themes
        themes = out.get("top_themes", []) or []
        st.subheader("Themes")
        if not themes:
            st.write("_None._")
        else:
            for th in themes:
                theme_name = th.get("theme", "Theme")
                sent = th.get("sentiment", "mixed")
                st.markdown(f"**{theme_name}** • sentiment: `{sent}`")
                ev = th.get("evidence", []) or []
                if ev:
                    for e in ev:
                        st.markdown(f"- `{e.get('chunk_id','')}` — {e.get('excerpt','')}")
                st.write("")

        # Issues
        rec = out.get("recurring_issues", []) or []
        iso = out.get("isolated_issues", []) or []

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Recurring issues")
            if not rec:
                st.write("_None._")
            else:
                for i, it in enumerate(rec, start=1):
                    issue = it.get("issue", "")
                    ucnt = it.get("unique_review_count", None)
                    ids = it.get("evidence_chunk_ids", []) or []
                    st.markdown(f"**{i}. {issue}**")
                    if ucnt is not None:
                        st.caption(f"Unique review count: `{ucnt}`")
                    if ids:
                        st.code("\n".join(ids))

        with col2:
            st.subheader("Isolated issues")
            if not iso:
                st.write("_None._")
            else:
                for i, it in enumerate(iso, start=1):
                    issue = it.get("issue", "")
                    ucnt = it.get("unique_review_count", None)
                    ids = it.get("evidence_chunk_ids", []) or []
                    st.markdown(f"**{i}. {issue}**")
                    if ucnt is not None:
                        st.caption(f"Unique review count: `{ucnt}`")
                    if ids:
                        st.code("\n".join(ids))

        # SMS + Ops
        sms = out.get("sms_draft") or {}
        msgs = sms.get("messages", []) or []
        counts = sms.get("character_counts", []) or []

        st.subheader("Draft SMS")
        if not msgs:
            st.write("_Not generated for this question._")
        else:
            for i, m in enumerate(msgs, start=1):
                c = counts[i - 1] if i - 1 < len(counts) else len(m)
                st.markdown(f"**Message {i}** ({c} chars)")
                st.code(m)

        recs = out.get("ops_recommendations", []) or []
        st.subheader("Ops recommendations")
        if not recs:
            st.write("_Not generated for this question._")
        else:
            for i, r in enumerate(recs, start=1):
                st.markdown(f"**{i}. {r.get('recommendation','')}**")
                ids = r.get("grounding_chunk_ids", []) or []
                if ids:
                    st.caption("Grounding chunk IDs")
                    st.code("\n".join(ids))

    # Sources always visible but compact
    src = out.get("sources", []) or []
    if src:
        st.markdown("**Sources (chunk IDs):**")
        st.code("\n".join(src))

    # Optional debug (hidden unless toggled)
    if include_debug:
        dbg = out.get("debug", {})
        if dbg:
            with st.expander("Debug", expanded=False):
                st.json(dbg)

# -----------------------------
# Chat input
# -----------------------------
prompt = st.chat_input("Ask a question about the reviews…")

if prompt:
    # user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving + generating grounded response…"):
            out = generate_grounded_response(
                query=prompt,
                top_k=int(top_k),
                min_recurring_reviews=int(min_recurring),
                include_debug=bool(include_debug),
            )

        if not isinstance(out, dict):
            st.error("Unexpected output type (expected dict).")
            st.write(out)
        elif out.get("error"):
            st.error(out.get("error"))
            st.code(out.get("raw_output", ""), language="text")
        else:
            render_assistant_payload(out)

    # store assistant message as a minimal chat bubble (don’t store giant JSON)
    # we store only the answer_summary for chat history readability
    summary = out.get("answer_summary") if isinstance(out, dict) else None
    st.session_state.messages.append(
        {"role": "assistant", "content": summary or "_(Response generated)_"}
    )
