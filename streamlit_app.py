import json
import streamlit as st
from app.rag import generate_grounded_response

st.set_page_config(page_title="Customer Review Insights Bot", layout="wide")

st.title("Customer Review Insights Bot")
st.caption("Ask a question about your reviews. The bot retrieves relevant chunks and generates a grounded response.")

# ----------------------------
# Sidebar (minimal)
# ----------------------------
with st.sidebar:
    st.header("Settings")

    top_k = st.slider("Retrieved chunks (top_k)", 3, 30, 8, 1)
    min_recurring = st.slider("Min recurring count", 1, 6, 2, 1)

    show_evidence = st.toggle("Show evidence", value=True)
    show_ops = st.toggle("Show ops recommendations", value=True)
    show_sms = st.toggle("Show SMS draft", value=True)

    # Keep debug truly optional + hidden
    with st.expander("Advanced"):
        include_debug = st.checkbox("Include debug block", value=False)
        st.caption("Tip: Keep Advanced off for demos.")

# ----------------------------
# Session state (chat history)
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"q":..., "out":...}

# ----------------------------
# Chat input
# ----------------------------
with st.container():
    st.subheader("Ask a question")
    with st.form("ask_form", clear_on_submit=False):
        q = st.text_input(
            label="",
            placeholder="e.g., What themes come up about service speed?",
            value="What themes come up about service speed?",
        )
        submitted = st.form_submit_button("Run analysis", type="primary")

# ----------------------------
# Run
# ----------------------------
if submitted and q.strip():
    try:
        with st.spinner("Retrieving + generating grounded response..."):
            out = generate_grounded_response(
                query=q.strip(),
                top_k=int(top_k),
                min_recurring_reviews=int(min_recurring),
                include_debug=bool(include_debug),
            )
    except Exception as e:
        st.error("Something went wrong while running the analysis.")
        st.exception(e)
        st.stop()

    if not isinstance(out, dict):
        st.error("Unexpected output type (expected dict).")
        st.write(out)
        st.stop()

    # Store in history (latest first)
    st.session_state.history.insert(0, {"q": q.strip(), "out": out})

# ----------------------------
# Render a single response card
# ----------------------------
def render_response_card(q: str, out: dict):
    # Header
    st.markdown(f"### ❓ {q}")

    # Main answer (clean + demo-friendly)
    ans = out.get("answer_summary", "")
    if ans:
        st.markdown("**Answer**")
        st.write(ans)

    # Overall sentiment (compact)
    osent = out.get("overall_sentiment") or {}
    if osent:
        st.markdown(
            f"**Overall sentiment:** `{osent.get('label', 'mixed')}` — {osent.get('rationale','')}"
        )

    # Tabs for optional detail (keeps UI uncluttered)
    tabs = st.tabs(["Themes", "Issues", "SMS", "Ops", "Evidence", "Raw JSON"])

    # THEMES
    with tabs[0]:
        themes = out.get("top_themes", []) or []
        if not themes:
            st.info("No themes returned.")
        else:
            for th in themes:
                theme_name = th.get("theme", "Theme")
                sent = th.get("sentiment", "mixed")
                with st.expander(f"{theme_name}  •  sentiment: {sent}", expanded=False):
                    evidence = th.get("evidence", []) or []
                    if not evidence:
                        st.write("_No evidence provided for this theme._")
                    else:
                        for e in evidence:
                            cid = e.get("chunk_id", "")
                            excerpt = e.get("excerpt", "")
                            st.markdown(f"- `{cid}` — {excerpt}")

    # ISSUES
    with tabs[1]:
        recurring = out.get("recurring_issues", []) or []
        isolated = out.get("isolated_issues", []) or []

        if not recurring and not isolated:
            st.info("No issues returned.")
        else:
            if recurring:
                st.markdown("**Recurring issues**")
                for i, it in enumerate(recurring, 1):
                    issue = it.get("issue", "")
                    ids = it.get("evidence_chunk_ids", []) or []
                    ucnt = it.get("unique_review_count")
                    with st.expander(f"{i}. {issue}", expanded=False):
                        if ucnt is not None:
                            st.write(f"Unique review count: `{ucnt}`")
                        if ids:
                            st.caption("Evidence chunk IDs")
                            st.code("\n".join(ids))

            if isolated:
                st.markdown("**Isolated issues**")
                for i, it in enumerate(isolated, 1):
                    issue = it.get("issue", "")
                    ids = it.get("evidence_chunk_ids", []) or []
                    with st.expander(f"{i}. {issue}", expanded=False):
                        if ids:
                            st.caption("Evidence chunk IDs")
                            st.code("\n".join(ids))

    # SMS
    with tabs[2]:
        if not show_sms:
            st.info("SMS draft hidden (toggle on in sidebar).")
        else:
            sms = out.get("sms_draft") or {}
            msgs = sms.get("messages", []) or []
            counts = sms.get("character_counts", []) or []
            if not msgs:
                st.info("No SMS draft returned.")
            else:
                for i, m in enumerate(msgs, 1):
                    c = counts[i - 1] if (i - 1) < len(counts) else len(m)
                    st.markdown(f"**Message {i}** ({c} chars)")
                    st.code(m)

    # OPS
    with tabs[3]:
        if not show_ops:
            st.info("Ops recommendations hidden (toggle on in sidebar).")
        else:
            recs = out.get("ops_recommendations", []) or []
            if not recs:
                st.info("No ops recommendations returned.")
            else:
                for i, r in enumerate(recs, 1):
                    rec = r.get("recommendation", "")
                    ids = r.get("grounding_chunk_ids", []) or []
                    with st.expander(f"{i}. {rec}", expanded=False):
                        if ids:
                            st.caption("Grounding chunk IDs")
                            st.code("\n".join(ids))

    # EVIDENCE + DEBUG (only if you want it)
    with tabs[4]:
        if not show_evidence:
            st.info("Evidence hidden (toggle on in sidebar).")
        else:
            # If your rag.py returns a debug block with retrieved chunks, show it nicely
            dbg = out.get("debug", {}) or {}
            retrieved = dbg.get("retrieved", []) or dbg.get("contexts", []) or []
            if retrieved:
                st.markdown("**Retrieved chunks (debug)**")
                for c in retrieved:
                    cid = c.get("id") or c.get("chunk_id") or ""
                    score = c.get("score")
                    rating = c.get("rating")
                    date = c.get("date")
                    text = (c.get("text") or "").strip()
                    with st.expander(f"{cid}  •  score={score}  •  rating={rating}  •  date={date}", expanded=False):
                        st.write(text)
            else:
                st.info("No retrieved chunk list found in debug output. (This is fine for demo mode.)")

            if include_debug:
                st.divider()
                st.markdown("**Debug block**")
                st.json(out.get("debug", {}))

    # RAW JSON
    with tabs[5]:
        st.code(json.dumps(out, ensure_ascii=False, indent=2), language="json")

    st.divider()


# ----------------------------
# Render history (latest first)
# ----------------------------
if st.session_state.history:
    st.subheader("Results")
    for item in st.session_state.history:
        render_response_card(item["q"], item["out"])
else:
    st.info("Ask a question to begin.")
