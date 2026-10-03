import os
import streamlit as st
import sys
sys.path.insert(0, "src")

from agent import MovieIntelAgent

st.set_page_config(
    page_title="Movie Intelligence & Follow-up Assistant",
    page_icon="🎬",
    layout="wide"
)

st.title("Movie Intelligence & Follow-up Assistant")
st.caption("RAG Subtitle Engine (BM25 + Vector + LLM Scene Summarization) with MCP Email Tool & Agentic Routing")

@st.cache_resource
def load_agent():
    agent = MovieIntelAgent(movies_dir="data/movies")
    agent.initialize(verbose=False)
    return agent

with st.spinner("Initializing Subtitle RAG Index & LLM Enrichment Engine..."):
    agent = load_agent()

st.sidebar.header("Ingestion Status")
st.sidebar.success(f"Indexed Chunks: {len(agent.rag_engine.chunks)}")
st.sidebar.info(f"Movies Directory: `data/movies/`")

st.sidebar.markdown("---")
st.sidebar.subheader("Example Queries")
example_queries = [
    "Who does Steve Rogers meet during his morning run?",
    "Who is Sam Wilson?",
    "What happens when Steve goes running?",
    "Email the morning run scene breakdown to director@marvel.com",
    "Send an email"
]

for eq in example_queries:
    if st.sidebar.button(eq, key=eq):
        st.session_state["user_input"] = eq

user_input = st.text_input("Enter your question or request:", key="user_input", value=st.session_state.get("user_input", ""))

if st.button("Submit Request", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a query or request.")
    else:
        with st.spinner("Processing request with Agentic Decision Router..."):
            res = agent.process_request(user_input)

        status = res.get("status")
        route_type = res.get("route_type", "UNKNOWN")

        st.markdown("###Agent Decision & Output")

        if route_type == "INFORMATIONAL":
            st.info(f"**Route:** `INFORMATIONAL` -> Routed to Subtitle RAG Engine")
        elif route_type == "ACTION_EMAIL":
            st.success(f"**Route:** `ACTION_EMAIL` -> Executed via MCP Email Tool")
        elif route_type == "AMBIGUOUS":
            st.warning(f"**Route:** `AMBIGUOUS` -> Clarification Required")

        if status == "CLARIFICATION_NEEDED":
            st.error(f"**Clarification Needed:** {res.get('message')}")

        elif status == "SUCCESS":
            st.markdown(res.get("answer"))

            citations = res.get("citations", [])
            if citations:
                st.markdown("---")
                st.subheader("Source Citations")
                for i, c in enumerate(citations):
                    with st.expander(f"Citation {i+1}: {c['movie']} [{c['start']} -> {c['end']}]"):
                        if c.get("summary"):
                            st.markdown(f"**Scene Context Summary:**\n_{c['summary']}_")
                        st.markdown(f"**Dialogue Snippet:**\n```\n{c['dialogue_snippet']}\n```")

            if "mcp_dispatch" in res:
                st.markdown("---")
                st.subheader("MCP Email Dispatch Log")
                st.json(res["mcp_dispatch"])
