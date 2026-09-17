import streamlit as st
import os
from pathlib import Path
import config
from graph import agentic_rag_app
from retrievers.vector_store import VectorDBRetriever

# Page Setup & Configuration
st.set_page_config(
    page_title="Agentic RAG Engine — LangGraph & Verifier/Critic",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .verifier-pass {
        background-color: #132a13;
        color: #ecfdf5;
        border: 1px solid #2d6a4f;
        border-left: 6px solid #10b981;
        padding: 16px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .verifier-pass strong {
        color: #34d399;
        font-weight: 700;
    }
    .verifier-fail {
        background-color: #3b1219;
        color: #fff1f2;
        border: 1px solid #9f1239;
        border-left: 6px solid #f43f5e;
        padding: 16px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .verifier-fail strong {
        color: #fb7185;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# App Title & Subtitle
st.markdown('<div class="main-header">⚡ Agentic RAG Engine (LangGraph + Verifier/Critic)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-source retrieval (Vector DB, SQL, Web Search, Knowledge Graph) with groundedness evaluation & critique retry loops.</div>', unsafe_allow_html=True)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ System Settings & Keys")
    
    provider = st.selectbox(
        "LLM Provider",
        ["gemini", "huggingface", "openai", "ollama"],
        index=0,
        help="Select the LLM engine for reasoning, synthesis, and verification."
    )
    
    gemini_key = st.text_input(
        "Google Gemini API Key",
        value=config.GEMINI_API_KEY,
        type="password",
        help="Enter your Gemini API key (or set GEMINI_API_KEY in .env)"
    )

    hf_token = st.text_input(
        "Hugging Face Access Token (HF_TOKEN)",
        value=config.HF_TOKEN,
        type="password",
        help="Stored locally in session memory. Used for HuggingFace model endpoints."
    )

    st.markdown("---")
    st.header("🎛️ Active Data Sources")
    st.caption("Toggle retrieval components to dynamically adjust the Agentic RAG pipeline:")

    toggle_vdb = st.checkbox("📚 Vector DB (PDF & TXT Reports)", value=True)
    toggle_sql = st.checkbox("📊 SQL Engine (Financial Metrics DB)", value=True)
    toggle_web = st.checkbox("🌐 Web Search (DuckDuckGo Live)", value=True)
    toggle_kg  = st.checkbox("🕸️ Knowledge Graph (Entity Relations)", value=True)

    toggles = {
        "vector_db": toggle_vdb,
        "sql_api": toggle_sql,
        "web_search": toggle_web,
        "knowledge_graph": toggle_kg
    }

    st.markdown("---")
    st.header("🧐 Verifier & Critic Parameters")
    
    threshold = st.slider(
        "Verifier Score Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.75,
        step=0.05,
        help="Minimum confidence score required for candidate answer to pass verification."
    )

    max_loops = st.slider(
        "Max Critique Revision Loops",
        min_value=1,
        max_value=4,
        value=2,
        help="Maximum times the Critic node can route feedback back to the Reasoning Agent."
    )

    st.markdown("---")
    st.header("📁 Corporate Dataset Status")
    data_path = Path(config.DEFAULT_DATA_DIR)
    if data_path.exists():
        st.success(f"Connected: `{data_path.name}`")
        if st.button("🔄 Re-Index Vector Store"):
            with st.spinner("Indexing PDFs & TXT files into ChromaDB..."):
                retriever = VectorDBRetriever()
                retriever.build_index()
            st.success("Vector DB re-indexed successfully!")
    else:
        st.error(f"Data directory not found at `{data_path}`")

# Sample Prompts
st.caption("💡 Try these sample prompts:")
col1, col2, col3 = st.columns(3)
sample_prompt = ""
if col1.button("Compare Tata Motors & Reliance EV Strategy"):
    sample_prompt = "Compare Tata Motors and Reliance Industries EV and Green Energy investments."
if col2.button("Tata Motors FY24 & FY25 Profits"):
    sample_prompt = "profits in financial year 2024/2025 for tata motors?"
if col3.button("JLR Reimagine Strategy & Subsidiaries"):
    sample_prompt = "Explain Jaguar Land Rover (JLR) Reimagine Strategy and its role in Tata Motors."

query_input = st.text_input("Enter your research query:", value=sample_prompt if sample_prompt else "")

if st.button("🚀 Run Agentic RAG Pipeline", type="primary", use_container_width=True):
    if not query_input.strip():
        st.warning("Please enter a query to proceed.")
    else:
        inputs = {
            "query": query_input,
            "toggles": toggles,
            "api_key": gemini_key,
            "model_provider": provider,
            "threshold": threshold,
            "max_iterations": max_loops,
            "iteration_count": 0,
            "sub_queries": {},
            "reasoning_trace": [],
            "raw_evidence": [],
            "fused_evidence": {},
            "candidate_answer": "",
            "verifier_result": {},
            "critique_feedback": "",
            "final_answer": ""
        }

        with st.spinner("Agent reasoning, retrieving multi-source evidence, and verifying output..."):
            result_state = agentic_rag_app.invoke(inputs)

        # Output Layout Tabs
        tab_answer, tab_verifier, tab_evidence, tab_trace = st.tabs([
            "💬 Final Verified Answer",
            "🧐 Verifier & Critic Audit",
            "🔍 Multi-Source Evidence",
            "🧠 Agent Reasoning Trace"
        ])

        with tab_answer:
            st.markdown("### Verified Response")
            st.markdown(result_state.get("final_answer", "No answer generated."))

        with tab_verifier:
            v_res = result_state.get("verifier_result", {})
            passed = v_res.get("passed", False)
            score = v_res.get("score", 0.0)
            
            st.markdown("### Verifier Evaluation Card")
            col_v1, col_v2, col_v3 = st.columns(3)
            col_v1.metric("Verification Status", "PASSED ✅" if passed else "FAILED / REVISED ⚠️")
            col_v2.metric("Confidence Score", f"{score:.2f} / {threshold:.2f}")
            col_v3.metric("Revision Loops Run", f"{result_state.get('iteration_count', 1)} / {max_loops}")

            if passed:
                st.markdown(f'<div class="verifier-pass"><strong>Verifier Outcome:</strong> {v_res.get("critique", "Answer passed all groundedness checks.")}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verifier-fail"><strong>Verifier Feedback:</strong> {v_res.get("critique", "Answer required critique revision.")}</div>', unsafe_allow_html=True)

        with tab_evidence:
            st.markdown("### Fused Evidence Context")
            fused = result_state.get("fused_evidence", {})
            st.caption(f"Total Snippets Merged: {fused.get('total_snippets', 0)} | Source Breakdown: {fused.get('source_breakdown', {})}")
            
            for item in fused.get("items", []):
                with st.expander(f"{item['label']} (Score: {item.get('score', 'N/A')})"):
                    st.write(item["content"])

        with tab_trace:
            st.markdown("### LangGraph Reasoning Execution Log")
            for line in result_state.get("reasoning_trace", []):
                st.text(line)
