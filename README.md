# ⚡ Agentic RAG Engine (LangGraph + Verifier/Critic)

A production-ready, autonomous **Agentic RAG (Retrieval-Augmented Generation)** framework built with **LangGraph**, **LangChain**, and **Streamlit**.

It features an intelligent **Reasoning Agent**, multi-source retrieval tools (**Vector DB**, **SQL Database**, **Web Search**, **Knowledge Graph**), an **Evidence Fusion** pipeline, and a strict **Verifier & Critic** evaluation node with automatic critique feedback retry loops.

---

## 🌟 Key Features

- **🧠 Intelligent Reasoning Agent**: Decomposes natural language queries, plans retrieval strategy, and routes targeted sub-queries to active tools.
- **🎛️ Toggleable Multi-Source Retrievers**:
  - **Vector DB (`ChromaDB`)**: Semantic vector search over corporate annual report PDFs and document summaries (indexed for **Reliance Industries** and **Tata Motors**).
  - **SQL Database (`SQLite`)**: Structured financial database for revenue, EBITDA, net profit, net debt, and EV market share across fiscal years.
  - **Web Search (`DuckDuckGo`)**: Live web search integration for real-time external information.
  - **Knowledge Graph (`NetworkX`)**: Entity-relationship graph linking corporate entities, subsidiaries (JLR, TPEM, Jio, Retail), and strategic initiatives.
- **🔀 Evidence Fusion**: Merges, deduplicates, and re-ranks evidence snippets into a structured context payload.
- **🧐 Verifier & Critic Node**: Evaluates answer groundedness and relevance. If confidence score < threshold, triggers an **automatic critique feedback loop** back to the Reasoning Agent.
- **💬 Dual Operating Modes**:
  - **RAG Multi-Source Mode**: Grounded retrieval & synthesis when data sources are active.
  - **Direct LLM Conversational Mode**: Seamless conversational assistant response when data sources are toggled off.
- **🖥️ Streamlit Web Dashboard**: Interactive UI with real-time execution trace, source toggles, score cards, and audit accordions.

---

## 📐 Architecture Diagram

```mermaid
flowchart TD
    UQ[User Query] --> RA[1. Reasoning Agent Node]
    
    subgraph Data Sources - Toggleable via Sidebar UI
        VDB[Vector DB Retriever - Chroma]
        SQL[SQL Database Engine - SQLite]
        WS[Web Search Tool - DuckDuckGo]
        KG[Knowledge Graph Engine - NetworkX]
    end

    RA -->|Routes active sources| VDB
    RA -->|Routes active sources| SQL
    RA -->|Routes active sources| WS
    RA -->|Routes active sources| KG

    VDB --> EF[2. Evidence Fusion Node]
    SQL --> EF
    WS --> EF
    KG --> EF

    EF --> VC{3. Verifier / Critic Node}
    
    VC -- "Passed (Score >= Threshold)" --> FA[4. Final Answer]
    VC -- "Failed / Low Confidence (Score < Threshold)" -->|Critique Feedback Loop| RA
```

---

## 📁 Repository Structure

```text
agentic-rag-engine/
├── app.py                      # Streamlit Web Application Dashboard
├── graph.py                    # LangGraph StateGraph pipeline definition
├── config.py                   # Configuration settings and paths
├── requirements.txt            # Python package dependencies
├── .env.example                # Environment variables template
├── README.md                   # Project documentation
├── agents/
│   ├── reasoning_agent.py      # Query decomposition & sub-query routing node
│   └── verifier_critic.py      # Verifier node assessing groundedness & score
├── retrievers/
│   ├── vector_store.py         # ChromaDB vector indexer & retriever
│   ├── sql_engine.py           # SQLite structured financial database engine
│   ├── web_search.py           # DuckDuckGo live web search retriever
│   └── knowledge_graph.py      # NetworkX entity-relationship graph
└── pipeline/
    └── evidence_fusion.py      # Evidence merge, re-ranking & candidate synthesis
```

---

## 🚀 Quickstart Guide

### 1. Clone & Navigate
```bash
git clone https://github.com/Vivek-Tippireddy/agentic-rag-engine.git
cd agentic-rag-engine
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables (Optional)
Copy `.env.example` to `.env` and set your API keys:
```ini
GEMINI_API_KEY=your_google_gemini_api_key
HF_TOKEN=your_huggingface_token
```

### 4. Launch the Web Application
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 📜 License
MIT License. Created for AI research & Agentic RAG development.
