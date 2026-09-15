# Agentic RAG Engine — Evaluation Harness & Benchmark Suite

This repository contains the official evaluation harness for the **Agentic RAG Engine** research project. The harness is designed to empirically benchmark multi-turn reasoning, retriever routing, source toggling, and verifier/critic feedback loops against standard RAG baselines.

---

## 🎯 Research Claims Tested

This evaluation suite tests two primary claims for research poster presentation:

1. **Claim 1 (Reasoning Agent & Verifier Loop Impact)**:
   - Multi-step query decomposition, evidence fusion, and verifier critique loops meaningfully improve answer **Faithfulness/Groundedness** and **Correctness** compared to a single-pass, single-source RAG baseline (ector_db only).

2. **Claim 2 (Source Toggling & Context Expansion)**:
   - User-controlled source toggles (ector_db, sql_api, knowledge_graph, web_search) allow the pipeline to retrieve domain-specific structured data and live web information that static vector stores alone cannot provide.

---

## 📁 Directory Structure

`	ext
agentic-rag-evaluation-harness/
├── test_set.json          # 24 Ground-truth questions across 4 domain buckets
├── run_conditions.py      # Executes evaluation across baseline, auto, and manual_toggle conditions
├── score.py               # Auditable LLM Judge & deterministic rubric scoring engine
├── analyze.py             # Statistical aggregation and summary report generator
├── README.md              # Documentation & reproduction guide (this file)
└── results/               # Output benchmark artifacts
    ├── baseline.json      # Raw execution traces for VectorDB-only single-pass baseline
    ├── auto.json          # Raw execution traces for full Reasoning Agent + Verifier loop
    ├── manual_toggle.json # Raw execution traces for target source toggle condition
    ├── scores.csv         # Per-question scores (1-5 Groundedness, 1-5 Correctness, Reasonings)
    ├── summary_data.json  # Aggregated JSON performance metrics
    └── summary.md         # Formatted Markdown summary report for research poster
`

---

## 🚀 Quickstart — How to Run

### 1. Prerequisites & Environment Setup

Ensure you have Python 3.10+ installed and install required dependencies from the main repository:

`ash
pip install -r requirements.txt
`

Set up your API credentials in .env:

`env
# Choose provider: 'gemini' or 'huggingface'
MODEL_PROVIDER=gemini

# API Keys
GEMINI_API_KEY=your_gemini_api_key
HF_TOKEN=your_huggingface_token
HF_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
`

---

### 2. Executing the Evaluation Pipeline

The evaluation pipeline is executed in three clean, single-command steps:

#### Step 1: Run Experimental Conditions

Execute all 24 questions across 3 experimental conditions (aseline, uto, manual_toggle):

`ash
python run_conditions.py
`

*Outputs:* 
esults/baseline.json, 
esults/auto.json, 
esults/manual_toggle.json

#### Step 2: Score System Responses

Evaluate Groundedness (1–5) and Correctness (1–5) using the auditable LLM judge / rubric engine:

`ash
python score.py
`

*Outputs:* 
esults/scores.csv

#### Step 3: Generate Summary Analysis

Aggregate metrics, compute condition means, calculate verifier score deltas, and generate the final report:

`ash
python analyze.py
`

*Outputs:* 
esults/summary.md and 
esults/summary_data.json

---

## 📊 Test Dataset (	est_set.json)

The ground-truth test set consists of **24 benchmark questions** balanced evenly across 4 domain categories (6 per bucket):

1. **ector_only**: Financial strategies, sustainability commitments, and corporate overview notes stored in PDF annual reports.
2. **sql_only**: Exact structured numerical metrics (PAT Net Profit, EBITDA, Revenue, Debt status) stored in SQLite tables.
3. **graph_only**: Corporate entity relations, subsidiary relationships, and de-merger approvals stored in the NetworkX Knowledge Graph.
4. **web_only**: Live stock market news, recent joint ventures, and 2024 product releases requiring DuckDuckGo web search.

---

## 📈 Evaluation Metrics

- **Faithfulness / Groundedness (1–5)**: Extent to which claims in the generated answer are strictly supported by retrieved evidence context.
- **Answer Correctness (1–5)**: Overlap and factual accuracy compared to gold reference answers.
- **Verifier Initial vs Final Score**: Trajectory of verifier critique scoring before and after retry loops.
- **Latency (ms)**: End-to-end execution time per query.
- **LLM Call Count**: Total model invocations per condition.
- **Fallback Tracking (used_fallback)**: Explicit boolean flag indicating whether deterministic rubrics were used during network/quota rate limit events.

---

## 🔄 Dual Provider Switching (Gemini & Hugging Face)

You can switch the LLM inference provider dynamically without code changes by editing .env:

`ash
# To run via Google Gemini 3.6 Flash:
MODEL_PROVIDER=gemini

# To run via Hugging Face Inference API (e.g. Llama-3.1-8B-Instruct):
MODEL_PROVIDER=huggingface
`
