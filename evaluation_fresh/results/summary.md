# Agentic RAG Engine — Evaluation Summary Report

This document summarizes the empirical evaluation testing the two core research claims for the NLP Poster:

1. **Claim 1**: Reasoning Agent + Verifier/Critic loop improves answer groundedness and correctness over single-pass single-source baseline.
2. **Claim 2**: User-controlled source toggling allows retrieval of domain-specific data outside default vector store.

## 1. Overall Performance Across Conditions

| Condition | Sample Size | Mean Groundedness (1-5) | Mean Correctness (1-5) | Mean Latency (ms) | Mean LLM Calls |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **baseline** | 24 | **4.54** | **4.17** | 21871.71 | 1.0 |
| **auto** | 24 | **4.62** | **4.29** | 30322.77 | 2.17 |
| **manual_toggle** | 24 | **4.46** | **4.0** | 25991.53 | 2.08 |

## 2. Verifier/Critic Retry Impact (Paired Comparison)

- **Total Questions Triggering Retry Loop**: 3
- **Mean Verifier Score BEFORE Retry**: `0.71`
- **Mean Verifier Score AFTER Retry**: `0.81`
- **Mean Score Delta**: `+0.1` (Verification Loop Improvement)

## 3. Per-Bucket Success Rates (Claim 2 Benchmark)

| Category | Baseline Success Rate (%) | Manual Toggle Success Rate (%) | Auto Success Rate (%) | Delta (Manual vs Baseline) |
| :--- | :--- | :--- | :--- | :--- |
| **vector_only** | 66.7% | **66.7%** | 66.7% | **+0.0%** |
| **sql_only** | 100.0% | **50.0%** | 100.0% | **+-50.0%** |
| **graph_only** | 66.7% | **83.3%** | 83.3% | **+16.7%** |
| **web_only** | 16.7% | **16.7%** | 16.7% | **+0.0%** |

## 4. Illustrative Case Studies
