# Agentic RAG Engine — Evaluation Summary Report

This document summarizes the empirical evaluation testing the two core research claims for the NLP Poster:

1. **Claim 1**: Reasoning Agent + Verifier/Critic loop improves answer groundedness and correctness over single-pass single-source baseline.
2. **Claim 2**: User-controlled source toggling allows retrieval of domain-specific data outside default vector store.

## 1. Overall Performance Across Conditions

| Condition | Sample Size | Mean Groundedness (1-5) | Mean Correctness (1-5) | Mean Latency (ms) | Mean LLM Calls |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **baseline** | 24 | **4.5** | **4.08** | 255.14 | 1.0 |
| **auto** | 24 | **4.71** | **4.42** | 1261.1 | 2.0 |
| **manual_toggle** | 24 | **4.42** | **4.0** | 390.53 | 2.0 |

## 2. Verifier/Critic Retry Impact (Paired Comparison)

- **Total Questions Triggering Retry Loop**: 0
- **Mean Verifier Score BEFORE Retry**: `0.0`
- **Mean Verifier Score AFTER Retry**: `0.0`
- **Mean Score Delta**: `+0.0` (Verification Loop Improvement)

## 3. Per-Bucket Success Rates (Claim 2 Benchmark)

| Category | Baseline Success Rate (%) | Manual Toggle Success Rate (%) | Auto Success Rate (%) | Delta (Manual vs Baseline) |
| :--- | :--- | :--- | :--- | :--- |
| **vector_only** | 50.0% | **50.0%** | 66.7% | **+0.0%** |
| **sql_only** | 100.0% | **83.3%** | 100.0% | **+-16.7%** |
| **graph_only** | 66.7% | **83.3%** | 100.0% | **+16.7%** |
| **web_only** | 16.7% | **16.7%** | 16.7% | **+0.0%** |

## 4. Illustrative Case Studies
