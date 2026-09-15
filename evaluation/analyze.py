import os
import sys
import json
import csv
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SCORES_CSV_PATH = os.path.join(RESULTS_DIR, "scores.csv")
SUMMARY_MD_PATH = os.path.join(RESULTS_DIR, "summary.md")
SUMMARY_DATA_JSON_PATH = os.path.join(RESULTS_DIR, "summary_data.json")

def load_scores():
    if not os.path.exists(SCORES_CSV_PATH):
        raise FileNotFoundError(f"Scores CSV file not found at {SCORES_CSV_PATH}. Run score.py first.")

    rows = []
    with open(SCORES_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["retry_fired"] = r["retry_fired"].lower() == "true"
            r["used_fallback"] = r.get("used_fallback", "false").lower() == "true"
            r["verifier_score_initial"] = float(r["verifier_score_initial"])
            r["verifier_score_final"] = float(r["verifier_score_final"])
            r["latency_ms"] = float(r["latency_ms"])
            r["llm_call_count"] = int(r["llm_call_count"])
            r["groundedness_score"] = int(r["groundedness_score"])
            r["correctness_score"] = int(r["correctness_score"])
            rows.append(r)
    return rows

def analyze():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rows = load_scores()

    by_condition = defaultdict(list)
    by_cond_cat = defaultdict(lambda: defaultdict(list))
    retry_cases = []

    for r in rows:
        cond = r["condition"]
        cat = r["category"]
        by_condition[cond].append(r)
        by_cond_cat[cond][cat].append(r)

        if cond in ["auto", "manual_toggle"] and r["retry_fired"]:
            retry_cases.append(r)

    # 1. Condition Level Means
    cond_summary = {}
    for cond, items in by_condition.items():
        n = len(items)
        mean_groundedness = sum(x["groundedness_score"] for x in items) / n
        mean_correctness = sum(x["correctness_score"] for x in items) / n
        mean_latency = sum(x["latency_ms"] for x in items) / n
        mean_llm_calls = sum(x["llm_call_count"] for x in items) / n
        cond_summary[cond] = {
            "sample_size": n,
            "mean_groundedness": round(mean_groundedness, 2),
            "mean_correctness": round(mean_correctness, 2),
            "mean_latency_ms": round(mean_latency, 2),
            "mean_llm_calls": round(mean_llm_calls, 2)
        }

    # 2. Retry Before vs After Verifier Score Analysis
    retry_summary = {}
    if retry_cases:
        mean_initial = sum(x["verifier_score_initial"] for x in retry_cases) / len(retry_cases)
        mean_final = sum(x["verifier_score_final"] for x in retry_cases) / len(retry_cases)
        retry_summary = {
            "retry_fired_count": len(retry_cases),
            "mean_verifier_score_before": round(mean_initial, 2),
            "mean_verifier_score_after": round(mean_final, 2),
            "mean_score_delta": round(mean_final - mean_initial, 2)
        }
    else:
        retry_summary = {
            "retry_fired_count": 0,
            "mean_verifier_score_before": 0.0,
            "mean_verifier_score_after": 0.0,
            "mean_score_delta": 0.0
        }

    # 3. Per-Bucket Success Rates (Baseline vs Manual Toggle for non-vector sources)
    categories = ["vector_only", "sql_only", "graph_only", "web_only"]
    bucket_summary = {}

    for cat in categories:
        b_items = by_cond_cat.get("baseline", {}).get(cat, [])
        m_items = by_cond_cat.get("manual_toggle", {}).get(cat, [])
        a_items = by_cond_cat.get("auto", {}).get(cat, [])

        b_success = (sum(1 for x in b_items if x["correctness_score"] >= 4) / len(b_items) * 100) if b_items else 0.0
        m_success = (sum(1 for x in m_items if x["correctness_score"] >= 4) / len(m_items) * 100) if m_items else 0.0
        a_success = (sum(1 for x in a_items if x["correctness_score"] >= 4) / len(a_items) * 100) if a_items else 0.0

        bucket_summary[cat] = {
            "baseline_success_rate": round(b_success, 1),
            "manual_toggle_success_rate": round(m_success, 1),
            "auto_success_rate": round(a_success, 1),
            "delta_manual_vs_baseline": round(m_success - b_success, 1)
        }

    # 4. Extract Case Studies
    # Find questions where baseline score <= 2 and manual_toggle score >= 4
    case_studies = []
    
    # Load raw result JSON files to extract answers
    raw_results = {}
    for c in ["baseline", "manual_toggle"]:
        path = os.path.join(RESULTS_DIR, f"{c}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw_results[c] = {item["id"]: item for item in json.load(f)}

    baseline_scores_map = {r["id"]: r for r in rows if r["condition"] == "baseline"}
    manual_scores_map = {r["id"]: r for r in rows if r["condition"] == "manual_toggle"}

    for q_id, b_score in baseline_scores_map.items():
        m_score = manual_scores_map.get(q_id)
        if m_score and b_score["correctness_score"] <= 2 and m_score["correctness_score"] >= 4:
            b_raw = raw_results.get("baseline", {}).get(q_id, {})
            m_raw = raw_results.get("manual_toggle", {}).get(q_id, {})
            case_studies.append({
                "id": q_id,
                "category": b_score["category"],
                "question": b_score["question"],
                "baseline_answer": b_raw.get("final_answer", "")[:200] + "...",
                "manual_toggle_answer": m_raw.get("final_answer", "")[:200] + "...",
                "gold_answer": b_raw.get("gold_answer", ""),
                "fixing_source": b_score["target_source"]
            })
            if len(case_studies) >= 3:
                break

    # Build flat JSON summary object
    flat_summary = {
        "conditions": cond_summary,
        "retries": retry_summary,
        "buckets": bucket_summary,
        "case_studies": case_studies
    }

    with open(SUMMARY_DATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(flat_summary, f, indent=2)

    # Build Markdown Summary Report
    md = []
    md.append("# Agentic RAG Engine — Evaluation Summary Report\n")
    md.append("This document summarizes the empirical evaluation testing the two core research claims for the NLP Poster:\n")
    md.append("1. **Claim 1**: Reasoning Agent + Verifier/Critic loop improves answer groundedness and correctness over single-pass single-source baseline.")
    md.append("2. **Claim 2**: User-controlled source toggling allows retrieval of domain-specific data outside default vector store.\n")

    md.append("## 1. Overall Performance Across Conditions\n")
    md.append("| Condition | Sample Size | Mean Groundedness (1-5) | Mean Correctness (1-5) | Mean Latency (ms) | Mean LLM Calls |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for cond in ["baseline", "auto", "manual_toggle"]:
        s = cond_summary.get(cond, {})
        md.append(f"| **{cond}** | {s.get('sample_size', 0)} | **{s.get('mean_groundedness', 0)}** | **{s.get('mean_correctness', 0)}** | {s.get('mean_latency_ms', 0)} | {s.get('mean_llm_calls', 0)} |")

    md.append("\n## 2. Verifier/Critic Retry Impact (Paired Comparison)\n")
    md.append(f"- **Total Questions Triggering Retry Loop**: {retry_summary.get('retry_fired_count', 0)}")
    md.append(f"- **Mean Verifier Score BEFORE Retry**: `{retry_summary.get('mean_verifier_score_before', 0)}`")
    md.append(f"- **Mean Verifier Score AFTER Retry**: `{retry_summary.get('mean_verifier_score_after', 0)}`")
    md.append(f"- **Mean Score Delta**: `+{retry_summary.get('mean_score_delta', 0)}` (Verification Loop Improvement)\n")

    md.append("## 3. Per-Bucket Success Rates (Claim 2 Benchmark)\n")
    md.append("| Category | Baseline Success Rate (%) | Manual Toggle Success Rate (%) | Auto Success Rate (%) | Delta (Manual vs Baseline) |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for cat, s in bucket_summary.items():
        md.append(f"| **{cat}** | {s['baseline_success_rate']}% | **{s['manual_toggle_success_rate']}%** | {s['auto_success_rate']}% | **+{s['delta_manual_vs_baseline']}%** |")

    md.append("\n## 4. Illustrative Case Studies\n")
    for idx, cs in enumerate(case_studies, 1):
        md.append(f"### Case Study {idx}: [{cs['category']}] `{cs['id']}`")
        md.append(f"**Question**: *\"{cs['question']}\"*")
        md.append(f"- **Gold Reference Answer**: {cs['gold_answer']}")
        md.append(f"- **Baseline Answer (VectorDB Only)**: {cs['baseline_answer']}")
        md.append(f"- **Manual Toggle Answer (Source: `{cs['fixing_source']}`)**: {cs['manual_toggle_answer']}")
        md.append(f"- **Outcome**: Fixed by toggling `{cs['fixing_source']}` tool.\n")

    with open(SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"✅ Generated evaluation summary markdown: {SUMMARY_MD_PATH}")
    print(f"✅ Generated flat evaluation data JSON: {SUMMARY_DATA_JSON_PATH}")

if __name__ == "__main__":
    analyze()
