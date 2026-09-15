import os
import sys
import json
import time

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

import config
from graph import agentic_rag_app

TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SOURCE_MAPPING = {
    "vector_db": "vector_db",
    "sql_api": "sql_api",
    "web_search": "web_search",
    "knowledge_graph": "knowledge_graph"
}

def load_test_set():
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def run_single_question(item, condition):
    q_id = item["id"]
    query = item["question"]
    category = item["category"]
    target_src = item["target_source"]

    # Configure toggles per condition
    if condition == "baseline":
        toggles = {"vector_db": True, "sql_api": False, "web_search": False, "knowledge_graph": False}
        max_iter = 1
        thresh = 0.0  # single pass, no retry cutoff
    elif condition == "auto":
        toggles = {"vector_db": True, "sql_api": True, "web_search": True, "knowledge_graph": True}
        max_iter = 2
        thresh = config.DEFAULT_VERIFIER_THRESHOLD
    elif condition == "manual_toggle":
        toggles = {"vector_db": False, "sql_api": False, "web_search": False, "knowledge_graph": False}
        toggles[target_src] = True
        max_iter = 2
        thresh = config.DEFAULT_VERIFIER_THRESHOLD
    else:
        raise ValueError(f"Unknown condition: {condition}")

    api_key_val = config.HF_TOKEN if config.DEFAULT_MODEL_PROVIDER in ["huggingface", "hf"] else config.GEMINI_API_KEY

    inputs = {
        "query": query,
        "toggles": toggles,
        "api_key": api_key_val,
        "model_provider": config.DEFAULT_MODEL_PROVIDER,
        "threshold": thresh,
        "max_iterations": max_iter,
        "iteration_count": 0,
        "sub_queries": {},
        "reasoning_trace": [],
        "raw_evidence": [],
        "fused_evidence": {},
        "candidate_answer": "",
        "verifier_result": {},
        "verifier_history": [],
        "critique_feedback": "",
        "final_answer": ""
    }

    start_time = time.time()
    res = agentic_rag_app.invoke(inputs)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    history = res.get("verifier_history", [])
    retry_fired = len(history) > 1
    
    initial_score = history[0].get("score", 0.0) if history else 0.0
    final_score = history[-1].get("score", initial_score) if history else 0.0

    # Estimate LLM call count (Reasoning + Synthesis + Verifier per iteration)
    num_iterations = res.get("iteration_count", 1)
    llm_calls = num_iterations * 2 if condition != "baseline" else 1

    sources_queried = [src for src, active in toggles.items() if active]

    verifier_res = res.get("verifier_result", {})
    used_fb = verifier_res.get("used_fallback", True)

    return {
        "id": q_id,
        "category": category,
        "target_source": target_src,
        "condition": condition,
        "question": query,
        "gold_answer": item.get("gold_answer", ""),
        "sources_queried": sources_queried,
        "retry_fired": retry_fired,
        "verifier_score_initial": round(initial_score, 2),
        "verifier_score_final": round(final_score, 2),
        "verifier_used_fallback": used_fb,
        "latency_ms": latency_ms,
        "llm_call_count": llm_calls,
        "candidate_answer": res.get("candidate_answer", ""),
        "final_answer": res.get("final_answer", ""),
        "fused_evidence": res.get("fused_evidence", {})
    }

def main():
    test_items = load_test_set()
    conditions = ["baseline", "auto", "manual_toggle"]

    print(f"Loaded {len(test_items)} evaluation questions.")

    for cond in conditions:
        print(f"\n================ Running Condition: {cond.upper()} ================")
        results = []
        for idx, item in enumerate(test_items, 1):
            print(f"[{cond}] Question {idx}/{len(test_items)}: '{item['question'][:60]}...'")
            out = run_single_question(item, cond)
            print(f"  -> Latency: {out['latency_ms']} ms | Retry Fired: {out['retry_fired']} | Final Score: {out['verifier_score_final']} | Used Fallback: {out['verifier_used_fallback']}")
            results.append(out)
            time.sleep(1.0)

        output_file = os.path.join(RESULTS_DIR, f"{cond}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} results to {output_file}")

if __name__ == "__main__":
    main()
