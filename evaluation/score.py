import os
import sys
import json
import csv
import re
from typing import Dict, Any

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

import config

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SCORES_CSV_PATH = os.path.join(RESULTS_DIR, "scores.csv")

LLM_JUDGE_PROMPT_TEMPLATE = """You are an Expert AI Evaluation Judge scoring an Agentic RAG System answer.

QUESTION:
{question}

GOLD REFERENCE ANSWER:
{gold_answer}

RETRIEVED EVIDENCE CONTEXT:
{evidence_context}

GENERATED SYSTEM ANSWER:
{generated_answer}

EVALUATION TASK:
Score the GENERATED SYSTEM ANSWER on two metrics from 1 to 5:

1. Faithfulness / Groundedness (1 to 5):
   - Score 5: Answer is fully grounded in the retrieved evidence with zero ungrounded claims.
   - Score 3-4: Answer is mostly grounded with minor ungrounded additions.
   - Score 1-2: Answer contains severe hallucinations or states "No relevant information found" when evidence exists.

2. Answer Correctness vs Gold Reference (1 to 5):
   - Score 5: Answer contains all key facts/figures present in the Gold Reference Answer.
   - Score 3-4: Answer contains partial facts or correct direction but misses specific numbers.
   - Score 1-2: Answer is incorrect, states no information found, or contradicts the Gold Reference Answer.

Respond ONLY in valid JSON format:
{{
  "groundedness_score": 5,
  "correctness_score": 5,
  "reasoning": "Short audit reasoning here."
}}
"""

def print_judge_prompt_template():
    print("\n--- AUDITABLE LLM JUDGE PROMPT TEMPLATE ---")
    print(LLM_JUDGE_PROMPT_TEMPLATE)
    print("-------------------------------------------\n")

def score_with_llm_judge(question, gold_answer, evidence_context, generated_answer, api_key="", model_provider=None):
    provider = model_provider or config.DEFAULT_MODEL_PROVIDER
    effective_gemini_key = api_key or config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
    effective_hf_token = api_key or config.HF_TOKEN or os.getenv("HF_TOKEN", "")

    prompt = LLM_JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        evidence_context=evidence_context[:1500],
        generated_answer=generated_answer[:1500]
    )

    content_text = None

    if provider == "gemini" and effective_gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=config.DEFAULT_MODEL_NAME,
                google_api_key=effective_gemini_key,
                temperature=0.0
            )
            res = llm.invoke(prompt)
            content_text = res.content
            if isinstance(content_text, list):
                content_text = "".join([p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in content_text])
            elif not isinstance(content_text, str):
                content_text = str(content_text)
        except Exception as e:
            import traceback
            print(f"[LLM Judge Gemini Warning] Full Exception Traceback:\n{traceback.format_exc()}", flush=True)

    elif provider in ["huggingface", "hf"] and effective_hf_token:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(model=config.DEFAULT_HF_MODEL, token=effective_hf_token)
            res = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000
            )
            content_text = res.choices[0].message.content
        except Exception as e:
            import traceback
            print(f"[LLM Judge HuggingFace Warning] Full Exception Traceback:\n{traceback.format_exc()}", flush=True)

    if content_text:
        try:
            match = re.search(r"\{.*\}", content_text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return (
                    int(data.get("groundedness_score", 3)),
                    int(data.get("correctness_score", 3)),
                    data.get("reasoning", "LLM judge evaluated."),
                    False
                )
        except Exception as e:
            import traceback
            print(f"[LLM Judge Parsing Warning] Full Traceback:\n{traceback.format_exc()}", flush=True)

    # Rule-based fallback rubric if LLM is unavailable
    g_score, c_score, reasoning = _rule_based_judge_score(question, gold_answer, evidence_context, generated_answer)
    return g_score, c_score, reasoning, True

def _rule_based_judge_score(question, gold_answer, evidence_context, generated_answer):
    if "No relevant information found" in generated_answer or not generated_answer.strip():
        return (1, 1, "Answer stated no information found or was empty.")

    # Check keyword overlap with gold answer
    gold_words = [w.lower() for w in re.sub(r'[^\w\s]', '', gold_answer).split() if len(w) > 3]
    gen_lower = generated_answer.lower()
    
    matches = sum(1 for w in gold_words if w in gen_lower)
    ratio = (matches / len(gold_words)) if gold_words else 0.5

    if ratio >= 0.6:
        correctness = 5
        groundedness = 5
    elif ratio >= 0.3:
        correctness = 3
        groundedness = 4
    else:
        correctness = 2
        groundedness = 3

    return (groundedness, correctness, f"Rule-based overlap score (Ratio: {ratio:.2f})")

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print_judge_prompt_template()

    conditions = ["baseline", "auto", "manual_toggle"]
    all_scores = []

    fieldnames = [
        "id", "category", "target_source", "condition", "question",
        "retry_fired", "verifier_score_initial", "verifier_score_final",
        "latency_ms", "llm_call_count", "groundedness_score", "correctness_score",
        "used_fallback", "judge_reasoning"
    ]

    for cond in conditions:
        json_path = os.path.join(RESULTS_DIR, f"{cond}.json")
        if not os.path.exists(json_path):
            print(f"[Warning] Result file {json_path} not found. Skipping {cond}.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            runs = json.load(f)

        print(f"Scoring condition '{cond}' ({len(runs)} questions)...")
        for item in runs:
            fused = item.get("fused_evidence", {})
            evidence_context = fused.get("formatted_context", "")

            api_key_val = config.HF_TOKEN if config.DEFAULT_MODEL_PROVIDER in ["huggingface", "hf"] else config.GEMINI_API_KEY

            g_score, c_score, reasoning, used_fb = score_with_llm_judge(
                question=item["question"],
                gold_answer=item.get("gold_answer", ""),
                evidence_context=evidence_context,
                generated_answer=item.get("final_answer", ""),
                api_key=api_key_val,
                model_provider=config.DEFAULT_MODEL_PROVIDER
            )

            record = {
                "id": item["id"],
                "category": item["category"],
                "target_source": item["target_source"],
                "condition": item["condition"],
                "question": item["question"],
                "retry_fired": item["retry_fired"],
                "verifier_score_initial": item["verifier_score_initial"],
                "verifier_score_final": item["verifier_score_final"],
                "latency_ms": item["latency_ms"],
                "llm_call_count": item["llm_call_count"],
                "groundedness_score": g_score,
                "correctness_score": c_score,
                "used_fallback": used_fb,
                "judge_reasoning": reasoning
            }
            all_scores.append(record)
            import time
            time.sleep(1.0)

    with open(SCORES_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_scores)

    print(f"✅ Successfully written per-question scores to {SCORES_CSV_PATH}")

if __name__ == "__main__":
    main()
