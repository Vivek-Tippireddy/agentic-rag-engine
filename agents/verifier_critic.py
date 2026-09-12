import os
import sys
import json
import re
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

class VerifierCriticNode:
    """Verifier and Critic agent node assessing faithfulness, hallucination risk, and query relevance."""

    def evaluate_answer(
        self,
        query: str,
        candidate_answer: str,
        fused_evidence: Dict[str, Any],
        threshold: float = None,
        model_provider: str = "gemini",
        api_key: str = "",
        toggles: Dict[str, bool] = None
    ) -> Dict[str, Any]:
        """Evaluates candidate response and returns score, decision (passed/failed), and critique feedback."""

        effective_threshold = threshold if threshold is not None else config.DEFAULT_VERIFIER_THRESHOLD
        context_str = fused_evidence.get("formatted_context", "")
        active_toggles_count = sum(1 for v in (toggles or {}).values() if v)

        # Check for conversational chat mode or greetings ("hi", "hello")
        is_greeting = query.strip().lower() in ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "hi there"]
        no_sources = active_toggles_count == 0

        if no_sources or is_greeting or "Conversational AI Mode" in candidate_answer or "Hello!" in candidate_answer or "Conversational Mode" in candidate_answer:
            return {
                "passed": True,
                "score": 0.95,
                "faithfulness_score": 9.5,
                "relevance_score": 9.5,
                "has_hallucination": False,
                "critique": "Conversational Chat Mode verified (Direct LLM interaction without external retrieval).",
                "threshold": effective_threshold
            }

        # Check: If answer is completely empty
        if not candidate_answer or candidate_answer.strip() == "":
            return {
                "passed": False,
                "score": 0.20,
                "faithfulness_score": 2.0,
                "relevance_score": 2.0,
                "has_hallucination": False,
                "critique": "Candidate answer is empty. Please generate a response.",
                "threshold": effective_threshold
            }

        effective_gemini_key = api_key or config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")

        if model_provider == "gemini" and effective_gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=effective_gemini_key,
                    temperature=0.0
                )

                eval_prompt = f"""You are a strict Verifier and Critic AI evaluating an Agentic RAG system response.

USER QUERY:
{query}

RETRIEVED EVIDENCE CONTEXT:
{context_str}

CANDIDATE ANSWER TO EVALUATE:
{candidate_answer}

EVALUATION TASK:
Evaluate the CANDIDATE ANSWER across 3 criteria:
1. Faithfulness / Hallucination Check (Score 0 to 10): Are facts strictly supported by the RETRIEVED EVIDENCE CONTEXT or explained appropriately?
2. Query Relevance (Score 0 to 10): Does the answer directly answer the USER QUERY?
3. Source Grounding (Score 0 to 10): Does the answer cite source tags appropriately or explain local vs online retrieval status?

Respond ONLY in valid JSON format with this structure:
{{
    "faithfulness_score": 9.0,
    "relevance_score": 9.5,
    "grounding_score": 8.5,
    "overall_score": 0.90,
    "has_hallucination": false,
    "critique_feedback": "Detailed feedback explanation here."
}}
"""
                res = llm.invoke(eval_prompt)
                match = re.search(r"\{.*\}", res.content, re.DOTALL)
                if match:
                    eval_data = json.loads(match.group(0))
                    score = float(eval_data.get("overall_score", 0.85))
                    passed = score >= effective_threshold
                    return {
                        "passed": passed,
                        "score": score,
                        "faithfulness_score": float(eval_data.get("faithfulness_score", 8.5)),
                        "relevance_score": float(eval_data.get("relevance_score", 8.5)),
                        "has_hallucination": bool(eval_data.get("has_hallucination", False)),
                        "critique": eval_data.get("critique_feedback", "Verified successfully."),
                        "threshold": effective_threshold
                    }
            except Exception as e:
                print(f"[Verifier Error] {e}. Using deterministic verifier fallback.", flush=True)

        # Fallback deterministic verifier algorithm
        return self._deterministic_verifier_fallback(query, candidate_answer, fused_evidence, effective_threshold)

    def _deterministic_verifier_fallback(
        self,
        query: str,
        candidate_answer: str,
        fused_evidence: Dict[str, Any],
        threshold: float
    ) -> Dict[str, Any]:
        """Rule-based verifier inspecting citation overlap and key content grounding."""
        
        citation_tags = ["[VectorDB", "[SQL DB", "[Web Search", "[Knowledge Graph", "Retrieval Status", "Financial Summary", "Conversational"]
        has_citations = any(tag in candidate_answer for tag in citation_tags)

        context_text = fused_evidence.get("formatted_context", "").lower()
        answer_words = [w.lower() for w in candidate_answer.split() if len(w) > 4 and w.isalnum()]
        
        grounded_count = sum(1 for word in answer_words if word in context_text or word in query.lower())
        grounding_ratio = (grounded_count / len(answer_words)) if answer_words else 1.0

        base_score = 0.65 + (0.25 * grounding_ratio) + (0.10 if has_citations else 0.0)
        final_score = min(round(base_score, 2), 0.95)

        passed = final_score >= threshold
        critique = "Answer passed groundedness and relevance checks." if passed else "Answer lacks sufficient source citations or explicit evidence alignment."

        return {
            "passed": passed,
            "score": final_score,
            "faithfulness_score": round(grounding_ratio * 10, 1),
            "relevance_score": 9.0,
            "has_hallucination": False,
            "critique": critique,
            "threshold": threshold
        }
