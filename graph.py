import os
import sys
from typing import TypedDict, List, Dict, Any

# Fix Windows console encoding for UTF-8 emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(__file__))

import config
from langgraph.graph import StateGraph, END
from agents.reasoning_agent import ReasoningAgent
from agents.verifier_critic import VerifierCriticNode
from pipeline.evidence_fusion import EvidenceFusionPipeline

from retrievers.vector_store import VectorDBRetriever
from retrievers.sql_engine import SQLEngineRetriever
from retrievers.web_search import WebSearchRetriever
from retrievers.knowledge_graph import KnowledgeGraphRetriever

# Lazy global singletons for efficient caching
_vector_db = None
_sql_engine = None
_web_search = None
_knowledge_graph = None

def get_retrievers():
    global _vector_db, _sql_engine, _web_search, _knowledge_graph
    if _vector_db is None:
        _vector_db = VectorDBRetriever()
    if _sql_engine is None:
        _sql_engine = SQLEngineRetriever()
    if _web_search is None:
        _web_search = WebSearchRetriever()
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraphRetriever()
    return _vector_db, _sql_engine, _web_search, _knowledge_graph

# LangGraph State Schema
class AgenticRAGState(TypedDict):
    query: str
    toggles: Dict[str, bool]
    api_key: str
    model_provider: str
    threshold: float
    max_iterations: int
    iteration_count: int
    sub_queries: Dict[str, str]
    reasoning_trace: List[str]
    raw_evidence: List[Dict[str, Any]]
    fused_evidence: Dict[str, Any]
    candidate_answer: str
    verifier_result: Dict[str, Any]
    verifier_history: List[Dict[str, Any]]
    critique_feedback: str
    final_answer: str

# 1. Reasoning Node
def reasoning_node(state: AgenticRAGState) -> Dict[str, Any]:
    agent = ReasoningAgent()
    iteration = state.get("iteration_count", 0) + 1
    feedback = state.get("critique_feedback", "")
    toggles = state.get("toggles", config.DEFAULT_SOURCE_TOGGLES)

    plan = agent.plan_and_route(
        query=state["query"],
        toggles=toggles,
        critique_feedback=feedback
    )

    trace = state.get("reasoning_trace", [])
    trace.append(f"--- Iteration {iteration} ---")
    trace.extend(plan["reasoning_trace"])

    return {
        "iteration_count": iteration,
        "reasoning_trace": trace,
        "sub_queries": plan["sub_queries"]
    }

# 2. Retrieval Node
def retrieval_node(state: AgenticRAGState) -> Dict[str, Any]:
    vector_db, sql_engine, web_search, knowledge_graph = get_retrievers()
    toggles = state.get("toggles", config.DEFAULT_SOURCE_TOGGLES)
    sub_queries = state.get("sub_queries", {})

    retrieved_items = []

    # Execute active tools
    if toggles.get("vector_db") and "vector_db" in sub_queries:
        res = vector_db.query(sub_queries["vector_db"], top_k=4)
        retrieved_items.extend(res)

    if toggles.get("sql_api") and "sql_api" in sub_queries:
        res = sql_engine.query(sub_queries["sql_api"])
        retrieved_items.extend(res)

    if toggles.get("web_search") and "web_search" in sub_queries:
        res = web_search.query(sub_queries["web_search"])
        retrieved_items.extend(res)

    if toggles.get("knowledge_graph") and "knowledge_graph" in sub_queries:
        res = knowledge_graph.query(sub_queries["knowledge_graph"])
        retrieved_items.extend(res)

    return {"raw_evidence": retrieved_items}

# 3. Evidence Fusion Node
def evidence_fusion_node(state: AgenticRAGState) -> Dict[str, Any]:
    pipeline = EvidenceFusionPipeline()
    raw_evidence = state.get("raw_evidence", [])
    
    fused = pipeline.fuse_evidence(raw_evidence)
    
    candidate = pipeline.generate_candidate_answer(
        query=state["query"],
        fused_evidence=fused,
        critique_feedback=state.get("critique_feedback", ""),
        model_provider=state.get("model_provider", "gemini"),
        api_key=state.get("api_key", ""),
        toggles=state.get("toggles", {})
    )

    return {
        "fused_evidence": fused,
        "candidate_answer": candidate
    }

# 4. Verifier / Critic Node
def verifier_critic_node(state: AgenticRAGState) -> Dict[str, Any]:
    verifier = VerifierCriticNode()
    
    result = verifier.evaluate_answer(
        query=state["query"],
        candidate_answer=state["candidate_answer"],
        fused_evidence=state["fused_evidence"],
        threshold=state.get("threshold", config.DEFAULT_VERIFIER_THRESHOLD),
        model_provider=state.get("model_provider", "gemini"),
        api_key=state.get("api_key", ""),
        toggles=state.get("toggles", {})
    )

    trace = state.get("reasoning_trace", [])
    trace.append(f"🧐 [Verifier Node] Score: {result['score']} / {result['threshold']} | Passed: {result['passed']}")
    trace.append(f"💬 [Critique]: {result['critique']}")

    history = list(state.get("verifier_history", []))
    history.append(result)

    return {
        "verifier_result": result,
        "verifier_history": history,
        "critique_feedback": result["critique"] if not result["passed"] else "",
        "reasoning_trace": trace
    }

# Conditional Router Function
def route_verifier_feedback(state: AgenticRAGState) -> str:
    verifier_res = state.get("verifier_result", {})
    passed = verifier_res.get("passed", False)
    iteration = state.get("iteration_count", 1)
    max_iter = state.get("max_iterations", config.DEFAULT_MAX_RETRY_LOOPS)

    if passed or iteration >= max_iter:
        return "finalize"
    else:
        return "retry_reasoning"

# 5. Final Output Node
def final_output_node(state: AgenticRAGState) -> Dict[str, Any]:
    candidate = state.get("candidate_answer", "")
    verifier_res = state.get("verifier_result", {})
    
    output = candidate
    if not verifier_res.get("passed", False):
        output += f"\n\n---\n⚠️ **Verifier Note**: Max revision attempts ({state.get('max_iterations', 2)}) reached. Last Critique: *{verifier_res.get('critique', '')}*"

    return {"final_answer": output}

# Build LangGraph State Graph
def build_agentic_rag_graph():
    builder = StateGraph(AgenticRAGState)

    builder.add_node("reasoning", reasoning_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("evidence_fusion", evidence_fusion_node)
    builder.add_node("verifier_critic", verifier_critic_node)
    builder.add_node("final_output", final_output_node)

    # Edge Definitions
    builder.set_entry_point("reasoning")
    builder.add_edge("reasoning", "retrieval")
    builder.add_edge("retrieval", "evidence_fusion")
    builder.add_edge("evidence_fusion", "verifier_critic")

    # Conditional Routing from Verifier
    builder.add_conditional_edges(
        "verifier_critic",
        route_verifier_feedback,
        {
            "retry_reasoning": "reasoning",
            "finalize": "final_output"
        }
    )

    builder.add_edge("final_output", END)

    return builder.compile()

# Global compiled graph
agentic_rag_app = build_agentic_rag_graph()

if __name__ == "__main__":
    print("[Graph] Compiling Agentic RAG graph...", flush=True)
    # Test conversational mode with all toggles OFF
    off_toggles = {"vector_db": False, "sql_api": False, "web_search": False, "knowledge_graph": False}
    inputs = {
        "query": "hi",
        "toggles": off_toggles,
        "api_key": "",
        "model_provider": "gemini",
        "threshold": 0.75,
        "max_iterations": 2,
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
    output = agentic_rag_app.invoke(inputs)
    print("\n=== CONVERSATIONAL TEST ANSWER ===")
    print(output["final_answer"])
