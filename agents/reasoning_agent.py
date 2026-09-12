import re
from typing import Dict, Any, List

class ReasoningAgent:
    """Reasoning Agent node responsible for query understanding, sub-query decomposition, and retriever routing."""

    def _extract_keywords(self, query: str) -> str:
        """Extracts clean keywords from natural language query for effective search engine retrieval."""
        q = query.lower()
        
        # Replace fiscal year formats
        q = re.sub(r'financial\s*year\s*2024/2025|2024-25|2024/25', 'FY24 FY25 2024 2025', q)
        q = re.sub(r'financial\s*year\s*2023/2024|2023-24|2023/24', 'FY23 FY24 2023 2024', q)
        q = re.sub(r'financial\s*year', 'FY', q)
        
        # Remove common question stop words & punctuation
        q = re.sub(r'[^\w\s]', ' ', q)
        stop_words = {'what', 'is', 'the', 'are', 'for', 'in', 'of', 'and', 'to', 'a', 'an', 'how', 'much', 'were', 'was', 'did'}
        words = [w for w in q.split() if w not in stop_words and len(w) > 1]
        
        return " ".join(words)

    def plan_and_route(
        self,
        query: str,
        toggles: Dict[str, bool],
        critique_feedback: str = ""
    ) -> Dict[str, Any]:
        """Analyzes query, integrates feedback from prior verifier iterations, and routes sub-queries."""
        
        reasoning_steps = []
        reasoning_steps.append(f"Analyzing query: '{query}'")

        if critique_feedback:
            reasoning_steps.append(f"Received Critic Feedback: '{critique_feedback}'. Refining retrieval strategy...")

        # Determine active sources based on toggles
        active_sources = [source for source, enabled in toggles.items() if enabled]
        reasoning_steps.append(f"Active retrieval sources ({len(active_sources)}): {', '.join(active_sources)}")

        sub_queries = {}
        clean_kw = self._extract_keywords(query)

        if toggles.get("vector_db", False):
            if critique_feedback:
                sub_queries["vector_db"] = f"{clean_kw} financial metrics profit revenue"
            else:
                sub_queries["vector_db"] = clean_kw or query
            reasoning_steps.append(f"VectorDB sub-query -> '{sub_queries['vector_db']}'")

        if toggles.get("sql_api", False):
            sub_queries["sql_api"] = query
            reasoning_steps.append(f"SQL Engine sub-query -> Search structured database for company & metrics in '{query}'")

        if toggles.get("web_search", False):
            # Clean keywords for DuckDuckGo/Web Search to ensure high recall
            sub_queries["web_search"] = f"{clean_kw} financial results"
            reasoning_steps.append(f"Web Search sub-query -> '{sub_queries['web_search']}'")

        if toggles.get("knowledge_graph", False):
            sub_queries["knowledge_graph"] = query
            reasoning_steps.append(f"Knowledge Graph sub-query -> Traversal for entities in '{query}'")

        return {
            "query": query,
            "active_sources": active_sources,
            "sub_queries": sub_queries,
            "reasoning_trace": reasoning_steps
        }
