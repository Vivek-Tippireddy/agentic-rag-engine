from typing import List, Dict, Any
import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

class EvidenceFusionPipeline:
    """Merges, re-ranks, and synthesizes evidence from multi-source retrievers or handles conversational mode."""

    def fuse_evidence(self, raw_retrievals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combines evidence snippets from VectorDB, SQL, WebSearch, and KnowledgeGraph."""
        fused_items = []
        local_items = []
        online_items = []
        source_counts = {}

        for item in raw_retrievals:
            source = item.get("source", "Unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            content = item.get("content", "").strip()

            if not content:
                continue

            if source == "VectorDB":
                label = f"[VectorDB: {item.get('company', '')} - {item.get('file_name', '')}]"
            elif source == "SQL_Database":
                label = f"[SQL DB: {item.get('details', {}).get('company', 'Corporate Data')}]"
            elif source == "WebSearch":
                label = f"[Web Search: {item.get('title', 'Web')}]"
            elif source == "KnowledgeGraph":
                label = f"[Knowledge Graph: Entity Relations]"
            else:
                label = f"[{source}]"

            entry = {
                "label": label,
                "source": source,
                "content": content,
                "score": item.get("score", 0.5)
            }

            fused_items.append(entry)
            if source in ["VectorDB", "SQL_Database", "KnowledgeGraph"]:
                local_items.append(entry)
            else:
                online_items.append(entry)

        # Deduplicate snippets
        unique_items = []
        seen_texts = set()
        for item in fused_items:
            preview = item["content"][:100].lower()
            if preview not in seen_texts:
                seen_texts.add(preview)
                unique_items.append(item)

        formatted_context_lines = []
        for idx, item in enumerate(unique_items, 1):
            formatted_context_lines.append(f"Source {idx} {item['label']}:\n{item['content']}\n")

        formatted_context = "\n".join(formatted_context_lines)

        return {
            "formatted_context": formatted_context,
            "items": unique_items,
            "local_items": local_items,
            "online_items": online_items,
            "source_breakdown": source_counts,
            "total_snippets": len(unique_items)
        }

    def generate_candidate_answer(
        self,
        query: str,
        fused_evidence: Dict[str, Any],
        critique_feedback: str = "",
        model_provider: str = "gemini",
        api_key: str = "",
        toggles: Dict[str, bool] = None
    ) -> str:
        """Synthesizes candidate answer from fused evidence or provides conversational response when retrievers are off."""
        
        context_str = fused_evidence.get("formatted_context", "")
        items = fused_evidence.get("items", [])
        active_toggles_count = sum(1 for v in (toggles or {}).values() if v)

        # 1. Conversational Chat Mode (No active data sources selected or simple greeting)
        is_greeting = query.strip().lower() in ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "hi there"]
        no_sources_enabled = active_toggles_count == 0

        if no_sources_enabled or (is_greeting and len(items) == 0):
            effective_gemini_key = api_key or config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
            if model_provider == "gemini" and effective_gemini_key:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        google_api_key=effective_gemini_key,
                        temperature=0.7
                    )
                    prompt = f"You are an intelligent, friendly AI Assistant in Conversational Chat Mode. Respond naturally and helpfully to the user's message: '{query}'. Mention briefly that external RAG retrieval sources are currently turned off."
                    response = llm.invoke(prompt)
                    return response.content
                except Exception as e:
                    print(f"[Gemini Conversational Error] {e}", flush=True)

            # Fallback conversational response
            if is_greeting:
                return "Hello! 👋 How can I help you today? *(Note: All external RAG data sources are currently toggled off in the sidebar, so I am operating in Direct LLM Conversational Mode.)*"
            else:
                return f"**Conversational AI Mode**: Answer for *'{query}'*\n\nAll external retrieval data sources (Vector DB, SQL, Web Search, Knowledge Graph) are currently toggled off in the sidebar. I am responding directly via Conversational LLM reasoning. To pull real-time evidence or search annual reports, toggle on your preferred data sources in the sidebar!"

        # 2. RAG Synthesis Mode (Data sources are active)
        revision_prompt = ""
        if critique_feedback:
            revision_prompt = f"\nIMPORTANT CRITIQUE FEEDBACK FROM VERIFIER NODE:\nPrevious attempt was rejected due to: '{critique_feedback}'. Please revise your response carefully.\n"

        system_prompt = f"""You are an Expert Financial & Business Intelligence AI Assistant.

User Query: "{query}"

{revision_prompt}

EVIDENCE CONTEXT FROM ACTIVE RETRIEVAL SOURCES:
{context_str}

CRITICAL RESPONSE FORMATTING RULES:
1. If local document evidence (Vector DB / SQL DB) is thin or missing direct specific answers for the query, BEGIN your answer with an explicit explanation note:
   "> ℹ️ **Retrieval Status**: The specific details for your query were not directly found in the uploaded local documents (Vector DB / SQL DB). I performed an online web search and compiled the following information regarding your question:"
2. If evidence IS present in local sources or web search, provide a detailed, well-structured financial summary using Markdown sections.
3. Cite sources explicitly using bracketed source tags like [VectorDB], [SQL DB], [Web Search], or [Knowledge Graph].
4. Include exact figures for Net Profit, EBITDA, Revenue, and Growth where present in the evidence.
"""

        effective_gemini_key = api_key or config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")

        if model_provider == "gemini" and effective_gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=effective_gemini_key,
                    temperature=0.2
                )
                response = llm.invoke(system_prompt)
                return response.content
            except Exception as e:
                print(f"[Gemini Synthesis Error] {e}. Using intelligent synthesis fallback.", flush=True)

        # Fallback synthesizer if offline/unauthenticated
        return self._intelligent_fallback_synthesis(query, fused_evidence, critique_feedback)

    def _intelligent_fallback_synthesis(self, query: str, fused_evidence: Dict[str, Any], critique_feedback: str) -> str:
        """Structured synthesis answering user query directly using available evidence."""
        items = fused_evidence.get("items", [])
        local_items = fused_evidence.get("local_items", [])
        online_items = fused_evidence.get("online_items", [])

        sections = []

        if not local_items and online_items:
            sections.append(f"> ℹ️ **Retrieval Status**: The specific figures for *'{query}'* were not found in the uploaded local documents (Vector DB / SQL DB). I searched online across active web sources and found the following information regarding your question:\n")
        elif local_items and online_items:
            sections.append(f"> ℹ️ **Multi-Source Summary**: Information compiled from both local corporate databases and live web search results.\n")
        elif not items:
            return f"Hello! How can I assist you today? *(Operating in Direct Conversational Mode for query: '{query}')*"

        sections.append(f"### Financial Summary for: *{query}*\n")

        for item in items:
            sections.append(f"#### {item['label']}\n{item['content']}\n")

        return "\n".join(sections)
