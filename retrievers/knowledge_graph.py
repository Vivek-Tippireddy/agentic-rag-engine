import networkx as nx
from typing import List, Dict, Any

class KnowledgeGraphRetriever:
    """NetworkX-based Entity-Relationship Knowledge Graph retriever."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_knowledge_graph()

    def _build_knowledge_graph(self):
        """Constructs knowledge graph nodes and directed relationship edges."""
        # Tata Motors Subgraph
        self.graph.add_edge("Tata Motors", "Jaguar Land Rover (JLR)", relation="owns_subsidiary", share="100%")
        self.graph.add_edge("Tata Motors", "Tata Passenger Electric Mobility (TPEM)", relation="owns_subsidiary", share="100%")
        self.graph.add_edge("TPEM", "TPG Rise Climate", relation="invested_by", commitment="$1 Billion")
        self.graph.add_edge("Jaguar Land Rover (JLR)", "Reimagine Strategy", relation="executes_strategy", investment="GBP 15 Billion")
        self.graph.add_edge("Tata Motors", "Commercial Vehicles (CV)", relation="operates_division")
        self.graph.add_edge("Commercial Vehicles (CV)", "Hydrogen FCEV & LNG", relation="developing_tech")
        self.graph.add_edge("Tata Motors", "Corporate De-merger", relation="approved_strategic_action", timeline="2024-2025")

        # Reliance Industries Subgraph
        self.graph.add_edge("Reliance Industries", "Jio Platforms", relation="owns_subsidiary", sector="Telecom")
        self.graph.add_edge("Reliance Industries", "Reliance Retail", relation="owns_subsidiary", stores="18,000+")
        self.graph.add_edge("Reliance Industries", "Oil to Chemicals (O2C)", relation="operates_division")
        self.graph.add_edge("Reliance Industries", "Dhirubhai Ambani Green Energy Giga Complex", relation="building_facility", location="Jamnagar")
        self.graph.add_edge("Reliance Industries", "Green Hydrogen & Solar Gigafactories", relation="target_investment", commitment="Rs 75,000 Crore")

    def query(self, query_text: str) -> List[Dict[str, Any]]:
        """Finds subgraphs & connected nodes related to query terms."""
        query_lower = query_text.lower()
        matched_nodes = []

        for node in self.graph.nodes():
            if node.lower() in query_lower or any(word in node.lower() for word in query_lower.split() if len(word) > 3):
                matched_nodes.append(node)

        if not matched_nodes:
            # Default fallback to root corporate nodes
            matched_nodes = ["Tata Motors", "Reliance Industries"]

        graph_facts = []
        for node in matched_nodes:
            # Outgoing relations
            for target in self.graph.successors(node):
                edge_data = self.graph.get_edge_data(node, target)
                rel = edge_data.get("relation", "connected_to")
                details = ", ".join([f"{k}={v}" for k, v in edge_data.items() if k != "relation"])
                detail_str = f" ({details})" if details else ""
                graph_facts.append(f"{node} -> [{rel}] -> {target}{detail_str}")

            # Incoming relations
            for source in self.graph.predecessors(node):
                edge_data = self.graph.get_edge_data(source, node)
                rel = edge_data.get("relation", "connected_to")
                details = ", ".join([f"{k}={v}" for k, v in edge_data.items() if k != "relation"])
                detail_str = f" ({details})" if details else ""
                graph_facts.append(f"{source} -> [{rel}] -> {node}{detail_str}")

        # Deduplicate
        unique_facts = list(set(graph_facts))

        return [{
            "source": "KnowledgeGraph",
            "content": "\n".join(unique_facts) if unique_facts else "No entity matches in Knowledge Graph.",
            "matched_entities": matched_nodes,
            "score": 0.90
        }]

if __name__ == "__main__":
    kg = KnowledgeGraphRetriever()
    print(kg.query("Tell me about JLR strategy and Tata EV"))
