"""Workflow graph definition for the RAG pipeline."""


def build_graph():
    nodes = [
        {"id": "start", "label": "START"},
        {"id": "intent", "label": "Intent"},
        {"id": "retrieval", "label": "Retrieval"},
        {"id": "policy", "label": "Policy"},
        {"id": "response", "label": "Response"},
        {"id": "end", "label": "END"},
    ]

    edges = [
        {"from": "start", "to": "intent"},
        {"from": "intent", "to": "retrieval"},
        {"from": "retrieval", "to": "policy"},
        {"from": "policy", "to": "response"},
        {"from": "response", "to": "end"},
    ]

    return {"nodes": nodes, "edges": edges}


if __name__ == "__main__":
    graph = build_graph()
    print(graph)
