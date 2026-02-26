"""
langgraph_inference.py
----------------------
Main LangGraph pipeline for memory-aware query answering.

Graph flow:
  START
    └─► route_query          (classify query → node_types + nodes_to_traverse)
          └─► traverse_graph  (walk memory graph, collect context from relevant nodes)
                └─► generate_response  (Azure OpenAI synthesises answer)
                      └─► END

Entry point to the MEMORY GRAPH is always the latest session node.
The traversal follows the edges defined in memory_graph_builder.py:
  - belongs_to    : persona → latent / strategy_agg / concession_agg (latest year)
  - derived_from  : latent/strategy/concession_{year} → session_{year}
  - temporal_next : session_{year} → session_{year-1}
  - similar_to    : session_{latest} → session_2015  (only this pair)
"""

import json
import os
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from openai import AzureOpenAI

from memory_graph_builder import build_memory_graph, MemoryGraph, MemoryNode
from query_router import route_query, QueryRoute

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "memory2", ".env"))

AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_MODEL              = os.getenv("AZURE_MODEL")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

_client = AzureOpenAI(
    api_key        = AZURE_OPENAI_API_KEY,
    api_version    = AZURE_OPENAI_API_VERSION,
    azure_endpoint = AZURE_OPENAI_ENDPOINT,
)

# ---------------------------------------------------------------------------
# Build memory graph (once at module load)
# ---------------------------------------------------------------------------

SUMMARIZED_DIR = os.path.join(os.path.dirname(__file__), "memory2", "summarized_jsons")
MEMORY_GRAPH: MemoryGraph = build_memory_graph(SUMMARIZED_DIR)

ENTRY_YEAR    = MEMORY_GRAPH.years[-1]
EARLIEST_YEAR = MEMORY_GRAPH.years[0]

print(f"[inference] Graph loaded | entry={MEMORY_GRAPH.entry_node} | years={MEMORY_GRAPH.years}")

# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    query:              str
    route:              QueryRoute | None
    nodes_to_traverse:  List[str]              # concrete node IDs
    collected_context:  Dict[str, Any]         # node_id → data
    traversal_log:      List[str]              # human-readable log of what was visited
    final_response:     str


# ---------------------------------------------------------------------------
# Node 1 — route_query
# ---------------------------------------------------------------------------

def node_route_query(state: GraphState) -> GraphState:
    """Classify the query and determine which memory nodes to visit."""
    query = state["query"]
    print(f"\n[route_query] Query: {query!r}")

    route = route_query(
        query          = query,
        available_years = MEMORY_GRAPH.years,
        entry_year      = ENTRY_YEAR,
        earliest_year   = EARLIEST_YEAR,
    )

    print(f"[route_query] node_types={route.node_types}  year_range={route.year_range}")
    print(f"[route_query] nodes_to_traverse={route.nodes_to_traverse}")
    print(f"[route_query] reasoning: {route.reasoning}")

    # Always start traversal from entry node (latest session)
    nodes = [MEMORY_GRAPH.entry_node]
    for n in route.nodes_to_traverse:
        if n not in nodes and n in MEMORY_GRAPH.nodes:
            nodes.append(n)

    return {
        **state,
        "route":             route,
        "nodes_to_traverse": nodes,
        "traversal_log":     [f"Query routed → types={route.node_types}, year_range={route.year_range}, reason={route.reasoning}"],
    }


# ---------------------------------------------------------------------------
# Node 2 — traverse_graph
# ---------------------------------------------------------------------------

def node_traverse_graph(state: GraphState) -> GraphState:
    """
    Walk the memory graph collecting data from all required nodes.
    Entry is always the latest session node; from there we follow edges
    to gather the context requested by the router.
    """
    nodes_to_visit    = state["nodes_to_traverse"]
    collected_context: Dict[str, Any] = {}
    traversal_log     = list(state.get("traversal_log", []))

    # BFS/targeted traversal: visit each requested node + follow its edges
    visited  = set()
    queue    = list(nodes_to_visit)       # start from pre-resolved nodes
    
    # Add 'similar_to' anchor: if session_<latest> is in the visit list,
    # also add session_<earliest> automatically (the similar_to edge target)
    entry_node_id = MEMORY_GRAPH.entry_node
    if entry_node_id in queue and f"session_{EARLIEST_YEAR}" not in queue:
        # Include it because of the explicit similar_to edge
        queue.append(f"session_{EARLIEST_YEAR}")

    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)

        node = MEMORY_GRAPH.get_node(node_id)
        if node is None:
            traversal_log.append(f"  ⚠ Node not found: {node_id}")
            continue

        collected_context[node_id] = {
            "node_type": node.node_type,
            "year":      node.year,
            "data":      node.data,
        }
        traversal_log.append(f"  ✓ Visited: {node_id} (type={node.node_type}, year={node.year})")

        # Follow edges that lead to nodes in our requested set
        for (edge_type, neighbor_id) in MEMORY_GRAPH.neighbors(node_id):
            if neighbor_id not in visited and neighbor_id in nodes_to_visit:
                traversal_log.append(f"    → following [{edge_type}] edge to {neighbor_id}")
                queue.append(neighbor_id)

    print("\n[traverse_graph] Traversal log:")
    for line in traversal_log:
        print(line)

    return {
        **state,
        "collected_context": collected_context,
        "traversal_log":     traversal_log,
    }


# ---------------------------------------------------------------------------
# Node 3 — generate_response
# ---------------------------------------------------------------------------

def node_generate_response(state: GraphState) -> GraphState:
    """Synthesise a final answer from the collected context using Azure OpenAI."""
    query   = state["query"]
    context = state["collected_context"]
    route   = state["route"]

    # Compact representation for the prompt
    context_summary = []
    for node_id, payload in context.items():
        context_summary.append({
            "node":      node_id,
            "node_type": payload["node_type"],
            "year":      payload["year"],
            "data":      payload["data"],
        })

    system_msg = (
        "You are a memory-aware negotiation assistant. "
        "You receive structured memory nodes (session history, latent psychological states, "
        "strategy performance, and concession patterns) of a persona across multiple years. "
        "Use ONLY the provided context to answer the query concisely and accurately. "
        "Cite which years or memory types your answer draws from."
    )

    user_msg = (
        f"Query: {query}\n\n"
        f"Traversed memory nodes ({len(context_summary)} nodes):\n"
        f"{json.dumps(context_summary, indent=2, default=str)}"
    )

    response = _client.chat.completions.create(
        model       = AZURE_MODEL,
        messages    = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens  = 400,
        temperature = 0.3,
    )

    answer = response.choices[0].message.content.strip()
    print(f"\n[generate_response] Answer:\n{answer}")

    return {
        **state,
        "final_response": answer,
    }


# ---------------------------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------------------------

def build_inference_graph() -> Any:
    """Construct and compile the LangGraph StateGraph."""
    builder = StateGraph(GraphState)

    builder.add_node("route_query",       node_route_query)
    builder.add_node("traverse_graph",    node_traverse_graph)
    builder.add_node("generate_response", node_generate_response)

    builder.add_edge(START,              "route_query")
    builder.add_edge("route_query",      "traverse_graph")
    builder.add_edge("traverse_graph",   "generate_response")
    builder.add_edge("generate_response", END)

    return builder.compile()


INFERENCE_GRAPH = build_inference_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_query(query: str) -> Dict[str, Any]:
    """
    Run a natural-language query through the memory graph.

    Returns
    -------
    dict with keys:
      - final_response   : str     — the answer
      - nodes_to_traverse: list    — node IDs the router selected
      - traversal_log    : list    — step-by-step log of visited nodes
      - route            : QueryRoute — routing metadata
    """
    initial_state: GraphState = {
        "query":             query,
        "route":             None,
        "nodes_to_traverse": [],
        "collected_context": {},
        "traversal_log":     [],
        "final_response":    "",
    }
    result = INFERENCE_GRAPH.invoke(initial_state)
    return {
        "final_response":    result["final_response"],
        "nodes_to_traverse": result["nodes_to_traverse"],
        "traversal_log":     result["traversal_log"],
        "route":             result["route"],
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_queries = [
        "What was the user's trust level in 2015?",
        "How has the negotiation strategy changed over the years?",
        "What is the current best strategy to use with this persona?",
        "Show me the concession pattern history",
    ]

    for q in demo_queries:
        print("\n" + "=" * 70)
        result = run_query(q)

        print(f"\n📌 QUERY          : {q}")
        print(f"📡 NODES ROUTED   : {result['nodes_to_traverse']}")
        print(f"🗺  TRAVERSAL LOG  :")
        for line in result["traversal_log"]:
            print(f"   {line}")
        print(f"\n💬 ANSWER:\n{result['final_response']}")
