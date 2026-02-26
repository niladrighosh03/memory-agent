"""
memory_graph_builder.py
-----------------------
Builds an in-memory graph from the latest summarized JSON file.

Node naming convention:
  persona                  — top-level persona node
  session_{year}           — yearly session block
  latent_{year}            — latent persuasion state (aggregated)
  strategy_{year}          — strategy memory stats for that year
  concession_{year}        — concession memory for that year

Edge types  (following the architecture diagram):
  belongs_to    : persona → latent/strategy/concession (latest year)
  derived_from  : latent/strategy/concession_{year} → session_{year}
  temporal_next : session_{year} → session_{year-1}
  similar_to    : latest_session_node → session_2015   (semantic anchor)
"""

import json
import os
import glob
from typing import Dict, Any, List, Tuple


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _latest_json_path(summarized_dir: str) -> str:
    """Return the path of the JSON with the most years (highest file index)."""
    files = glob.glob(os.path.join(summarized_dir, "[0-9]*.json"))
    if not files:
        raise FileNotFoundError(f"No numbered JSON files found in {summarized_dir}")
    return max(files, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))


# ---------------------------------------------------------------------------
# Graph data-structures
# ---------------------------------------------------------------------------

class MemoryNode:
    """A single node in the memory graph."""

    def __init__(self, node_id: str, node_type: str, year: int | None, data: Any):
        self.node_id   = node_id    # e.g. "session_2015", "latent_2018"
        self.node_type = node_type  # "session" | "latent" | "strategy" | "concession" | "persona"
        self.year      = year       # None for the persona node
        self.data      = data       # raw payload from the JSON

    def __repr__(self):
        return f"MemoryNode(id={self.node_id!r}, type={self.node_type!r}, year={self.year})"


class MemoryGraph:
    """
    Directed graph holding all memory nodes and typed edges.

    nodes : dict[node_id, MemoryNode]
    edges : list[(from_id, edge_type, to_id)]
    adj   : dict[node_id, list[(edge_type, to_id)]]   # forward adjacency
    """

    def __init__(self):
        self.nodes: Dict[str, MemoryNode]          = {}
        self.edges: List[Tuple[str, str, str]]     = []
        self.adj:   Dict[str, List[Tuple[str,str]]] = {}
        self.persona_id: str = ""
        self.years:      List[int] = []
        self.entry_node: str = ""   # latest session_<year>

    # ---- mutation helpers ------------------------------------------------

    def _add_node(self, node: MemoryNode):
        self.nodes[node.node_id] = node
        self.adj.setdefault(node.node_id, [])

    def _add_edge(self, from_id: str, edge_type: str, to_id: str):
        self.edges.append((from_id, edge_type, to_id))
        self.adj.setdefault(from_id, []).append((edge_type, to_id))

    # ---- traversal helpers -----------------------------------------------

    def neighbors(self, node_id: str) -> List[Tuple[str, str]]:
        """Return [(edge_type, neighbor_id), …] for a node."""
        return self.adj.get(node_id, [])

    def get_node(self, node_id: str) -> MemoryNode | None:
        return self.nodes.get(node_id)

    def nodes_of_type(self, node_type: str) -> List[MemoryNode]:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def nodes_of_type_and_year(self, node_type: str, year: int) -> MemoryNode | None:
        nid = f"{node_type}_{year}"
        return self.nodes.get(nid)

    def summary(self):
        print(f"MemoryGraph | persona={self.persona_id} | years={self.years}")
        print(f"  nodes : {len(self.nodes)}")
        print(f"  edges : {len(self.edges)}")
        print(f"  entry : {self.entry_node}")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_memory_graph(summarized_dir: str | None = None) -> MemoryGraph:
    """
    Load the latest summarized JSON and construct the MemoryGraph.

    Architecture rules (matching the diagram):

    1. Persona node  ──belongs_to──►  latent/strategy/concession of LATEST year
    2. {latent|strategy|concession}_{year}  ──derived_from──►  session_{year}
    3. session_{year}  ──temporal_next──►  session_{year-1}
    4. latest session  ──similar_to──►  session_2015
       (direct semantic anchor; NOT chained through 2016…latest-1)
    """
    if summarized_dir is None:
        summarized_dir = os.path.join(
            os.path.dirname(__file__), "memory2", "summarized_jsons"
        )

    json_path = _latest_json_path(summarized_dir)
    print(f"[graph_builder] Loading: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    graph = MemoryGraph()
    graph.persona_id = data.get("persona_id", "unknown")

    # ------------------------------------------------------------------
    # 1. Persona node
    # ------------------------------------------------------------------
    persona_node = MemoryNode(
        node_id   = "persona",
        node_type = "persona",
        year      = None,
        data      = {
            "persona_id": graph.persona_id,
            "summary":    data.get("summary", ""),
            "life_events": data.get("life_events", {}),
        },
    )
    graph._add_node(persona_node)

    # ------------------------------------------------------------------
    # 2. One node per session from sessions[]
    #    If a year appears multiple times, suffix with _a, _b, ...
    # ------------------------------------------------------------------
    sessions: List[Dict] = data.get("sessions", [])

    # Build unique session node IDs
    year_counts: Dict[int, int] = {}
    session_node_ids: List[str] = []   # positional, parallel to sessions[]
    for session in sessions:
        year = session["year"]
        year_counts[year] = year_counts.get(year, 0) + 1

    year_seen: Dict[int, int] = {}
    for session in sessions:
        year = session["year"]
        year_seen[year] = year_seen.get(year, 0) + 1
        if year_counts[year] > 1:
            suffix = chr(ord('a') + year_seen[year] - 1)   # 0→'a', 1→'b', …
            nid = f"session_{year}_{suffix}"
        else:
            nid = f"session_{year}"
        session_node_ids.append(nid)
        graph._add_node(MemoryNode(nid, "session", year, session))

    # Unique years (for latent / strategy / concession per-year nodes)
    unique_years = sorted(set(s["year"] for s in sessions))
    graph.years  = unique_years
    earliest_year = unique_years[0]   # 2015
    latest_year   = unique_years[-1]

    # Map year → LAST session node of that year (for temporal_next edges)
    year_to_last_session: Dict[int, str] = {}
    for nid, session in zip(session_node_ids, sessions):
        year_to_last_session[session["year"]] = nid

    # ------------------------------------------------------------------
    # 3. Global memory nodes (latent / strategy / concession)
    #    These carry the AGGREGATED state up to latest_year.
    #    They live under latest_year in the graph.
    # ------------------------------------------------------------------
    latent_data     = data.get("latent_state", {})
    strategy_data   = data.get("strategy_memory", {})
    concession_data = data.get("concession_memory", {})

    for year in unique_years:
        # Per-year strategy is embedded in each session; for the graph we
        # expose the session-level avg_strategy_vector as "strategy_{year}"
        # Use the LAST session block for that year
        year_sessions = [s for s in sessions if s["year"] == year]
        session_block = year_sessions[-1]   # most recent if duplicates exist

        graph._add_node(MemoryNode(
            f"strategy_{year}", "strategy", year,
            session_block.get("avg_strategy_vector", {})
        ))
        graph._add_node(MemoryNode(
            f"concession_{year}", "concession", year,
            {
                "negotiation_moves": session_block.get("negotiation_moves", []),
                "final_outcome":     session_block.get("final_outcome"),
            }
        ))

    # Latent state is global (not stored per-year in the JSON) → attach to latest
    graph._add_node(MemoryNode(
        f"latent_{latest_year}", "latent", latest_year, latent_data
    ))
    # Also store global aggregated strategy/concession under latest_year label
    graph._add_node(MemoryNode(
        f"strategy_agg_{latest_year}", "strategy_agg", latest_year, strategy_data
    ))
    graph._add_node(MemoryNode(
        f"concession_agg_{latest_year}", "concession_agg", latest_year, concession_data
    ))

    # ------------------------------------------------------------------
    # 4. EDGES
    # ------------------------------------------------------------------

    # 4a. belongs_to  :  persona → {latent, strategy_agg, concession_agg} of latest year
    graph._add_edge("persona", "belongs_to", f"latent_{latest_year}")
    graph._add_edge("persona", "belongs_to", f"strategy_agg_{latest_year}")
    graph._add_edge("persona", "belongs_to", f"concession_agg_{latest_year}")

    # 4b. derived_from :  {strategy, concession}_{year} → last session of that year
    for year in unique_years:
        last_session_nid = year_to_last_session[year]
        graph._add_edge(f"strategy_{year}",   "derived_from", last_session_nid)
        graph._add_edge(f"concession_{year}", "derived_from", last_session_nid)
    graph._add_edge(f"latent_{latest_year}", "derived_from", year_to_last_session[latest_year])

    # 4c. temporal_next :  chain all session nodes chronologically
    #     within a year: session_{year}_a → session_{year}_b → …
    #     across years:  last session of year Y → last session of year Y-1
    for i in range(1, len(session_node_ids)):
        graph._add_edge(session_node_ids[i], "temporal_next", session_node_ids[i - 1])

    # 4d. similar_to :  entry node (latest session)  →  session_2015  (ONLY this pair)
    entry_nid    = year_to_last_session[latest_year]
    earliest_nid = year_to_last_session[earliest_year]
    if entry_nid != earliest_nid:
        graph._add_edge(entry_nid, "similar_to", earliest_nid)

    # ------------------------------------------------------------------
    # 5. Entry point  — last session node of the latest year
    # ------------------------------------------------------------------
    graph.entry_node = year_to_last_session[latest_year]

    return graph


# ---------------------------------------------------------------------------
# Convenience: print the graph
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    g = build_memory_graph()
    g.summary()
    print("\nAll nodes:")
    for nid, node in g.nodes.items():
        print(f"  {nid:40s}  type={node.node_type}")
    print("\nAll edges:")
    for (f, e, t) in g.edges:
        print(f"  {f:40s} --[{e}]--> {t}")
