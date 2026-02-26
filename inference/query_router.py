"""
query_router.py  (keyword / rule-based — no OpenAI call)
---------------------------------------------------------
Classifies a user query entirely using keyword matching to determine:
  - which node types to traverse  (session, latent, strategy, concession)
  - which year range to target    (latest, all_years, earliest, or a specific YYYY)

Returns a QueryRoute object consumed by the LangGraph pipeline.
No API key or network call is needed here.
"""

import re
from dataclasses import dataclass, field
from typing import List, Set


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class QueryRoute:
    node_types:        List[str]        # which node types to visit
    year_range:        str              # "latest" | "all_years" | "earliest" | "YYYY"
    reasoning:         str = ""         # human-readable explanation
    nodes_to_traverse: List[str] = field(default_factory=list)  # concrete node IDs


# ---------------------------------------------------------------------------
# Keyword rule tables
# ---------------------------------------------------------------------------

# Maps each node type to a list of trigger keywords / phrases
_NODE_TYPE_KEYWORDS: dict[str, list[str]] = {
    "latent": [
        "trust", "commitment", "price sensitivity", "emotional response",
        "logical response", "credibility response", "concession expectation",
        "psychological", "latent", "internal state", "persuasion state",
        "belief", "attitude",
    ],
    "strategy": [
        "strategy", "strategies", "tactic", "approach", "persuasion",
        "logical", "emotional", "credibility", "personali", "technique",
        "method", "best strategy", "what strategy", "which strategy",
        "how to convince", "how to persuade", "delta readiness",
        "success rate", "strategy change", "strategy shift",
    ],
    "concession": [
        "concession", "offer", "counter offer", "price", "request",
        "negotiation", "deal", "amount", "final offer", "user request",
        "agent offer", "payment", "quote", "acceptance", "reject",
        "outcome", "history", "pattern",
    ],
    "session": [
        "session", "year", "conversation", "interaction", "meeting",
        "negotiation round", "intent", "decision readiness", "outcome",
        "information seeking", "emotional valence",
    ],
}

# If the query contains any of these, override to "all_years"
_ALL_YEARS_KEYWORDS = [
    "over the years", "across the years", "all years", "history",
    "changed", "change", "evolution", "trend", "growth", "progression",
    "from 2015", "since 2015", "over time", "year by year", "compare",
    "comparison", "each year", "every year",
]

# Override to "latest" / "earliest"
_LATEST_KEYWORDS  = ["current", "latest", "now", "today", "recent", "right now", "best strategy"]
_EARLIEST_KEYWORDS = ["first", "initial", "earliest", "oldest", "start", "beginning", "2015"]


# ---------------------------------------------------------------------------
# Core routing logic
# ---------------------------------------------------------------------------

def _match_node_types(query_lower: str) -> Set[str]:
    """Return the set of node types whose keywords appear in the query."""
    matched: Set[str] = set()
    for node_type, keywords in _NODE_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in query_lower:
                matched.add(node_type)
                break
    # Default: if nothing matched, visit everything
    return matched if matched else {"session", "latent", "strategy", "concession"}


def _match_year_range(query_lower: str, available_years: List[int]) -> str:
    """Return year range string from the query."""
    # Check for explicit 4-digit year mention
    found_years = re.findall(r"\b(20\d{2})\b", query_lower)
    if found_years:
        yr = int(found_years[0])
        if yr in available_years:
            return str(yr)

    # Check multi-year keywords (check before latest/earliest)
    for kw in _ALL_YEARS_KEYWORDS:
        if kw in query_lower:
            return "all_years"

    # Check for latest
    for kw in _LATEST_KEYWORDS:
        if kw in query_lower:
            return "latest"

    # Check for earliest
    for kw in _EARLIEST_KEYWORDS:
        if kw in query_lower:
            return "earliest"

    # Default: latest
    return "latest"


def _build_reasoning(node_types: Set[str], year_range: str) -> str:
    return (
        f"Keyword match → node_types={sorted(node_types)}, year_range={year_range!r}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route_query(
    query: str,
    available_years: List[int],
    entry_year: int,
    earliest_year: int,
) -> QueryRoute:
    """
    Classify the query and resolve concrete node IDs to traverse.

    No API calls — pure keyword matching.

    Parameters
    ----------
    query           : user's natural language question
    available_years : sorted list of years in the graph
    entry_year      : latest year (graph entry point)
    earliest_year   : first year (2015)
    """
    q = query.lower()

    node_types = _match_node_types(q)
    year_range = _match_year_range(q, available_years)
    reasoning  = _build_reasoning(node_types, year_range)

    # Resolve target years
    if year_range == "latest":
        target_years = [entry_year]
    elif year_range == "earliest":
        target_years = [earliest_year]
    elif year_range == "all_years":
        target_years = available_years
    elif year_range.isdigit() and int(year_range) in available_years:
        target_years = [int(year_range)]
    else:
        target_years = [entry_year]

    # Build concrete node ID list
    nodes_to_traverse: List[str] = []
    for year in target_years:
        if "session" in node_types:
            nodes_to_traverse.append(f"session_{year}")
        if "strategy" in node_types:
            nodes_to_traverse.append(f"strategy_{year}")
        if "concession" in node_types:
            nodes_to_traverse.append(f"concession_{year}")

    # Latent is attached to the latest year
    if "latent" in node_types:
        nodes_to_traverse.append(f"latent_{entry_year}")

    # Always include aggregated memory for full context
    nodes_to_traverse.append(f"strategy_agg_{entry_year}")
    nodes_to_traverse.append(f"concession_agg_{entry_year}")

    # Deduplicate while preserving order
    seen: Set[str] = set()
    nodes_to_traverse = [n for n in nodes_to_traverse if not (n in seen or seen.add(n))]

    return QueryRoute(
        node_types        = sorted(node_types),
        year_range        = year_range,
        reasoning         = reasoning,
        nodes_to_traverse = nodes_to_traverse,
    )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_years = list(range(2015, 2026))
    queries = [
        "What was the user's trust level in 2015?",
        "How has the negotiation strategy changed over the years?",
        "What is the current best strategy to use?",
        "Show me the concession pattern history",
        "What is the latest latent state?",
        "Compare prices from 2018",
    ]
    for q in queries:
        r = route_query(q, sample_years, 2025, 2015)
        print(f"\nQuery   : {q}")
        print(f"Types   : {r.node_types}  |  Year: {r.year_range}")
        print(f"Reason  : {r.reasoning}")
        print(f"Nodes   : {r.nodes_to_traverse}")
