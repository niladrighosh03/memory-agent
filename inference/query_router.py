"""
query_router.py  (LLM-based semantic router)
--------------------------------------------
Uses Azure OpenAI to understand the semantic intent of ANY user query
and decide which memory nodes to traverse.

Key difference from keyword approach:
  - The LLM receives the persona's memory SUMMARY as context
  - It understands domain intent (e.g. "I want to buy insurance" →
    needs strategy + latent + concession to handle a live negotiation)
  - Returns structured JSON with node_types and year_range
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "memory2", ".env"))

_client = AzureOpenAI(
    api_key        = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version    = os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
)
AZURE_MODEL = os.getenv("AZURE_MODEL", "gpt-4.1")


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class QueryRoute:
    node_types:        List[str]
    year_range:        str          # "latest" | "all_years" | "earliest" | "YYYY"
    reasoning:         str = ""
    nodes_to_traverse: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# System prompt — includes memory schema knowledge
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are the routing brain of a memory-aware negotiation AI agent.

The agent stores a persona's negotiation history as a graph with these node types:

  session     → Per-year conversation summary: intent vectors (trust, price_sensitivity,
                decision_readiness, emotional_valence, information_seeking), negotiation moves
                (user_request, agent_offer), and final_outcome (Accept/Reject).

  latent      → Current latent psychological state: trust, commitment, price_sensitivity,
                decision_readiness_end, emotional_valence_end. Reflects WHO the person is RIGHT NOW.

  strategy    → Per-year usage of persuasion strategies: credibility, emotional_appeal,
                logical_argument, personalization scores.

  strategy_agg → Lifetime aggregated strategy stats: which strategy had highest success rate,
                 delta_readiness impact per strategy.

  concession  → Per-year price negotiation history: what the user requested, what the agent
                offered, and the final outcome.

  concession_agg → Aggregated: average prices, acceptance rate, typical concession range.

Given a user query (which may be a negotiation utterance, a question, or an intent),
you must output ONLY a JSON object with these EXACT keys:

{
  "node_types": ["session" | "latent" | "strategy" | "concession" | "all"],
  "year_range": "latest" | "all_years" | "earliest" | "<YYYY>",
  "reasoning":  "<one sentence explaining why these nodes are needed>"
}

Rules for node_type selection:
  - Negotiation utterances (buying intent, price questions, offers, counter-offers):
    → ["latent", "strategy", "concession"]  — need current state + best approach + price history
  - Questions about trust, commitment, psychological state, readiness:
    → ["latent"]
  - Questions about past prices, offers, acceptance/rejection history:
    → ["concession"]
  - Questions about what strategy worked, persuasion approach, tactics:
    → ["strategy"]
  - Questions about past conversations, what happened in a year:
    → ["session"]
  - Broad questions about evolution, trends, comparisons across years:
    → ["all"]  and year_range: "all_years"

Rules for year_range:
  - "latest"    — current/live negotiation session queries
  - "all_years" — history, trends, comparisons
  - "earliest"  — specifically about first/2015 interactions
  - "YYYY"      — explicit year mentioned in query

Output ONLY the JSON, no markdown, no extra text.
""".strip()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def route_query(
    query: str,
    available_years: List[int],
    entry_year: int,
    earliest_year: int,
    persona_summary: str = "",
) -> QueryRoute:
    """
    Use Azure OpenAI to semantically route the query.

    Parameters
    ----------
    query           : user's natural language query / utterance
    available_years : sorted list of years in the memory graph
    entry_year      : latest year (graph entry point)
    earliest_year   : first year (2015)
    persona_summary : optional — the persona's memory summary for extra context
    """
    user_content = f"User query: {query}"
    if persona_summary:
        user_content = f"Persona context: {persona_summary}\n\n{user_content}"

    try:
        resp = _client.chat.completions.create(
            model    = AZURE_MODEL,
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens       = 200,
            temperature      = 0.0,
            response_format  = {"type": "json_object"},
        )
        raw    = resp.choices[0].message.content.strip()
        parsed = json.loads(raw)
    except Exception as e:
        print(f"[router] LLM error: {e} — falling back to full traversal")
        parsed = {"node_types": ["all"], "year_range": "all_years",
                  "reasoning": f"LLM error: {e}"}

    # ── Validate ────────────────────────────────────────────────────────────
    VALID_TYPES = {"session", "latent", "strategy", "concession", "all"}
    node_types  = [t for t in parsed.get("node_types", ["all"]) if t in VALID_TYPES]
    if not node_types:
        node_types = ["all"]

    year_range = parsed.get("year_range", "latest")
    reasoning  = parsed.get("reasoning", "")

    # ── Resolve target years ────────────────────────────────────────────────
    if year_range == "latest":
        target_years = [entry_year]
    elif year_range == "earliest":
        target_years = [earliest_year]
    elif year_range == "all_years":
        target_years = available_years
    elif re.match(r"^\d{4}$", year_range) and int(year_range) in available_years:
        target_years = [int(year_range)]
    else:
        target_years = [entry_year]   # safe fallback

    # ── Build concrete node IDs ─────────────────────────────────────────────
    effective_types = {"session","latent","strategy","concession"} \
                      if "all" in node_types else set(node_types)

    nodes_to_traverse: List[str] = []
    for year in target_years:
        if "session"    in effective_types:
            nodes_to_traverse.append(f"session_{year}")
        if "strategy"   in effective_types:
            nodes_to_traverse.append(f"strategy_{year}")
        if "concession" in effective_types:
            nodes_to_traverse.append(f"concession_{year}")

    if "latent" in effective_types:
        nodes_to_traverse.append(f"latent_{entry_year}")

    # Always include aggregated summaries
    nodes_to_traverse.extend([f"strategy_agg_{entry_year}", f"concession_agg_{entry_year}"])

    # Deduplicate preserving order
    seen: set = set()
    nodes_to_traverse = [n for n in nodes_to_traverse if not (n in seen or seen.add(n))]

    return QueryRoute(
        node_types        = node_types,
        year_range        = year_range,
        reasoning         = reasoning,
        nodes_to_traverse = nodes_to_traverse,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    years = list(range(2015, 2026))
    queries = [
        "I want to buy the insurance",
        "What price should I offer?",
        "How has trust changed over the years?",
        "What offer did the agent make in 2018?",
        "Show me the full negotiation history",
        "What is the current best strategy to use?",
        "The user seems hesitant, what should I do?",
    ]
    for q in queries:
        r = route_query(q, years, 2025, 2015)
        print(f"\nQuery   : {q}")
        print(f"Types   : {r.node_types}   Year: {r.year_range}")
        print(f"Reason  : {r.reasoning}")
        print(f"Nodes   : {r.nodes_to_traverse}")
