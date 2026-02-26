"""
graph_visualizer.py  (v2 — cleaner hierarchical layout)
---------------------------------------------------------
Flask web app with a fully redesigned interactive memory graph.
  - Grid layout: years as columns × node types as rows
  - Step-by-step animated traversal with LangGraph pipeline indicator
  - Edges visible on hover or when traversed
  - Click nodes to inspect data

Run:  python graph_visualizer.py
Open: http://localhost:5050
"""

import json
import os
import sys

from flask import Flask, jsonify, render_template_string, request

sys.path.insert(0, os.path.dirname(__file__))
from memory_graph_builder import build_memory_graph, MemoryGraph
from query_router import route_query

SUMMARIZED_DIR = os.path.join(os.path.dirname(__file__), "memory2", "summarized_jsons")
GRAPH: MemoryGraph = build_memory_graph(SUMMARIZED_DIR)
ENTRY_YEAR    = GRAPH.years[-1]
EARLIEST_YEAR = GRAPH.years[0]

# node-type → row index (y position)
ROW = {"persona": 0, "session": 1, "latent": 2, "strategy": 3,
       "strategy_agg": 3, "concession": 4, "concession_agg": 4}

EDGE_COLOURS = {
    "belongs_to":    "#4A90D9",
    "derived_from":  "#A855F7",
    "temporal_next": "#EF4444",
    "similar_to":    "#22C55E",
}

NODE_STYLE = {
    "persona":        {"bg": "#4A90D9", "border": "#2171B5", "fg": "#fff"},
    "session":        {"bg": "#22C55E", "border": "#15803D", "fg": "#fff"},
    "latent":         {"bg": "#A855F7", "border": "#7E22CE", "fg": "#fff"},
    "strategy":       {"bg": "#EC4899", "border": "#9D174D", "fg": "#fff"},
    "strategy_agg":   {"bg": "#F43F5E", "border": "#9F1239", "fg": "#fff"},
    "concession":     {"bg": "#F59E0B", "border": "#B45309", "fg": "#1a1a1a"},
    "concession_agg": {"bg": "#FBBF24", "border": "#92400E", "fg": "#1a1a1a"},
}

X_STEP  = 160
Y_STEP  = 140
X_START = 80
Y_START = 80


def _assign_positions(graph: MemoryGraph):
    """Return {node_id: (x, y)} using a grid layout."""
    unique_years = graph.years  # already sorted unique
    year_col     = {yr: i for i, yr in enumerate(unique_years)}

    # Within each year, track how many session-suffixed entries we've seen
    suffix_row_offset = {}   # node_id → extra y offset for _a/_b stacking

    pos = {}
    for nid, node in graph.nodes.items():
        row = ROW.get(node.node_type, 5)

        if node.node_type == "persona":
            x = (len(unique_years) - 1) * X_STEP / 2 + X_START
            y = Y_START
        else:
            col = year_col.get(node.year, 0)
            x   = X_START + col * X_STEP
            y   = Y_START + row * Y_STEP

            # Stack duplicate-year session nodes side-by-side (slight x-offset)
            if node.node_type == "session" and nid.endswith("_b"):
                x += 20
                y += 30
            elif node.node_type == "session" and nid.endswith("_a"):
                x -= 20
                y -= 30

        pos[nid] = (int(x), int(y))
    return pos


def _short_label(nid: str, node_type: str, year) -> str:
    """Human-readable two-line label."""
    type_map = {
        "persona":        "👤 Persona",
        "session":        "📋 Session",
        "latent":         "🧠 Latent",
        "strategy":       "🎯 Strategy",
        "strategy_agg":   "🎯 Strategy\n(agg)",
        "concession":     "💰 Concession",
        "concession_agg": "💰 Concession\n(agg)",
    }
    label = type_map.get(node_type, node_type)
    if year:
        label += f"\n{year}"
    # Append suffix indicator
    if nid.endswith("_b"):
        label += " ②"
    elif nid.endswith("_a"):
        label += " ①"
    return label


def build_vis_data(graph: MemoryGraph,
                   highlighted_nodes: set = None,
                   highlighted_edges: set = None):
    highlighted_nodes = highlighted_nodes or set()
    highlighted_edges = highlighted_edges or set()
    pos = _assign_positions(graph)

    vis_nodes, vis_edges = [], []

    for nid, node in graph.nodes.items():
        style = NODE_STYLE.get(node.node_type, NODE_STYLE["session"])
        is_hl = nid in highlighted_nodes
        is_entry = (nid == graph.entry_node)

        x, y = pos.get(nid, (0, 0))
        vis_nodes.append({
            "id":    nid,
            "label": _short_label(nid, node.node_type, node.year),
            "x": x, "y": y,
            "fixed": {"x": True, "y": True},
            "color": {
                "background": style["bg"]   if is_hl else "#1E2235",
                "border":     style["border"] if is_hl else "#3A3F58",
                "highlight":  {"background": style["bg"], "border": style["border"]},
            },
            "font": {
                "color": style["fg"] if is_hl else "#4A5070",
                "size": 13, "multi": True, "bold": "14px arial",
            },
            "borderWidth":       4 if is_entry else (3 if is_hl else 1),
            "borderWidthSelected": 5,
            "shadow": is_hl,
            "size":  36 if is_entry else 26,
            "shape": "ellipse",
            "nodeType": node.node_type,
            "year":     node.year,
            "isEntry":  is_entry,
            "data":     node.data,
        })

    for i, (frm, etype, to) in enumerate(graph.edges):
        ekey  = f"{frm}→{etype}→{to}"
        is_hl = ekey in highlighted_edges
        col   = EDGE_COLOURS.get(etype, "#888")
        vis_edges.append({
            "id":     i,
            "from":   frm,
            "to":     to,
            "label":  etype if is_hl else "",
            "title":  f"<b>{etype}</b><br>{frm} → {to}",
            "color":  {
                "color":     col if is_hl else "#1E2235",
                "highlight": col,
                "opacity":   1.0 if is_hl else 0.0,
            },
            "width":   3 if is_hl else 1,
            "arrows":  "to",
            "dashes":  etype == "similar_to",
            "font":    {"size": 11, "color": col, "strokeWidth": 2, "strokeColor": "#0f1117", "align": "middle"},
            "hidden":  not is_hl,
            "edgeType": etype,
        })

    return {"nodes": vis_nodes, "edges": vis_edges}


app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/api/graph")
def api_graph():
    return jsonify(build_vis_data(GRAPH))


@app.route("/api/query", methods=["POST"])
def api_query():
    body  = request.get_json(force=True)
    query = body.get("query", "").strip()
    if not query:
        return jsonify({"error": "empty query"}), 400

    route = route_query(query, GRAPH.years, ENTRY_YEAR, EARLIEST_YEAR)

    # Resolve node IDs
    base_nodes = [GRAPH.entry_node]
    for n in route.nodes_to_traverse:
        if n not in base_nodes and n in GRAPH.nodes:
            base_nodes.append(n)

    # Include session_2015 via similar_to edge
    for sfx in ["", "_a", "_b"]:
        cand = f"session_{EARLIEST_YEAR}{sfx}"
        if cand in GRAPH.nodes and cand not in base_nodes:
            base_nodes.append(cand)

    # Resolve suffixed variants
    resolved = set()
    for nid in base_nodes:
        if nid in GRAPH.nodes:
            resolved.add(nid)
        else:
            for sfx in ["_a", "_b", "_c"]:
                if nid + sfx in GRAPH.nodes:
                    resolved.add(nid + sfx)

    # Highlighted edges
    h_edges = {f"{f}→{e}→{t}" for (f, e, t) in GRAPH.edges
               if f in resolved and t in resolved}

    # Build ordered traversal steps (for animation)
    steps = []
    q_set: set = set()
    queue = list(resolved)
    for nid in queue:
        node = GRAPH.nodes.get(nid)
        if node:
            steps.append({
                "node_id":   nid,
                "node_type": node.node_type,
                "year":      node.year,
            })

    # Traversal log
    log = [
        f"🔍 Query: {query!r}",
        f"📦 Node types: {route.node_types}",
        f"📅 Year range: {route.year_range}",
        f"💡 {route.reasoning}",
        "── Visited nodes ──",
    ]
    for s in steps:
        log.append(f"  ✓ {s['node_id']}  ({s['node_type']}, {s['year']})")

    data = build_vis_data(GRAPH, resolved, h_edges)
    return jsonify({
        **data,
        "traversal_log":  log,
        "traversal_steps": steps,
        "node_types":     route.node_types,
        "year_range":     route.year_range,
        "visited_nodes":  sorted(resolved),
    })


# ─────────────────────────────────────────────────────────────────────────────
# HTML / JS (single-file, no extra assets except vis-network CDN)
# ─────────────────────────────────────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Memory Graph Explorer</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
:root{
  --bg:#0B0D14;--panel:#12151F;--card:#1A1D2E;--border:#252840;
  --accent:#4A90D9;--green:#22C55E;--purple:#A855F7;--red:#EF4444;
  --yellow:#F59E0B;--pink:#EC4899;--text:#D1D5E8;--muted:#525870;
  --entry-glow: 0 0 20px #22C55E88;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
     height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* ── TOP BAR ──────────────────────────────────────────────────────── */
.topbar{
  background:var(--panel);border-bottom:1px solid var(--border);
  padding:12px 18px;display:flex;align-items:center;gap:14px;flex-shrink:0;
  box-shadow:0 2px 12px #0006
}
.logo{font-size:1rem;font-weight:800;color:var(--accent);letter-spacing:.5px;white-space:nowrap}
.search-wrap{flex:1;display:flex;gap:8px;max-width:600px}
#qi{
  flex:1;background:#1E2235;border:1px solid var(--border);color:var(--text);
  border-radius:8px;padding:9px 14px;font-size:.88rem;outline:none;
  transition:border-color .2s;
}
#qi:focus{border-color:var(--accent)}
#qi::placeholder{color:var(--muted)}
.btn-primary{
  background:linear-gradient(135deg,#4A90D9,#2171B5);border:none;color:#fff;
  border-radius:8px;padding:9px 22px;font-size:.88rem;font-weight:700;cursor:pointer;
  transition:opacity .15s;white-space:nowrap
}
.btn-primary:hover{opacity:.85}
.btn-ghost{
  background:none;border:1px solid var(--border);color:var(--muted);
  border-radius:8px;padding:9px 14px;font-size:.82rem;cursor:pointer;
  transition:all .2s
}
.btn-ghost:hover{border-color:var(--accent);color:var(--accent)}

/* ── LANGGRAPH PIPELINE ───────────────────────────────────────────── */
.pipeline{
  display:flex;align-items:center;gap:0;background:var(--panel);
  border-bottom:1px solid var(--border);padding:10px 20px;flex-shrink:0
}
.pipe-step{
  display:flex;align-items:center;gap:8px;padding:7px 16px;
  border-radius:8px;font-size:.78rem;font-weight:600;
  color:var(--muted);border:1px solid transparent;transition:all .35s;
  white-space:nowrap
}
.pipe-step.active{
  color:#fff;border-color:var(--accent);
  background:linear-gradient(135deg,#1E3152,#0E1929);
  box-shadow:0 0 18px #4A90D944
}
.pipe-step.done{color:var(--green);border-color:#15803D44;background:#052E16CC}
.pipe-arrow{color:var(--muted);font-size:1rem;padding:0 4px;flex-shrink:0}
.pipe-icon{font-size:1rem}
.pipe-label{}

/* ── MAIN ─────────────────────────────────────────────────────────── */
.main{display:flex;flex:1;overflow:hidden}
#network{flex:1;background:var(--bg);position:relative}

/* ── NODE TYPE LEGEND (floating over graph) ───────────────────────── */
.graph-legend{
  position:absolute;top:16px;left:16px;background:#12151Fee;
  border:1px solid var(--border);border-radius:10px;padding:12px 14px;
  font-size:.74rem;display:flex;flex-direction:column;gap:7px;
  backdrop-filter:blur(6px);z-index:10
}
.gl-row{display:flex;align-items:center;gap:8px;color:var(--text)}
.gl-dot{width:12px;height:12px;border-radius:50%;flex-shrink:0}
.gl-edge-row{display:flex;align-items:center;gap:8px;color:var(--muted)}
.gl-line{width:24px;height:2px;border-radius:1px;flex-shrink:0}
.gl-dashed{width:24px;border-top:2px dashed var(--green);flex-shrink:0}
.gl-sep{border-top:1px solid var(--border);margin:4px 0}

/* ── RIGHT PANEL ──────────────────────────────────────────────────── */
.rpanel{
  width:300px;background:var(--panel);border-left:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;flex-shrink:0
}
.tabs{display:flex;border-bottom:1px solid var(--border)}
.tab{
  flex:1;padding:10px 4px;text-align:center;font-size:.72rem;font-weight:700;
  cursor:pointer;color:var(--muted);border:none;background:none;
  text-transform:uppercase;letter-spacing:.5px;border-bottom:2px solid transparent;
  transition:all .2s
}
.tab.on{color:var(--accent);border-bottom-color:var(--accent)}
.pane{display:none;padding:14px;overflow-y:auto;flex:1}
.pane.on{display:flex;flex-direction:column;gap:6px}

/* traversal log lines */
.ll{font-size:.75rem;font-family:Consolas,monospace;padding:4px 7px;border-radius:5px;line-height:1.5}
.ll.ok{color:var(--green)}
.ll.hd{color:var(--accent);font-weight:700;font-family:'Segoe UI',sans-serif;font-size:.73rem}
.ll.info{color:var(--muted)}

/* node detail */
.dk{font-size:.67rem;text-transform:uppercase;letter-spacing:.4px;color:var(--muted);margin-top:8px}
.dv{font-size:.82rem;margin-bottom:2px}
pre.dj{
  background:#0B0D14;border:1px solid var(--border);border-radius:6px;
  padding:10px;font-size:.7rem;color:#86EFAC;overflow-x:auto;white-space:pre-wrap;
  max-height:340px
}

/* badges */
.badge{
  display:inline-block;padding:2px 9px;border-radius:12px;
  font-size:.7rem;font-weight:700;margin:2px 2px;background:#1E2235;
  border:1px solid var(--border)
}
.badge.session{color:var(--green);border-color:#15803D44}
.badge.latent{color:var(--purple);border-color:#7E22CE44}
.badge.strategy{color:var(--pink);border-color:#9D174D44}
.badge.concession{color:var(--yellow);border-color:#B4530944}

/* ── STATUS BAR ───────────────────────────────────────────────────── */
.sbar{
  display:flex;align-items:center;gap:20px;padding:6px 16px;
  background:#0B0D14;border-top:1px solid var(--border);
  font-size:.72rem;color:var(--muted);flex-shrink:0
}
.dot-live{width:7px;height:7px;background:var(--green);border-radius:50%;
          box-shadow:0 0 6px var(--green)}
</style>
</head>
<body>

<!-- ── TOP BAR ──────────────────────────────────────────────────────────── -->
<div class="topbar">
  <div class="logo">🧠 Memory Graph Explorer</div>
  <div class="search-wrap">
    <input id="qi" type="text"
      placeholder="e.g.  What was the trust level in 2015?  |  Show concession history  |  Best strategy now"/>
    <button class="btn-primary" id="go">▶ Traverse</button>
    <button class="btn-ghost"   id="rst">⟳ Reset</button>
  </div>
</div>

<!-- ── LANGGRAPH PIPELINE ────────────────────────────────────────────────── -->
<div class="pipeline" id="pipeline">
  <div class="pipe-step" id="ps-start"><span class="pipe-icon">🟢</span><span class="pipe-label">START</span></div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step" id="ps-route"><span class="pipe-icon">🔍</span><span class="pipe-label">route_query</span></div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step" id="ps-traverse"><span class="pipe-icon">🗺</span><span class="pipe-label">traverse_graph</span></div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step" id="ps-generate"><span class="pipe-icon">💬</span><span class="pipe-label">generate_response</span></div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step" id="ps-end"><span class="pipe-icon">🏁</span><span class="pipe-label">END</span></div>
</div>

<!-- ── MAIN ─────────────────────────────────────────────────────────────── -->
<div class="main">

  <!-- graph -->
  <div id="network" style="position:relative">

    <!-- floating legend -->
    <div class="graph-legend">
      <div class="gl-row"><div class="gl-dot" style="background:#4A90D9"></div>Persona</div>
      <div class="gl-row"><div class="gl-dot" style="background:#22C55E"></div>Session Memory</div>
      <div class="gl-row"><div class="gl-dot" style="background:#A855F7"></div>Latent State</div>
      <div class="gl-row"><div class="gl-dot" style="background:#EC4899"></div>Strategy Memory</div>
      <div class="gl-row"><div class="gl-dot" style="background:#F59E0B"></div>Concession Memory</div>
      <div class="gl-sep"></div>
      <div class="gl-edge-row"><div class="gl-line" style="background:#4A90D9"></div>belongs_to</div>
      <div class="gl-edge-row"><div class="gl-line" style="background:#A855F7"></div>derived_from</div>
      <div class="gl-edge-row"><div class="gl-line" style="background:#EF4444"></div>temporal_next</div>
      <div class="gl-edge-row"><div class="gl-dashed"></div>similar_to</div>
    </div>
  </div>

  <!-- right panel -->
  <div class="rpanel">
    <div class="tabs">
      <button class="tab on" data-t="traversal">Traversal</button>
      <button class="tab"    data-t="node">Node Data</button>
    </div>
    <div class="pane on" id="pane-traversal">
      <div class="ll info" style="color:var(--muted)">Type a query and press ▶ Traverse to see the LangGraph pipeline execute step by step.</div>
    </div>
    <div class="pane" id="pane-node">
      <div class="ll info">Click any node in the graph to inspect its data.</div>
    </div>
  </div>

</div>

<!-- ── STATUS BAR ─────────────────────────────────────────────────────────── -->
<div class="sbar">
  <span><div class="dot-live"></div>&nbsp;Live</span>
  <span id="s-nodes">–</span>
  <span id="s-edges">–</span>
  <span id="s-visited">–</span>
  <span id="s-entry" style="color:var(--green)"></span>
</div>

<!-- ────────────────────────────────────────────────────────────────────────── -->
<script>
const qi       = document.getElementById("qi");
const goBt     = document.getElementById("go");
const rstBt    = document.getElementById("rst");
const pTrav    = document.getElementById("pane-traversal");
const pNode    = document.getElementById("pane-node");

// Pipeline steps
const PS = {
  start:    document.getElementById("ps-start"),
  route:    document.getElementById("ps-route"),
  traverse: document.getElementById("ps-traverse"),
  generate: document.getElementById("ps-generate"),
  end:      document.getElementById("ps-end"),
};

let NET = null;
let DS_N = null, DS_E = null;

// ── Network options ────────────────────────────────────────────────────────
const OPTIONS = {
  physics:     { enabled: false },
  interaction: { hover: true, tooltipDelay: 150, zoomView: true },
  edges: {
    smooth:  { type: "cubicBezier", forceDirection: "vertical", roundness: 0.5 },
    arrows:  { to: { enabled: true, scaleFactor: 0.65 } },
    font:    { size: 11, align: "middle" },
  },
  nodes: {
    shape: "ellipse",
    font:  { size: 13, multi: false },
    widthConstraint: { maximum: 110 },
  },
};

// ── Load full graph ────────────────────────────────────────────────────────
async function loadGraph() {
  const r    = await fetch("/api/graph");
  const data = await r.json();
  renderNetwork(data.nodes, data.edges);
  document.getElementById("s-nodes").textContent   = `Nodes: ${data.nodes.length}`;
  document.getElementById("s-edges").textContent   = `Edges shown: 0`;
  document.getElementById("s-visited").textContent = "Visited: –";
  const entry = data.nodes.find(n => n.isEntry);
  if (entry) document.getElementById("s-entry").textContent = `Entry: ${entry.id}`;
  resetPipeline();
}

function renderNetwork(nodes, edges) {
  const el = document.getElementById("network");
  if (NET) { NET.destroy(); NET = null; }
  DS_N = new vis.DataSet(nodes);
  DS_E = new vis.DataSet(edges);
  NET  = new vis.Network(el, { nodes: DS_N, edges: DS_E }, OPTIONS);
  NET.fit({ animation: { duration: 600, easingFunction: "easeInOutQuad" } });

  NET.on("click", params => {
    if (!params.nodes.length) return;
    const n = DS_N.get(params.nodes[0]);
    showNodeDetail(n);
    switchTab("node");
  });
}

// ── Run traversal ──────────────────────────────────────────────────────────
async function runQuery() {
  const q = qi.value.trim();
  if (!q) return;

  goBt.disabled = true; goBt.textContent = "…";
  pTrav.innerHTML = "";
  resetPipeline();

  // Step 1 — route
  await sleep(100);
  setPipeStep("route", "active");
  addLog("🔍 Routing query…", "info");
  await sleep(400);

  const res  = await fetch("/api/query", {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ query: q }),
  });
  const data = await res.json();

  setPipeStep("route", "done");
  addLog(`📦 Types: ${data.node_types.map(t=>`<span class="badge ${t}">${t}</span>`).join("")}`, "hd");
  addLog(`📅 Year range: <b style="color:var(--text)">${data.year_range}</b>`, "info");
  await sleep(300);

  // Step 2 — traverse (animate each node one by one)
  setPipeStep("traverse", "active");
  addLog("── Traversal ──", "hd");

  // First update the full graph with grayed colours
  renderNetwork(data.nodes, data.edges);

  const visited = data.traversal_steps || [];
  let   lit     = new Set();

  for (let i = 0; i < visited.length; i++) {
    const step = visited[i];
    lit.add(step.node_id);
    addLog(`✓ ${step.node_id}  <span style="color:var(--muted)">(${step.node_type}, ${step.year})</span>`, "ok");

    // Flash the node
    DS_N.update([{
      id: step.node_id,
      shadow: { enabled: true, color: "#ffffff55", size: 20, x: 0, y: 0 },
    }]);
    await sleep(180);
  }

  setPipeStep("traverse", "done");
  await sleep(300);

  // Step 3 — "generate" (just visual)
  setPipeStep("generate", "active");
  addLog("💬 Generating response with Azure OpenAI…", "info");
  await sleep(600);
  setPipeStep("generate", "done");
  setPipeStep("end", "done");

  addLog(`\n✅ ${visited.length} nodes fetched and context compiled.`, "hd");

  document.getElementById("s-edges").textContent   = `Edges shown: ${data.edges.filter(e=>!e.hidden).length}`;
  document.getElementById("s-visited").textContent = `Visited: ${data.visited_nodes.length}`;

  // Fit to visited nodes
  if (data.visited_nodes.length) {
    NET.fit({ nodes: data.visited_nodes, animation: { duration: 700, easingFunction: "easeInOutQuad" } });
  }

  goBt.disabled = false; goBt.textContent = "▶ Traverse";
}

// ── Helpers ────────────────────────────────────────────────────────────────

function addLog(html, cls) {
  const d = document.createElement("div");
  d.className = `ll ${cls}`;
  d.innerHTML = html;
  pTrav.appendChild(d);
  pTrav.scrollTop = pTrav.scrollHeight;
}

function showNodeDetail(n) {
  if (!n) return;
  const styleMap = {
    session: "var(--green)", latent:"var(--purple)",
    strategy:"var(--pink)", concession:"var(--yellow)",
    strategy_agg:"var(--pink)", concession_agg:"var(--yellow)",
    persona:"var(--accent)",
  };
  const col = styleMap[n.nodeType] || "var(--text)";
  pNode.innerHTML = `
    <div class="dk">Node ID</div>
    <div class="dv" style="font-weight:700;font-size:.9rem;color:${col}">${n.id}</div>
    <div class="dk">Type</div>
    <div class="dv"><span class="badge ${n.nodeType}">${n.nodeType}</span></div>
    <div class="dk">Year</div>
    <div class="dv">${n.year ?? "—"}</div>
    <div class="dk">Raw Data</div>
    <pre class="dj">${JSON.stringify(n.data, null, 2)}</pre>
  `;
}

function setPipeStep(key, state) {
  for (const el of Object.values(PS)) el.classList.remove("active","done");
  if (state === "active") {
    // Mark all steps before this as done
    const order = ["start","route","traverse","generate","end"];
    const idx   = order.indexOf(key);
    for (let i = 0; i < idx; i++) PS[order[i]].classList.add("done");
    PS[key].classList.add("active");
  } else if (state === "done") {
    const order = ["start","route","traverse","generate","end"];
    const idx   = order.indexOf(key);
    for (let i = 0; i <= idx; i++) PS[order[i]].classList.add("done");
  }
}

function resetPipeline() {
  for (const el of Object.values(PS)) el.classList.remove("active","done");
  PS.start.classList.add("done");
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("on", t.dataset.t === name));
  document.querySelectorAll(".pane").forEach(p => p.classList.toggle("on", p.id === `pane-${name}`));
}

document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => switchTab(t.dataset.t)));

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

goBt.addEventListener("click", runQuery);
qi.addEventListener("keydown", e => { if (e.key === "Enter") runQuery(); });
rstBt.addEventListener("click", () => {
  qi.value = ""; pTrav.innerHTML = `<div class="ll info">Type a query and press ▶ Traverse.</div>`;
  pNode.innerHTML = `<div class="ll info">Click any node to inspect its data.</div>`;
  document.getElementById("s-visited").textContent = "Visited: –";
  loadGraph();
});

loadGraph();
</script>
</body>
</html>"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n🚀  Memory Graph Explorer  →  http://localhost:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
