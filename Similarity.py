
# ============================================================
# MITRE ATT&CK — PURE SIMILARITY SEARCH  (No LLM)
# app10.py
# ============================================================
# Embeds the user's query with FastEmbed (BAAI/bge-small-en-v1.5)
# and retrieves the most semantically similar ATT&CK objects
# directly from the local Qdrant vector store.
# No language model is called at any point.
# ============================================================

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

os.environ["HF_HOME"]                         = str(BASE_DIR / "hf_cache")
os.environ["HF_HUB_CACHE"]                    = str(BASE_DIR / "hf_cache" / "hub")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"]         = "1"

import json
import math
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import gradio as gr
except ImportError as e:
    raise ImportError("pip install gradio") from e

try:
    from fastembed import TextEmbedding
except ImportError as e:
    raise ImportError("pip install fastembed") from e

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ImportError as e:
    raise ImportError("pip install qdrant-client") from e


# ============================================================
# CONFIG
# ============================================================

QDRANT_PATH      = BASE_DIR / "vector_store" / "qdrant_attack_enterprise"
COLLECTION_NAME  = "mitre_attack_enterprise"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
FASTEMBED_CACHE_DIR  = BASE_DIR / "fastembed_cache"

DEFAULT_TOP_K     = 15
MAX_TOP_K         = 100
SCROLL_BATCH_SIZE = 256

SERVER_NAME = "127.0.0.1"
SERVER_PORT = 7861       # different port from app6

# ============================================================
# TACTICS / PLATFORMS  (for filter dropdowns)
# ============================================================

ATT_TACTICS = [
    "All",
    "reconnaissance", "resource-development", "initial-access",
    "execution", "persistence", "privilege-escalation",
    "defense-evasion", "credential-access", "discovery",
    "lateral-movement", "collection", "command-and-control",
    "exfiltration", "impact",
]

ATT_PLATFORMS = [
    "All",
    "Windows", "Linux", "macOS", "Android", "iOS",
    "Cloud", "Azure AD", "Office 365", "Google Workspace",
    "SaaS", "IaaS", "Network", "Containers", "PRE",
]

OBJECT_TYPES = [
    "All",
    "technique", "sub-technique", "group", "software",
    "malware", "tool", "mitigation", "campaign", "data-source",
]

# ============================================================
# SCORE BADGE COLOURS
# ============================================================

def score_badge(score: float) -> str:
    """Return a coloured HTML badge for a cosine similarity score."""
    pct = int(score * 100)
    if score >= 0.75:
        color = "#22c55e"   # green
    elif score >= 0.55:
        color = "#f59e0b"   # amber
    elif score >= 0.35:
        color = "#f97316"   # orange
    else:
        color = "#ef4444"   # red
    return (
        f'<span style="display:inline-block;background:{color};color:#fff;'
        f'font-weight:700;font-size:0.78rem;padding:2px 9px;border-radius:20px;'
        f'letter-spacing:0.04em;">{pct}%</span>'
    )

# ============================================================
# CUSTOM CSS
# ============================================================

CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

html, body {
    background: #05050a !important;
    color: #e4e4e7 !important;
    overflow-y: auto !important;
    height: auto !important;
    min-height: 100vh;
    margin: 0;
    padding: 0;
}

.gradio-container {
    background: #05050a !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    color: #e4e4e7 !important;
    height: auto !important;
    min-height: 100% !important;
}

/* ── Header ─────────────────────────────────────────────── */
.header-banner {
    background: linear-gradient(145deg, #0f0f1a 0%, #05050a 100%);
    border: 1px solid #1e1e3a;
    border-radius: 18px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
}
.header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899, #f59e0b);
}
.header-banner::after {
    content: '';
    position: absolute;
    right: -60px; top: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
}
.header-banner h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    margin: 0 0 10px 0 !important;
    background: linear-gradient(to right, #ffffff, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-banner p {
    color: #71717a !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
    line-height: 1.6 !important;
}
.header-banner .badge-row {
    display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px;
}
.header-banner .badge {
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
}

/* ── Result cards ────────────────────────────────────────── */
.results-panel {
    background: #0a0a14 !important;
    border: 1px solid #1e1e3a !important;
    border-radius: 16px !important;
    padding: 24px !important;
    min-height: 500px !important;
}
.result-card {
    background: #111128;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
    transition: all 0.2s ease;
    position: relative;
}
.result-card:hover {
    border-color: #6366f1;
    box-shadow: 0 4px 20px rgba(99,102,241,0.15);
    transform: translateY(-2px);
}
.result-card .rank {
    position: absolute;
    top: 16px; right: 16px;
    color: #3f3f5a;
    font-size: 1.6rem;
    font-weight: 800;
    font-family: 'Fira Code', monospace;
}
.result-card h3 {
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    margin: 0 0 6px 0 !important;
}
.result-card .meta {
    display: flex; flex-wrap: wrap; gap: 8px;
    margin: 8px 0;
}
.result-card .tag {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
}
.result-card .tag-id    { background: rgba(99,102,241,0.15); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.25); }
.result-card .tag-type  { background: rgba(139,92,246,0.15); color: #c4b5fd; border: 1px solid rgba(139,92,246,0.25); }
.result-card .tag-tact  { background: rgba(236,72,153,0.12); color: #f9a8d4; border: 1px solid rgba(236,72,153,0.2); }
.result-card .tag-plat  { background: rgba(20,184,166,0.12); color: #5eead4; border: 1px solid rgba(20,184,166,0.2); }
.result-card .preview {
    color: #71717a;
    font-size: 0.85rem;
    line-height: 1.6;
    margin-top: 10px;
    border-top: 1px solid #1e1e3a;
    padding-top: 10px;
}
.result-card .url-link {
    color: #6366f1 !important;
    font-size: 0.8rem;
    text-decoration: none;
}
.result-card .url-link:hover { text-decoration: underline; }

/* ── Search bar area ─────────────────────────────────────── */
.search-area {
    background: #0a0a14 !important;
    border: 1px solid #1e1e3a !important;
    border-radius: 16px !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
}
.gr-textbox textarea, .gr-textbox input {
    background: #111128 !important;
    border: 1px solid #2d2d5a !important;
    border-radius: 12px !important;
    color: #fafafa !important;
    font-size: 1rem !important;
    padding: 14px 16px !important;
    transition: all 0.2s ease !important;
}
.gr-textbox textarea:focus, .gr-textbox input:focus {
    border-color: #6366f1 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
}
.gr-textbox label { color: #71717a !important; font-size: 0.85rem !important; font-weight: 600 !important; }

/* ── Buttons ─────────────────────────────────────────────── */
button.primary {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 13px 32px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 14px rgba(124,58,237,0.35) !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    box-shadow: 0 6px 24px rgba(124,58,237,0.55) !important;
    transform: translateY(-2px) !important;
}
button.secondary {
    background: #1a1a2e !important;
    border: 1px solid #2d2d5a !important;
    border-radius: 10px !important;
    color: #a1a1aa !important;
    font-weight: 500 !important;
    padding: 13px 24px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
button.secondary:hover {
    background: #1e1e3a !important;
    border-color: #4f46e5 !important;
    color: #e4e4e7 !important;
}

/* ── Filter panel ────────────────────────────────────────── */
.filter-panel {
    background: #0a0a14 !important;
    border: 1px solid #1e1e3a !important;
    border-radius: 14px !important;
    padding: 20px !important;
}
.filter-panel h3 { color: #a1a1aa !important; font-size: 0.8rem !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 0.1em; margin: 0 0 16px 0 !important; }
.gr-dropdown select, .gr-dropdown .wrap {
    background: #111128 !important;
    border: 1px solid #2d2d5a !important;
    border-radius: 10px !important;
    color: #fafafa !important;
}
.gr-dropdown label { color: #71717a !important; font-size: 0.82rem !important; font-weight: 500 !important; }
.gr-slider label   { color: #71717a !important; font-size: 0.82rem !important; font-weight: 500 !important; }
.gr-checkbox label { color: #c4b5fd !important; font-weight: 500 !important; }
.gr-checkbox input[type=checkbox]:checked { accent-color: #8b5cf6 !important; }

/* ── Stats bar ───────────────────────────────────────────── */
.stats-bar {
    display: flex; gap: 20px; flex-wrap: wrap;
    padding: 14px 20px;
    background: #0a0a14;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    margin-bottom: 16px;
}
.stat-item { text-align: center; }
.stat-val  { font-size: 1.4rem; font-weight: 800; color: #a5b4fc; }
.stat-lbl  { font-size: 0.72rem; color: #52525b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Dataframe ───────────────────────────────────────────── */
.gr-dataframe table {
    background: #0a0a14 !important;
    border: 1px solid #1e1e3a !important;
    border-radius: 12px !important;
    border-collapse: separate !important;
    font-size: 0.83rem !important;
}
.gr-dataframe th { color: #6366f1 !important; padding: 8px 12px !important; border-bottom: 2px solid #1e1e3a !important; }
.gr-dataframe td { color: #c4c4d4 !important; padding: 6px 12px !important; border-bottom: 1px solid #111128 !important; }
.gr-dataframe tr:hover td { background: #111128 !important; }

/* ── Logs ────────────────────────────────────────────────── */
.logs-box textarea {
    background: #05050a !important;
    border: 1px solid #1e1e3a !important;
    border-radius: 10px !important;
    color: #4ade80 !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.82rem !important;
    padding: 14px !important;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #05050a; }
::-webkit-scrollbar-thumb { background: #2d2d5a; border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }

/* ── Example query buttons ───────────────────────────────── */
.btn-example button {
    background: #0f0f20 !important;
    border: 1px solid #2d2d5a !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    padding: 8px 14px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
.btn-example button:hover {
    background: linear-gradient(135deg, #1e1b4b, #2e1065) !important;
    color: #c4b5fd !important;
    border-color: #6366f1 !important;
    transform: translateY(-1px) !important;
}
"""

# ============================================================
# GLOBALS
# ============================================================

qdrant_client:   Optional[QdrantClient]   = None
embedding_model: Optional[TextEmbedding] = None
startup_status   = "Not initialized."
collection_info: Dict[str, Any]           = {}


# ============================================================
# HELPERS
# ============================================================

def now() -> str:
    return time.strftime("%H:%M:%S")

def safe_join(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(x) for x in value if x)
    return "" if value is None else str(value)

def normalize_filter(value: Optional[str]) -> Optional[str]:
    if not value or str(value).strip().lower() in {"all", "any", "none", ""}:
        return None
    return str(value).strip()

def short_preview(text: str, max_chars: int = 350) -> str:
    text = (text or "").replace("\r", "").strip()
    if len(text) <= max_chars:
        return text
    # cut at last full sentence boundary if possible
    cut = text[:max_chars]
    for sep in (". ", "! ", "? ", "\n"):
        idx = cut.rfind(sep)
        if idx > max_chars // 2:
            return cut[: idx + 1].strip() + "…"
    return cut.rstrip() + "…"


# ============================================================
# INITIALIZATION
# ============================================================

def init_clients() -> str:
    global qdrant_client, embedding_model, startup_status, collection_info
    logs: List[str] = []

    def log(msg: str) -> None:
        line = f"[{now()}] {msg}"
        logs.append(line)
        print(line)

    try:
        if not QDRANT_PATH.exists():
            raise FileNotFoundError(f"Qdrant path not found: {QDRANT_PATH}")

        log("Opening Qdrant local store …")
        qdrant_client = QdrantClient(path=str(QDRANT_PATH))

        collections = [c.name for c in qdrant_client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found. Available: {collections}"
            )

        info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        pts  = getattr(info, "points_count", "N/A")
        log(f"Collection '{COLLECTION_NAME}' — {pts} vectors")
        collection_info["points"] = pts

        log("Loading FastEmbed model …")
        FASTEMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        embedding_model = TextEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            cache_dir=str(FASTEMBED_CACHE_DIR),
        )
        test_vec = list(embedding_model.embed(["test"]))[0]
        log(f"Embedding dimension: {len(test_vec)}")
        collection_info["dim"] = len(test_vec)

        startup_status = "\n".join(logs + ["✅ Ready — no LLM required."])
        return startup_status

    except Exception as e:
        startup_status = "\n".join(logs + ["❌ FAILED.", str(e), traceback.format_exc()])
        print(startup_status)
        return startup_status


# ============================================================
# QDRANT FILTER
# ============================================================

def make_filter(
    object_type: Optional[str],
    tactic:      Optional[str],
    platform:    Optional[str],
) -> Optional[models.Filter]:
    must = []
    for key, val in [
        ("object_type", normalize_filter(object_type)),
        ("tactics",     normalize_filter(tactic)),
        ("platforms",   normalize_filter(platform)),
    ]:
        if val:
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=val)))
    return models.Filter(must=must) if must else None


# ============================================================
# CORE SIMILARITY SEARCH
# ============================================================

def similarity_search(
    query:       str,
    top_k:       int,
    object_type: Optional[str],
    tactic:      Optional[str],
    platform:    Optional[str],
    min_score:   float,
) -> List[Dict[str, Any]]:
    """Embed query → cosine search → return ranked hits above min_score."""
    if qdrant_client is None or embedding_model is None:
        raise RuntimeError("Clients not initialized. Check the Startup Logs tab.")

    query = query.strip()
    if not query:
        raise ValueError("Query is empty.")

    vec = list(embedding_model.embed([query]))[0]
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    else:
        vec = list(vec)

    qf = make_filter(object_type, tactic, platform)

    try:
        result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            query_filter=qf,
            limit=int(top_k),
            with_payload=True,
        )
        hits = result.points
    except Exception:
        hits = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            query_filter=qf,
            limit=int(top_k),
            with_payload=True,
        )

    docs = []
    for h in hits:
        score = float(h.score)
        if score < min_score:
            continue
        p = h.payload or {}
        docs.append({
            "score":       score,
            "object_type": p.get("object_type"),
            "attack_id":   p.get("attack_id"),
            "name":        p.get("name"),
            "status":      p.get("status"),
            "tactics":     p.get("tactics", []),
            "platforms":   p.get("platforms", []),
            "url":         p.get("url", ""),
            "preview":     (p.get("text_for_embedding", "") or "").replace("\r", "").strip(),
            "full_doc":    p.get("full_doc", {}),
        })
    return docs


# ============================================================
# RENDERING
# ============================================================

def render_result_cards(docs: List[Dict[str, Any]], query: str) -> str:
    """Build a rich HTML string with one card per result."""
    if not docs:
        return (
            "<div style='text-align:center;padding:60px 20px;color:#3f3f5a;'>"
            "<div style='font-size:3rem;margin-bottom:16px;'>🔍</div>"
            "<p style='font-size:1rem;font-weight:600;color:#52525b;'>No results found.</p>"
            "<p style='font-size:0.85rem;color:#3f3f5a;'>"
            "Try lowering the minimum score, removing filters, or rephrasing your query.</p>"
            "</div>"
        )

    parts = [
        f"<div style='font-size:0.85rem;color:#52525b;margin-bottom:16px;'>"
        f"Showing <strong style='color:#a5b4fc;'>{len(docs)}</strong> results "
        f"for &ldquo;<em style='color:#c4b5fd;'>{query}</em>&rdquo;</div>"
    ]

    for i, doc in enumerate(docs, 1):
        score       = doc["score"]
        attack_id   = doc.get("attack_id") or "—"
        name        = doc.get("name")      or "Unnamed"
        obj_type    = doc.get("object_type") or "unknown"
        tactics_str = safe_join(doc.get("tactics"))
        plat_str    = safe_join(doc.get("platforms"))
        url         = doc.get("url", "")
        preview     = short_preview(doc.get("preview", ""), 300)

        # Build tactic tags (max 3)
        tactic_tags = ""
        for t in (doc.get("tactics") or [])[:3]:
            tactic_tags += f'<span class="tag tag-tact">{t}</span> '

        # Build platform tags (max 3)
        plat_tags = ""
        for p in (doc.get("platforms") or [])[:3]:
            plat_tags += f'<span class="tag tag-plat">{p}</span> '

        url_html = (
            f'<a class="url-link" href="{url}" target="_blank" '
            f'rel="noopener noreferrer">🔗 {url}</a>'
            if url else ""
        )

        # Score bar (visual)
        bar_width = max(4, int(score * 100))
        bar_color = (
            "#22c55e" if score >= 0.75 else
            "#f59e0b" if score >= 0.55 else
            "#f97316" if score >= 0.35 else "#ef4444"
        )

        card = f"""
<div class="result-card">
  <span class="rank">#{i}</span>

  <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
    {score_badge(score)}
    <span style="width:100%;max-width:180px;height:6px;background:#1e1e3a;border-radius:3px;overflow:hidden;">
      <span style="display:block;width:{bar_width}%;height:100%;background:{bar_color};border-radius:3px;"></span>
    </span>
    <span style="color:#3f3f5a;font-size:0.78rem;font-family:'Fira Code',monospace;">
      {score:.4f}
    </span>
  </div>

  <h3>{name}</h3>

  <div class="meta">
    <span class="tag tag-id">{attack_id}</span>
    <span class="tag tag-type">{obj_type}</span>
    {tactic_tags}
    {plat_tags}
  </div>

  {"<div class='preview'>" + preview + "</div>" if preview else ""}

  <div style="margin-top:10px;">{url_html}</div>
</div>
"""
        parts.append(card)

    return "\n".join(parts)


def render_table(docs: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Rank":        i,
        "Score":       round(d["score"], 4),
        "Object Type": d.get("object_type", ""),
        "ATT&CK ID":   d.get("attack_id", ""),
        "Name":        d.get("name", ""),
        "Tactics":     safe_join(d.get("tactics")),
        "Platforms":   safe_join(d.get("platforms")),
        "URL":         d.get("url", ""),
    } for i, d in enumerate(docs, 1)])


def render_stats_bar(docs: List[Dict[str, Any]], elapsed: float) -> str:
    if not docs:
        return ""
    scores = [d["score"] for d in docs]
    type_counts: Dict[str, int] = {}
    for d in docs:
        t = d.get("object_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    top_types = sorted(type_counts.items(), key=lambda x: -x[1])[:3]
    top_str   = " · ".join(f"{t}×{n}" for t, n in top_types)

    return f"""
<div class="stats-bar">
  <div class="stat-item">
    <div class="stat-val">{len(docs)}</div>
    <div class="stat-lbl">Results</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">{max(scores):.2f}</div>
    <div class="stat-lbl">Best Score</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">{sum(scores)/len(scores):.2f}</div>
    <div class="stat-lbl">Avg Score</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">{min(scores):.2f}</div>
    <div class="stat-lbl">Min Score</div>
  </div>
  <div class="stat-item" style="flex:1;text-align:left;">
    <div class="stat-val" style="font-size:0.9rem;color:#c4b5fd;">{top_str}</div>
    <div class="stat-lbl">Top Types</div>
  </div>
  <div class="stat-item">
    <div class="stat-val" style="font-size:1rem;">{elapsed:.2f}s</div>
    <div class="stat-lbl">Latency</div>
  </div>
</div>
"""


# ============================================================
# MAIN SEARCH HANDLER
# ============================================================

def do_search(
    query:       str,
    top_k:       int,
    min_score:   float,
    object_type: str,
    tactic:      str,
    platform:    str,
    show_table:  bool,
) -> Tuple[str, str, pd.DataFrame, str]:
    """
    Returns: (stats_html, cards_html, table_df, log_str)
    """
    t0 = time.time()
    if not query.strip():
        empty_df = pd.DataFrame(columns=["Rank","Score","Object Type","ATT&CK ID","Name","Tactics","Platforms","URL"])
        return "", "<p style='color:#52525b;padding:40px;text-align:center;'>Enter a query above and press <b>Search</b>.</p>", empty_df, ""

    try:
        docs = similarity_search(
            query=query,
            top_k=top_k,
            object_type=object_type,
            tactic=tactic,
            platform=platform,
            min_score=min_score,
        )
    except Exception as e:
        msg = f"❌ Error: {e}\n{traceback.format_exc()}"
        empty_df = pd.DataFrame()
        return "", f"<pre style='color:#ef4444;padding:20px;'>{msg}</pre>", empty_df, msg

    elapsed = time.time() - t0
    log_line = (
        f"[{now()}] Query='{query}' | top_k={top_k} | min_score={min_score} "
        f"| type={object_type} | tactic={tactic} | platform={platform} "
        f"→ {len(docs)} hits in {elapsed:.2f}s"
    )
    print(log_line)

    stats  = render_stats_bar(docs, elapsed)
    cards  = render_result_cards(docs, query)
    table  = render_table(docs) if show_table and docs else pd.DataFrame()
    return stats, cards, table, log_line


# ============================================================
# GRADIO UI
# ============================================================

EXAMPLE_QUERIES = [
    "credential dumping from LSASS memory",
    "lateral movement via SMB and pass-the-hash",
    "DNS tunneling for command and control",
    "persistence using scheduled tasks or cron jobs",
    "privilege escalation via DLL hijacking",
    "data exfiltration over encrypted channels",
    "living-off-the-land binaries LOLBins evasion",
    "spearphishing attachment initial access",
    "detect mimikatz credential theft windows event logs",
    "ransomware impact encrypt files",
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(css=CUSTOM_CSS, title="ATT&CK Similarity Search") as demo:

        # ── Header ──────────────────────────────────────────
        gr.HTML("""
<div class="header-banner">
  <h1>🔍 MITRE ATT&CK Similarity Search</h1>
  <p>
    Pure vector-based semantic retrieval over the local ATT&amp;CK knowledge base.<br>
    <strong style="color:#a5b4fc;">No LLM · No internet · 100% local.</strong>
    Your query is embedded with FastEmbed and matched against the Qdrant vector store
    using cosine similarity.
  </p>
  <div class="badge-row">
    <span class="badge">⚡ FastEmbed BAAI/bge-small-en-v1.5</span>
    <span class="badge">🗄️ Qdrant local</span>
    <span class="badge">🛡️ MITRE ATT&CK Enterprise</span>
    <span class="badge">🚫 LLM-free</span>
  </div>
</div>
""")

        # ── Main layout ──────────────────────────────────────
        with gr.Row(equal_height=False):

            # ── Left: search + results ────────────────────────
            with gr.Column(scale=3):

                # Search box
                with gr.Group(elem_classes="search-area"):
                    query_box = gr.Textbox(
                        label="🔎  Semantic Query",
                        placeholder="e.g.  credential dumping from LSASS using Mimikatz …",
                        lines=2,
                        elem_id="query-input",
                    )
                    with gr.Row():
                        search_btn = gr.Button("⚡ Search", variant="primary", scale=3)
                        clear_btn  = gr.Button("✕  Clear",  variant="secondary", scale=1)

                # Example queries
                gr.HTML('<div style="color:#3f3f5a;font-size:0.78rem;font-weight:700;'
                        'text-transform:uppercase;letter-spacing:0.1em;margin:16px 0 8px 0;">'
                        'Example Queries</div>')
                with gr.Row(elem_classes="btn-example"):
                    for q in EXAMPLE_QUERIES[:5]:
                        ex_btn = gr.Button(q, size="sm")
                        ex_btn.click(lambda v=q: v, outputs=query_box)
                with gr.Row(elem_classes="btn-example"):
                    for q in EXAMPLE_QUERIES[5:]:
                        ex_btn = gr.Button(q, size="sm")
                        ex_btn.click(lambda v=q: v, outputs=query_box)

                # Stats bar
                stats_html = gr.HTML(label="", value="")

                # Result cards
                with gr.Group(elem_classes="results-panel"):
                    cards_html = gr.HTML(
                        value="<div style='text-align:center;padding:60px 20px;color:#3f3f5a;'>"
                              "<div style='font-size:3rem;margin-bottom:12px;'>🛡️</div>"
                              "<p style='color:#52525b;font-size:0.9rem;'>Results will appear here.</p>"
                              "</div>"
                    )

                # Tabular view (toggleable)
                with gr.Accordion("📊 Tabular View", open=False):
                    result_table = gr.Dataframe(
                        headers=["Rank","Score","Object Type","ATT&CK ID","Name","Tactics","Platforms","URL"],
                        datatype=["number","number","str","str","str","str","str","str"],
                        interactive=False,
                        wrap=True,
                        elem_classes="gr-dataframe",
                    )

            # ── Right: filters ───────────────────────────────
            with gr.Column(scale=1):
                with gr.Group(elem_classes="filter-panel"):
                    gr.HTML("<h3>⚙️ Search Options</h3>")

                    top_k_slider = gr.Slider(
                        minimum=1, maximum=MAX_TOP_K, value=DEFAULT_TOP_K, step=1,
                        label="Max Results (top-k)",
                    )
                    min_score_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.20, step=0.01,
                        label="Min Similarity Score",
                    )

                    gr.HTML('<div style="border-top:1px solid #1e1e3a;margin:16px 0 12px;"></div>')
                    gr.HTML('<div style="color:#52525b;font-size:0.78rem;font-weight:700;'
                            'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">'
                            'Filters</div>')

                    type_dd = gr.Dropdown(
                        choices=OBJECT_TYPES,
                        value="All",
                        label="Object Type",
                    )
                    tactic_dd = gr.Dropdown(
                        choices=ATT_TACTICS,
                        value="All",
                        label="Tactic",
                    )
                    platform_dd = gr.Dropdown(
                        choices=ATT_PLATFORMS,
                        value="All",
                        label="Platform",
                    )

                    gr.HTML('<div style="border-top:1px solid #1e1e3a;margin:16px 0 12px;"></div>')
                    show_table_chk = gr.Checkbox(
                        value=True, label="Show tabular view",
                        elem_classes="gr-checkbox",
                    )

                # Startup logs
                with gr.Accordion("🖥️ Startup Logs", open=False):
                    log_box = gr.Textbox(
                        value=startup_status,
                        label="",
                        lines=10,
                        interactive=False,
                        elem_classes="logs-box",
                    )

                # Query log
                with gr.Accordion("📜 Query Log", open=False):
                    query_log = gr.Textbox(
                        label="",
                        lines=6,
                        interactive=False,
                        elem_classes="logs-box",
                        placeholder="Search history will appear here …",
                    )

        # ── Event wiring ─────────────────────────────────────
        search_inputs = [
            query_box, top_k_slider, min_score_slider,
            type_dd, tactic_dd, platform_dd, show_table_chk,
        ]
        search_outputs = [stats_html, cards_html, result_table, query_log]

        search_btn.click(do_search, inputs=search_inputs, outputs=search_outputs)
        query_box.submit(do_search, inputs=search_inputs, outputs=search_outputs)
        clear_btn.click(
            lambda: ("", "", pd.DataFrame(), ""),
            outputs=search_outputs,
        )
        clear_btn.click(lambda: "", outputs=query_box)

    return demo


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MITRE ATT&CK — Pure Similarity Search  (app10.py)")
    print("=" * 70)
    print(f"Qdrant path : {QDRANT_PATH}")
    print(f"Collection  : {COLLECTION_NAME}")
    print(f"Embed model : {EMBEDDING_MODEL_NAME}")
    print(f"Server      : http://{SERVER_NAME}:{SERVER_PORT}")
    print("=" * 70)

    init_result = init_clients()
    print(init_result)

    demo = build_ui()
    demo.launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        show_error=True,
        inbrowser=True,
    )
