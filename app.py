
# ============================================================
# MITRE ATT&CK RAG CHATBOT — UNLIMITED RETRIEVAL + MODERN UI
# ============================================================
 
import os
from pathlib import Path
 
BASE_DIR = Path(__file__).resolve().parent
 
os.environ["HF_HOME"]                         = str(BASE_DIR / "hf_cache")
os.environ["HF_HUB_CACHE"]                    = str(BASE_DIR / "hf_cache" / "hub")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"]         = "1"
 
import json
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple
 
import pandas as pd
import requests
 
try:
    import gradio as gr
except ImportError as e:
    raise ImportError("pip install gradio qdrant-client fastembed pandas requests") from e
 
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
 
OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_URL  = f"{OLLAMA_HOST}/api/chat"
OLLAMA_TAGS_URL  = f"{OLLAMA_HOST}/api/tags"
 
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS = ["Groq: llama-3.1-8b-instant", "Groq: llama-3.3-70b-versatile", "Groq: qwen/qwen3-32b", "Groq: qwen/qwen3.6-27b"]
 
DEFAULT_TOP_K      = 20
MAX_TOP_K_SLIDER   = 200
SCROLL_BATCH_SIZE  = 256
 
SERVER_NAME = "127.0.0.1"
SERVER_PORT = 7860
 
# ============================================================
# MODERN CSS THEME
# ============================================================
 
CUSTOM_CSS = ""
 
# ============================================================
# GLOBALS
# ============================================================
 
qdrant_client:   Optional[QdrantClient]   = None
embedding_model: Optional[TextEmbedding] = None
startup_status = "Not initialized."
 
 
# ============================================================
# HELPERS
# ============================================================
 
def now() -> str:
    return time.strftime("%H:%M:%S")
 
def safe_join(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return "" if value is None else str(value)
 
def normalize_filter_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"all", "any", "none"}:
        return None
    return value
 
def short_text(text: str, max_chars: int = 800) -> str:
    text = (text or "").replace("\r", "").strip()
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."
 
def print_startup_info() -> None:
    print("=" * 80)
    print("MITRE ATT&CK RAG CHATBOT — UNLIMITED RETRIEVAL + MODERN UI")
    print("=" * 80)
    print(f"Base directory:    {BASE_DIR}")
    print(f"Qdrant path:       {QDRANT_PATH}")
    print(f"Collection:        {COLLECTION_NAME}")
    print(f"Embedding model:   {EMBEDDING_MODEL_NAME}")
    print(f"Ollama host:       {OLLAMA_HOST}")
    print(f"Default model:     {DEFAULT_OLLAMA_MODEL}")
    print(f"Scroll batch size: {SCROLL_BATCH_SIZE}")
    print("=" * 80)
 
 
# ============================================================
# INITIALIZATION
# ============================================================
 
def get_ollama_models() -> List[str]:
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=10)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []) if m.get("name"))
    except Exception:
        return []
 
 
def init_clients() -> str:
    global qdrant_client, embedding_model, startup_status
    logs: List[str] = []
 
    def log(msg: str) -> None:
        line = f"[{now()}] {msg}"
        logs.append(line)
        print(line)
 
    try:
        if not QDRANT_PATH.exists():
            raise FileNotFoundError(f"Qdrant path not found: {QDRANT_PATH}")
 
        log("Opening existing Qdrant local store...")
        qdrant_client = QdrantClient(path=str(QDRANT_PATH))
 
        collections = [c.name for c in qdrant_client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found. Available: {collections}"
            )
 
        info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        log(f"Collection: {COLLECTION_NAME} | points: {getattr(info,'points_count','N/A')} "
            f"| status: {getattr(info,'status','N/A')}")
 
        log("Loading FastEmbed model...")
        FASTEMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        embedding_model = TextEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            cache_dir=str(FASTEMBED_CACHE_DIR),
        )
        test_vec = list(embedding_model.embed(["test"]))[0]
        log(f"Embedding dim: {len(test_vec)}")
 
        log("Checking Ollama...")
        local_models = get_ollama_models()
        if local_models:
            log(f"Ollama models: {', '.join(local_models)}")
            if DEFAULT_OLLAMA_MODEL not in local_models:
                log(f"WARNING: '{DEFAULT_OLLAMA_MODEL}' not found. "
                    f"Run: ollama pull {DEFAULT_OLLAMA_MODEL}")
        else:
            log("WARNING: Ollama not reachable or no models installed.")
 
        startup_status = "\n".join(logs + ["Initialization finished."])
        return startup_status
 
    except Exception as e:
        startup_status = "\n".join(logs + ["FAILED.", str(e), traceback.format_exc()])
        print(startup_status)
        raise
 
 
# ============================================================
# QDRANT RETRIEVAL
# ============================================================
 
def make_qdrant_filter(
    object_type: Optional[str] = None,
    tactic:      Optional[str] = None,
    platform:    Optional[str] = None,
) -> Optional[models.Filter]:
    must = []
    for key, val in [
        ("object_type", normalize_filter_value(object_type)),
        ("tactics",     normalize_filter_value(tactic)),
        ("platforms",   normalize_filter_value(platform)),
    ]:
        if val:
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=val)))
    return models.Filter(must=must) if must else None
 
 
def _payload_to_doc(payload: Dict[str, Any], score: float = 1.0) -> Dict[str, Any]:
    return {
        "score":              score,
        "row_id":             payload.get("row_id"),
        "doc_id":             payload.get("doc_id"),
        "stix_id":            payload.get("stix_id"),
        "object_type":        payload.get("object_type"),
        "attack_id":          payload.get("attack_id"),
        "name":               payload.get("name"),
        "status":             payload.get("status"),
        "tactics":            payload.get("tactics", []),
        "platforms":          payload.get("platforms", []),
        "url":                payload.get("url"),
        "text_for_embedding": payload.get("text_for_embedding", ""),
        "full_doc":           payload.get("full_doc", {}),
    }
 
 
def retrieve_attack_docs(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    object_type: Optional[str] = None,
    tactic: Optional[str] = None,
    platform: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if qdrant_client is None or embedding_model is None:
        raise RuntimeError("Clients not initialized.")
 
    query_vector = list(embedding_model.embed([query]))[0]
    if hasattr(query_vector, "tolist"):
        query_vector = query_vector.tolist()
    else:
        query_vector = list(query_vector)
 
    qf = make_qdrant_filter(object_type, tactic, platform)
 
    try:
        result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=qf,
            limit=int(top_k),
            with_payload=True,
        )
        hits = result.points
    except Exception:
        hits = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=qf,
            limit=int(top_k),
            with_payload=True,
        )
 
    return [_payload_to_doc(h.payload or {}, float(h.score)) for h in hits]
 
 
def retrieve_all_attack_docs(
    object_type: Optional[str] = None,
    tactic: Optional[str] = None,
    platform: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Scroll through the entire Qdrant collection — no vector similarity limit."""
    if qdrant_client is None:
        raise RuntimeError("Qdrant client not initialized.")
 
    qf     = make_qdrant_filter(object_type, tactic, platform)
    docs:  List[Dict[str, Any]] = []
    offset = None
 
    while True:
        scroll_kwargs: Dict[str, Any] = {
            "collection_name": COLLECTION_NAME,
            "limit":           SCROLL_BATCH_SIZE,
            "with_payload":    True,
            "with_vectors":    False,
        }
        if qf is not None:
            scroll_kwargs["scroll_filter"] = qf
        if offset is not None:
            scroll_kwargs["offset"] = offset
 
        records, next_offset = qdrant_client.scroll(**scroll_kwargs)
        for r in records:
            docs.append(_payload_to_doc(r.payload or {}, score=1.0))
 
        if next_offset is None:
            break
        offset = next_offset
 
    return docs
 
 
def retrieved_docs_table(docs: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{
        "score":       round(float(d.get("score", 0.0)), 4),
        "object_type": d.get("object_type"),
        "attack_id":   d.get("attack_id"),
        "name":        d.get("name"),
        "tactics":     safe_join(d.get("tactics")),
        "platforms":   safe_join(d.get("platforms")),
        "url":         d.get("url"),
    } for d in docs])
 
 
def docs_to_context(
    docs: List[Dict[str, Any]],
    max_chars_per_doc: Optional[int] = None,
) -> str:
    blocks = []
    for i, doc in enumerate(docs, 1):
        raw = (doc.get("text_for_embedding", "") or "").replace("\r", "").strip()
        content = short_text(raw, max_chars_per_doc) if max_chars_per_doc else raw
        blocks.append(
            f"[DOCUMENT {i}]\n"
            f"Score: {doc.get('score',0):.4f} | Type: {doc.get('object_type')} | "
            f"ID: {doc.get('attack_id')} | Name: {doc.get('name')}\n"
            f"Tactics: {safe_join(doc.get('tactics'))} | Platforms: {safe_join(doc.get('platforms'))}\n"
            f"URL: {doc.get('url')}\n\n{content}"
        )
    return "\n\n---\n\n".join(blocks)
 
 
# ============================================================
# INTENT DETECTION
# ============================================================
 
OFFENSIVE_KEYWORDS = {
    "how to", "how do i", "show me", "give me", "command", "commands",
    "exploit", "attack", "test", "simulate", "execute", "run", "use",
    "perform", "demonstrate", "example", "nmap", "metasploit", "hydra",
    "mimikatz", "sqlmap", "john", "hashcat", "netcat", "nc", "msfvenom",
    "payload", "reverse shell", "bind shell", "brute force", "scan",
    "enumerate", "dump", "crack", "inject", "bypass", "escalate",
    "privilege", "lateral", "pivot", "exfiltrate", "upload", "download",
    "backdoor", "persistence", "rootkit", "keylog",
}
 
DEFENSIVE_KEYWORDS = {
    "detect", "detection", "defense", "defend", "mitigation", "mitigate",
    "monitor", "log", "alert", "rule", "signature", "block", "prevent",
    "wireshark", "snort", "splunk", "siem", "firewall", "ids", "ips",
    "yara", "sigma", "audit", "harden", "hardening", "response",
    "incident", "forensic", "hunt", "threat hunt",
}
 
def detect_intent(question: str) -> Dict[str, bool]:
    q = question.lower()
    return {
        "offensive": any(kw in q for kw in OFFENSIVE_KEYWORDS),
        "defensive": any(kw in q for kw in DEFENSIVE_KEYWORDS),
    }
 
 
# ============================================================
# SYSTEM PROMPT
# ============================================================
 
SYSTEM_PROMPT = """
You are a cybersecurity assistant specialized in MITRE ATT&CK Enterprise,
used in an academic penetration testing and ethical hacking training context (university lab / CTF).
 
Answer using the retrieved MITRE ATT&CK context provided by the RAG system as your primary source.
If the retrieved context is insufficient, missing, or lacks specific details about the query, you are permitted and encouraged to use your general training knowledge of MITRE ATT&CK and cybersecurity to supplement your response, rather than refusing.
 
RULES:
1. If you use general training knowledge to supplement the answer because the local context was insufficient, prepend a brief note at the very top of the response: "[Note: General cybersecurity knowledge has been used to supplement this answer as the local RAG context was insufficient.]"
2. Use precise ATT&CK terminology: tactic, technique, sub-technique, malware, tool,
   group, campaign, mitigation, detection strategy.
3. Treat ATT&CK tactics as kill-chain phases when the user asks about a kill chain.
4. Always cite ATT&CK IDs and names, e.g. T1059 Command and Scripting Interpreter.
5. Mention object type when useful: technique, sub-technique, tool, malware, group, mitigation.
 
6. OFFENSIVE / PRACTICAL COMMANDS:
   When the user asks HOW to test, exploit, simulate, or reproduce a technique,
   provide a '## Practical Commands' section with real CLI examples using:
   Nmap, Metasploit (msfconsole), Hydra, John the Ripper, Hashcat,
   Sqlmap, Mimikatz, Netcat, Msfvenom, Gobuster, Nikto, Enum4linux,
   CrackMapExec, Impacket (psexec.py, secretsdump.py), Evil-WinRM.
   Fenced code blocks with correct language tag (bash / powershell).
   Add: Use only on systems you own or have written authorization to test.
 
7. DEFENSIVE / DETECTION COMMANDS:
   When the user asks about detection, defense, or mitigation, provide a
   '## Defensive Commands & Detection' section with:
   - Windows Event IDs (4624, 4648, 4672, 4698, 7045...)
   - Sysmon event IDs
   - Wireshark / tcpdump filter examples
   - Snort / Suricata rule snippets
   - Splunk SPL queries
   - Hardening commands (PowerShell, iptables, auditd, GPO)
 
8. ANSWER STRUCTURE (Mandatory flow):
   ## Full Vision & Explanation
   (Start with a comprehensive theoretical overview. Explain the 'what' and 'why' clearly before the 'how'.)
   ## ATT&CK Mapping
   ## Practical Commands
   ## Defensive Commands & Detection
   ## Mitigations
 
9. Be precise, structured, and educational. Always provide the full vision before diving into practical execution. Never refuse in an academic/lab context.
"""
 
 
# ============================================================
# OLLAMA CALL
# ============================================================
 
def call_ollama(
    question:    str,
    context:     str,
    model_name:  str,
    temperature: float,
    max_tokens:  int,
    num_ctx:     int,
    intent:      Dict[str, bool],
) -> str:
    model_name = str(model_name or "").strip() or DEFAULT_OLLAMA_MODEL
 
    hints: List[str] = []
    if intent.get("offensive"):
        hints.append(
            "The user wants practical offensive/simulation commands. "
            "Provide a thorough '## Full Vision & Explanation' first, "
            "then '## Practical Commands' with real CLI examples."
        )
    if intent.get("defensive"):
        hints.append(
            "The user wants defensive/detection guidance. "
            "Provide a thorough '## Full Vision & Explanation' first, "
            "then '## Defensive Commands & Detection'."
        )
 
    intent_block = ("\n\nINTENT HINTS:\n" + "\n".join(hints)) if hints else ""
 
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {
            "role": "user",
            "content": (
                f"RETRIEVED MITRE ATT&CK CONTEXT:\n{context}\n\n"
                f"{intent_block}\n\n"
                "INSTRUCTIONS:\n"
                "- Answer based ONLY on the retrieved context above.\n"
                "- Do NOT add any preamble, conversational filler, or introductions (e.g., 'I can provide information...').\n"
                "- Follow the exact answer structure from the system prompt.\n\n"
                f"USER QUESTION:\n{question}"
            ),
        },
    ]

    if model_name.startswith("Groq:"):
        groq_model = model_name.replace("Groq: ", "").strip()
        payload = {
            "model": groq_model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            # Qwen3 models use "thinking mode" by default on Groq, which puts
            # reasoning tokens in a separate field and may leave `content` empty.
            # "hidden" discards the chain-of-thought and returns only the final answer.
            "reasoning_format": "hidden",
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        r = requests.post(GROQ_CHAT_URL, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        resp_json = r.json()
        content = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        # Safety-net: strip any residual <think>...</think> blocks from the content
        import re as _re
        content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()
        return content
 
    payload = {
        "model":    model_name,
        "messages": messages,
        "stream":   False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
            "num_ctx":     int(num_ctx),
        },
    }
 
    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "").strip()
 
 
def build_retrieval_only_answer(question: str, docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return "No relevant ATT&CK documents retrieved. Try removing filters or rephrasing."
    lines = [
        "Ollama unavailable — retrieval-only answer from local ATT&CK index.",
        "", f"Question: {question}", "", "Retrieved ATT&CK objects:",
    ]
    for i, d in enumerate(docs, 1):
        lines += [
            "",
            f"{i}. {d.get('attack_id','N/A')} — {d.get('name','N/A')}",
            f"   Type: {d.get('object_type','N/A')} | Score: {float(d.get('score',0)):.4f}",
        ]
        if safe_join(d.get("tactics")):
            lines.append(f"   Tactics: {safe_join(d.get('tactics'))}")
        if d.get("url"):
            lines.append(f"   URL: {d['url']}")
        preview = short_text(d.get("text_for_embedding", ""), 400)
        if preview:
            lines.append(f"   Preview: {preview}")
    return "\n".join(lines)
 
 
# ============================================================
# RENDERING
# ============================================================
 
def render_sources_markdown(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return "### Retrieved sources\nNo documents retrieved."
    lines = ["---", "### Retrieved MITRE ATT&CK Sources", ""]
    for i, d in enumerate(docs, 1):
        attack_id = d.get("attack_id") or "N/A"
        name      = d.get("name")      or "N/A"
        obj_type  = d.get("object_type") or "N/A"
        score     = float(d.get("score", 0.0))
        url       = d.get("url") or ""
        line = f"`{i}.` **{attack_id} — {name}** `{obj_type}` · score `{score:.4f}`"
        lines.append(f"{line}  \n   Link: {url}" if url else line)
    return "\n".join(lines)
 
 
def render_final_answer(
    answer: str,
    docs:   List[Dict[str, Any]],
    intent: Dict[str, bool],
) -> str:
    has_commands = "```" in answer or "Practical Commands" in answer
    warning = ""
    if has_commands and intent.get("offensive"):
        warning = (
            "\n\n> **Lab / Authorized Use Only** — "
            "Run offensive commands ONLY on systems you own or have "
            "explicit written permission to test. Unauthorized use is illegal.\n"
        )
        
    note_text = "[Note: General cybersecurity knowledge has been used to supplement this answer as the local RAG context was insufficient.]"
    if note_text in answer:
        answer = answer.replace(note_text, f'<span style="color: red; font-weight: bold;">{note_text}</span>')
        
    return f"## Answer\n\n{answer}\n{warning}\n\n{render_sources_markdown(docs)}".strip()
 
 
def render_conversation(history: List[Tuple[str, str]]) -> str:
    if not history:
        return (
            "### Conversation\n\n"
            "_Ask a MITRE ATT\\&CK question below to get started._\n\n"
            "**Tip:** Use the example buttons to try pre-built queries."
        )
    parts = ["### Conversation"]
    for i, (u, a) in enumerate(history, 1):
        parts += [
            f"\n---\n#### You · Turn {i}\n{u}",
            f"\n#### Assistant · Turn {i}\n{a}",
        ]
    return "\n".join(parts)
 
 
def parse_history(history_json: str) -> List[Tuple[str, str]]:
    if not history_json:
        return []
    try:
        raw = json.loads(history_json)
        return [
            (str(x[0]), str(x[1]))
            for x in raw
            if isinstance(x, (list, tuple)) and len(x) == 2
        ]
    except Exception:
        return []
 
 
def dump_history(history: List[Tuple[str, str]]) -> str:
    return json.dumps(history, ensure_ascii=False)
 
 
# ============================================================
# MAIN CHAT FUNCTION
# ============================================================
 
def rag_chat(
    question:     str,
    history_json: str,
    model_name:   str,
    top_k:        int,
    retrieve_all: bool,
    object_type:  str,
    tactic:       str,
    platform:     str,
    temperature:  float,
    max_tokens:   int,
    num_ctx:      int,
):
    logs:    List[str]              = []
    history: List[Tuple[str, str]] = parse_history(history_json)
 
    def log(msg: str) -> None:
        entry = f"[{now()}] {msg}"
        logs.append(entry)
        print(entry)
 
    question = (question or "").strip()
    if not question:
        return (
            render_conversation(history), "",
            "Please enter a question.",
            pd.DataFrame(), dump_history(history),
        )
 
    try:
        log(f"Question: {question}")
        intent = detect_intent(question)
        log(f"Intent — offensive:{intent['offensive']}  defensive:{intent['defensive']}")
 
        if retrieve_all:
            log("Mode: RETRIEVE ALL (scroll entire collection)...")
            docs = retrieve_all_attack_docs(
                object_type=object_type,
                tactic=tactic,
                platform=platform,
            )
            log(f"Total docs (full collection): {len(docs)}")
        else:
            log(f"Mode: TOP-K semantic search  k={top_k}")
            docs = retrieve_attack_docs(
                query=question,
                top_k=int(top_k),
                object_type=object_type,
                tactic=tactic,
                platform=platform,
            )
            log(f"Retrieved docs: {len(docs)}")
 
        docs_table = retrieved_docs_table(docs)
 
        if not docs:
            answer  = "No relevant ATT&CK documents retrieved. Try removing filters or rephrasing."
            history = history + [(question, answer)]
            return (
                render_conversation(history), "",
                "\n".join(logs), docs_table, dump_history(history),
            )
 
        # For Groq models, we need a hard limit on context size to prevent HTTP 413 Payload Too Large
        # The prompt limits for Groq models are stricter than local Ollama.
        max_chars = 4000 if model_name.startswith("Groq:") else None
        context = docs_to_context(docs, max_chars_per_doc=max_chars)
        
        # If the total context string is still massively huge (e.g., retrieving all 4000+ docs), 
        # truncate the total string size for Groq to a very safe limit of 10,000 characters.
        if model_name.startswith("Groq:") and len(context) > 10000:
            log(f"Context too large for Groq ({len(context)} chars). Truncating.")
            context = context[:10000] + "\n... [TRUNCATED DUE TO SIZE LIMIT] ..."
            
        log(f"Context: {len(context):,} chars")
        log(f"Calling model: {model_name}")
 
        try:
            answer = call_ollama(
                question=question,
                context=context,
                model_name=model_name,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                num_ctx=int(num_ctx),
                intent=intent,
            )
            log("Response received.")
            if not answer:
                log("Empty response — fallback.")
                answer = build_retrieval_only_answer(question, docs)
        except Exception as e:
            log(f"Ollama failed: {e} — fallback.")
            answer = build_retrieval_only_answer(question, docs)
 
        final   = render_final_answer(answer, docs, intent)
        history = history + [(question, final)]
 
        return (
            render_conversation(history), "",
            "\n".join(logs), docs_table, dump_history(history),
        )
 
    except Exception as e:
        log(f"ERROR: {e}\n{traceback.format_exc()}")
        err     = f"Error:\n\n```text\n{e}\n```"
        history = history + [(question, err)]
        return (
            render_conversation(history), "",
            "\n".join(logs), pd.DataFrame(), dump_history(history),
        )
 
 
# ============================================================
# EXAMPLE BUTTONS
# ============================================================
 
def clear_chat():
    return render_conversation([]), "", "", pd.DataFrame(), dump_history([])
 
def ex_t1059():
    return "Explain T1059 Command and Scripting Interpreter sub-techniques. Show practical commands."
 
def ex_mimikatz():
    return "What techniques are related to Mimikatz and credential dumping? Show Mimikatz commands."
 
def ex_initial_access():
    return "Show me Initial Access techniques related to phishing and public-facing applications."
 
def ex_cred_access():
    return "What are common Credential Access techniques on Windows? Show practical commands."
 
def ex_defense_evasion():
    return "What techniques are related to defense evasion, stealth, or defense impairment?"
 
def ex_brute():
    return "How to test T1110 Brute Force on SSH using Hydra? Show practical commands."
 
def ex_nmap():
    return "How to perform service enumeration and port scanning (T1046) using Nmap? Show commands."
 
def ex_metasploit():
    return "How to exploit a vulnerability using Metasploit? Show msfconsole commands step by step."
 
def ex_detect_lateral():
    return "How to detect lateral movement on Windows? Show Event IDs and Splunk queries."
 
def ex_sqli():
    return "How to test SQL injection (T1190) using Sqlmap? Show commands and detection methods."
 
 
def app_health_check() -> str:
    local_models = get_ollama_models()
    lines = [
        "## App Health",
        f"- **Qdrant path exists**: `{QDRANT_PATH.exists()}`",
        f"- **Collection**: `{COLLECTION_NAME}`",
        f"- **Embedding model**: `{EMBEDDING_MODEL_NAME}`",
        f"- **Ollama host**: `{OLLAMA_HOST}`",
        f"- **Default model**: `{DEFAULT_OLLAMA_MODEL}`",
        f"- **Detected models**: `{', '.join(local_models) if local_models else 'None / Ollama not reachable'}`",
        f"- **Scroll batch size**: `{SCROLL_BATCH_SIZE}`",
        "",
        "### Startup log",
        "```text",
        startup_status,
        "```",
    ]
    return "\n".join(lines)
 
 
# ============================================================
# GRADIO UI
# ============================================================
 
def build_ui():
    local_models  = get_ollama_models()
    model_choices = local_models if local_models else [DEFAULT_OLLAMA_MODEL]
    model_choices.extend(GROQ_MODELS)
    default_model = (
        DEFAULT_OLLAMA_MODEL
        if DEFAULT_OLLAMA_MODEL in model_choices
        else model_choices[0]
    )
 
    with gr.Blocks(title="MITRE ATT&CK RAG Chatbot") as demo:
 
        # ── Header ──────────────────────────────────────────────────────
        gr.Markdown(
            """
# MITRE ATT&CK RAG Chatbot
 
**Unlimited Document Retrieval · Local Ollama · Academic Lab**
 
Capabilities: **Full collection scroll** · Offensive commands · Defensive detection · Kill-chain guidance
 
> For academic / lab / CTF use only. Run commands only on systems you are authorized to test.
"""
        )
 
        history_box = gr.Textbox(
            value=dump_history([]), visible=False, label="history_json"
        )
 
        with gr.Row(equal_height=False):
 
            # ── Left column ──────────────────────────────────────────────
            with gr.Column(scale=3, min_width=480):
 
                # Conversation panel
                conversation_md = gr.Markdown(
                    value=render_conversation([]),
                    elem_classes=["conversation-panel"],
                )
 
                # Question input
                question = gr.Textbox(
                    label="Your Question",
                    placeholder=(
                        "e.g.  How to brute force SSH using Hydra? (T1110)\n"
                        "      Show Mimikatz commands for credential dumping\n"
                        "      How to detect lateral movement on Windows?\n"
                        "      Nmap commands for port scanning (T1046)"
                    ),
                    lines=3,
                )
 
                with gr.Row():
                    ask_btn   = gr.Button("Ask",   variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)
 
                # Offensive examples
                gr.Markdown(
                    "#### Offensive / Simulation",
                    elem_classes=["section-label"],
                )
                with gr.Row(elem_classes=["btn-offensive"]):
                    b_brute = gr.Button("Brute Force SSH")
                    b_nmap  = gr.Button("Nmap Scan")
                    b_meta  = gr.Button("Metasploit")
                with gr.Row(elem_classes=["btn-offensive"]):
                    b_mimi  = gr.Button("Mimikatz Dump")
                    b_sqli  = gr.Button("SQLmap")
                    b_t1059 = gr.Button("T1059 Scripting")
 
                # Defensive examples
                gr.Markdown(
                    "#### Defensive / Detection",
                    elem_classes=["section-label"],
                )
                with gr.Row(elem_classes=["btn-defensive"]):
                    b_detect  = gr.Button("Detect Lateral Move")
                    b_defense = gr.Button("Defense Evasion")
                with gr.Row(elem_classes=["btn-defensive"]):
                    b_cred = gr.Button("Credential Access")
                    b_ia   = gr.Button("Initial Access")
 
            # ── Right column — Settings ──────────────────────────────────
            with gr.Column(scale=2, min_width=300, elem_classes=["settings-panel"]):
 
                gr.Markdown("### Settings")
 
                model_name = gr.Dropdown(
                    label="AI Model",
                    choices=model_choices,
                    value=default_model,
                    allow_custom_value=True,
                )
 
                gr.Markdown("---")
                gr.Markdown("#### Retrieval Mode")
 
                retrieve_all = gr.Checkbox(
                    label="Retrieve ALL documents (ignores Top-K)",
                    value=True,
                    info="Scrolls the entire Qdrant collection. Uncheck to use semantic Top-K search.",
                )
 
                top_k = gr.Slider(
                    label="Top K  (used only when 'Retrieve ALL' is OFF)",
                    minimum=1, maximum=MAX_TOP_K_SLIDER,
                    value=DEFAULT_TOP_K, step=1,
                )
 
                gr.Markdown("---")
                gr.Markdown("#### Filters")
 
                object_type = gr.Dropdown(
                    label="Object Type",
                    choices=[
                        ("All", "All"),
                        ("Techniques (attack-pattern)", "attack-pattern"),
                        ("Groups (intrusion-set)", "intrusion-set"),
                        ("Malware (malware)", "malware"),
                        ("Tools (tool)", "tool"),
                        ("Campaigns (campaign)", "campaign"),
                        ("Mitigations (course-of-action)", "course-of-action"),
                        ("Detection Strategies", "x-mitre-detection-strategy"),
                        ("Analytics", "x-mitre-analytic"),
                        ("Data Sources", "x-mitre-data-source"),
                        ("Data Components", "x-mitre-data-component"),
                        ("Tactics", "x-mitre-tactic"),
                    ],
                    value="All",
                )
 
                tactic = gr.Dropdown(
                    label="Tactic / Kill-Chain Phase",
                    choices=[
                        "All", "reconnaissance", "resource-development", "initial-access",
                        "execution", "persistence", "privilege-escalation", "stealth",
                        "defense-impairment", "credential-access", "discovery",
                        "lateral-movement", "collection", "command-and-control",
                        "exfiltration", "impact",
                    ],
                    value="All",
                )
 
                platform = gr.Dropdown(
                    label="Platform",
                    choices=[
                        "All", "Windows", "Linux", "macOS", "ESXi",
                        "IaaS", "SaaS", "Office Suite", "Containers",
                        "Network Devices", "Identity Provider", "PRE",
                    ],
                    value="All",
                )
 
                gr.Markdown("---")
                gr.Markdown("#### Generation Parameters")
 
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.0, maximum=1.0, value=0.1, step=0.05,
                )
                max_tokens = gr.Slider(
                    label="Max Output Tokens",
                    minimum=128, maximum=2048, value=1400, step=64,
                )
                num_ctx = gr.Slider(
                    label="Ollama Context Window (num_ctx)",
                    minimum=2048, maximum=8192, value=6144, step=512,
                )
 
        # ── Accordion: health ────────────────────────────────────────────
        with gr.Accordion("App Health & Startup Logs", open=False):
            health_md = gr.Markdown(value=app_health_check())
 
        # ── Logs ─────────────────────────────────────────────────────────
        logs_box = gr.Textbox(
            label="Runtime Logs",
            lines=10,
            interactive=False,
            elem_classes=["logs-box"],
        )
 
        # ── Retrieved docs table ─────────────────────────────────────────
        retrieved_table = gr.Dataframe(
            label="Retrieved Documents",
            interactive=False,
        )
 
        # ── Wiring ───────────────────────────────────────────────────────
        ask_inputs = [
            question, history_box, model_name, top_k, retrieve_all,
            object_type, tactic, platform,
            temperature, max_tokens, num_ctx,
        ]
        ask_outputs = [
            conversation_md, question, logs_box, retrieved_table, history_box
        ]
 
        ask_btn.click(fn=rag_chat,   inputs=ask_inputs, outputs=ask_outputs)
        question.submit(fn=rag_chat, inputs=ask_inputs, outputs=ask_outputs)
        clear_btn.click(
            fn=clear_chat, inputs=[],
            outputs=[conversation_md, question, logs_box, retrieved_table, history_box],
        )
 
        # Offensive
        b_brute.click(fn=ex_brute,      inputs=[], outputs=question)
        b_nmap.click(fn=ex_nmap,        inputs=[], outputs=question)
        b_meta.click(fn=ex_metasploit,  inputs=[], outputs=question)
        b_mimi.click(fn=ex_mimikatz,    inputs=[], outputs=question)
        b_sqli.click(fn=ex_sqli,        inputs=[], outputs=question)
        b_t1059.click(fn=ex_t1059,      inputs=[], outputs=question)
        # Defensive
        b_detect.click(fn=ex_detect_lateral,  inputs=[], outputs=question)
        b_defense.click(fn=ex_defense_evasion, inputs=[], outputs=question)
        b_cred.click(fn=ex_cred_access,       inputs=[], outputs=question)
        b_ia.click(fn=ex_initial_access,      inputs=[], outputs=question)
 
        demo.load(fn=app_health_check, inputs=[], outputs=health_md)
 
    return demo
 
 
# ============================================================
# MAIN
# ============================================================
 
if __name__ == "__main__":
    print_startup_info()
    init_clients()
 
    print("[APP] Launching Gradio app...")
    print(f"[APP] URL: http://{SERVER_NAME}:{SERVER_PORT}")
 
    demo = build_ui()
    demo.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, share=False)
 
    try:
        if qdrant_client is not None:
            qdrant_client.close()
    except Exception:
        pass
