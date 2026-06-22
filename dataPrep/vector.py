# ============================================================
# MITRE ATT&CK RAG VECTOR INDEX BUILDER
# Windows-friendly .py script
#
# Input:
#   rag_output/enterprise_attack_rag_corpus.jsonl
#
# Output:
#   vector_store/qdrant_attack_enterprise/
#
# Uses:
#   - Qdrant local mode
#   - FastEmbed embedding model
#   - Full enriched JSON document stored as payload
#
# Important:
#   Put this whole file in:
#   C:\Users\rihem\Desktop\stage\vector.py
# ============================================================

import os
from pathlib import Path

# ============================================================
# IMPORTANT WINDOWS CACHE CONFIG
# Must be set BEFORE importing fastembed / huggingface_hub
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

os.environ["HF_HOME"] = str(BASE_DIR / "hf_cache")
os.environ["HF_HUB_CACHE"] = str(BASE_DIR / "hf_cache" / "hub")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# This variable is harmless if unsupported by your installed huggingface_hub.
# It helps in some environments.
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"


# ============================================================
# STANDARD IMPORTS
# ============================================================

import sys
import json
import subprocess
from typing import List, Dict, Any, Optional

import pandas as pd


# ============================================================
# AUTO-INSTALL DEPENDENCIES
# ============================================================

def install_if_missing(import_name: str, pip_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        print(f"[INSTALL] Installing {pip_name} ...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            pip_name,
        ])


install_if_missing("qdrant_client", "qdrant-client")
install_if_missing("fastembed", "fastembed")
install_if_missing("tqdm", "tqdm")
install_if_missing("pandas", "pandas")


# ============================================================
# IMPORT AFTER INSTALL
# ============================================================

from tqdm import tqdm
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models


# ============================================================
# CONFIG
# ============================================================

CORPUS_PATH = BASE_DIR / "rag_output" / "enterprise_attack_rag_corpus.jsonl"

QDRANT_PATH = BASE_DIR / "vector_store" / "qdrant_attack_enterprise"
COLLECTION_NAME = "mitre_attack_enterprise"

FASTEMBED_CACHE_DIR = BASE_DIR / "fastembed_cache"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

BATCH_SIZE = 64
REBUILD_COLLECTION = True


# ============================================================
# PRINT HELPERS
# ============================================================

def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_df(df: pd.DataFrame, max_rows: int = 30) -> None:
    if df.empty:
        print("None.")
    else:
        print(df.head(max_rows).to_string(index=False))


# ============================================================
# LOAD CORPUS
# ============================================================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {path}\n"
            "Expected: rag_output/enterprise_attack_rag_corpus.jsonl"
        )

    docs = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number}: {e}")

            text = doc.get("text_for_embedding", "")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Line {line_number} has empty or invalid text_for_embedding."
                )

            docs.append(doc)

    if not docs:
        raise ValueError("Corpus file is empty.")

    return docs


def safe_join(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return ""


def build_docs_df(docs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for i, doc in enumerate(docs):
        text = doc.get("text_for_embedding", "")

        rows.append({
            "row_id": i,
            "doc_id": doc.get("doc_id"),
            "stix_id": doc.get("stix_id"),
            "object_type": doc.get("object_type"),
            "attack_id": doc.get("attack_id"),
            "name": doc.get("name"),
            "status": doc.get("status"),
            "tactics": safe_join(doc.get("tactics")),
            "platforms": safe_join(doc.get("platforms")),
            "text_chars": len(text),
            "url": doc.get("url"),
        })

    return pd.DataFrame(rows)


# ============================================================
# PRE-INDEX VALIDATION
# ============================================================

def validate_docs(docs_df: pd.DataFrame) -> None:
    checks = {
        "no_duplicate_doc_id": docs_df["doc_id"].duplicated().sum() == 0,
        "no_duplicate_stix_id": docs_df["stix_id"].duplicated().sum() == 0,
        "all_status_active": docs_df["status"].eq("active").all(),
        "no_empty_text": docs_df["text_chars"].gt(0).all(),
        "has_attack_patterns": (docs_df["object_type"] == "attack-pattern").any(),
        "has_T1059": (docs_df["attack_id"] == "T1059").any(),
        "has_Mimikatz": docs_df["name"].fillna("").str.contains(
            "Mimikatz",
            case=False,
            regex=False,
        ).any(),
    }

    checks_df = pd.DataFrame([
        {"check": key, "passed": bool(value)}
        for key, value in checks.items()
    ])

    section("PRE-INDEX CHECKS")
    print_df(checks_df)

    failed = checks_df[checks_df["passed"] == False]

    if not failed.empty:
        raise ValueError("Pre-index checks failed. Fix corpus before indexing.")

    print("\nPre-index checks passed.")


# ============================================================
# EMBEDDING MODEL
# ============================================================

def load_embedding_model() -> tuple[TextEmbedding, int]:
    section(f"LOADING EMBEDDING MODEL: {EMBEDDING_MODEL_NAME}")

    FASTEMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        embedding_model = TextEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            cache_dir=str(FASTEMBED_CACHE_DIR),
        )

        sample_vector = list(
            embedding_model.embed(["test embedding dimension"])
        )[0]

        vector_size = len(sample_vector)

        print(f"Embedding model loaded.")
        print(f"Vector size: {vector_size}")
        print(f"FastEmbed cache: {FASTEMBED_CACHE_DIR}")

        return embedding_model, vector_size

    except Exception as e:
        print("\nFAILED TO LOAD EMBEDDING MODEL")
        print(str(e))
        print("\nMost likely cause on Windows:")
        print("- Hugging Face cache symlink permission issue.")
        print("- Or Python 3.14 compatibility issue.")
        print("\nRecommended fix:")
        print("1. Enable Windows Developer Mode, OR run terminal as Administrator.")
        print("2. Prefer Python 3.12 for this RAG project.")
        print("\nPowerShell setup:")
        print("py -3.12 -m venv .venv")
        print(r".\.venv\Scripts\activate")
        print("python -m pip install --upgrade pip")
        print("pip install qdrant-client fastembed tqdm pandas")
        print("python vector.py")
        raise


# ============================================================
# QDRANT SETUP
# ============================================================

def create_qdrant_collection(vector_size: int) -> QdrantClient:
    section("SETTING UP QDRANT LOCAL VECTOR STORE")

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(QDRANT_PATH))

    existing_collections = [
        c.name for c in client.get_collections().collections
    ]

    if COLLECTION_NAME in existing_collections and REBUILD_COLLECTION:
        print(f"Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(collection_name=COLLECTION_NAME)

    existing_collections = [
        c.name for c in client.get_collections().collections
    ]

    if COLLECTION_NAME not in existing_collections:
        print(f"Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
    else:
        print(f"Using existing collection: {COLLECTION_NAME}")

    print(f"Qdrant path: {QDRANT_PATH}")

    return client


# ============================================================
# PAYLOAD BUILDER
# ============================================================

def make_payload(doc: Dict[str, Any], row_id: int) -> Dict[str, Any]:
    tactics = doc.get("tactics", [])
    platforms = doc.get("platforms", [])

    if not isinstance(tactics, list):
        tactics = []

    if not isinstance(platforms, list):
        platforms = []

    return {
        "row_id": row_id,
        "doc_id": doc.get("doc_id"),
        "stix_id": doc.get("stix_id"),
        "object_type": doc.get("object_type"),
        "attack_id": doc.get("attack_id"),
        "name": doc.get("name"),
        "status": doc.get("status"),
        "tactics": tactics,
        "platforms": platforms,
        "url": doc.get("url"),
        "text_for_embedding": doc.get("text_for_embedding", ""),
        "full_doc": doc,
    }


# ============================================================
# BUILD INDEX
# ============================================================
def build_vector_index(
    client: QdrantClient,
    embedding_model: TextEmbedding,
    docs: List[Dict[str, Any]],
) -> None:
    section("BUILDING VECTOR INDEX")

    total_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE

    for start in tqdm(
        range(0, len(docs), BATCH_SIZE),
        total=total_batches,
        desc="Indexing batches",
    ):
        end = min(start + BATCH_SIZE, len(docs))
        batch_docs = docs[start:end]

        batch_texts = [
            doc["text_for_embedding"]
            for doc in batch_docs
        ]

        batch_vectors = list(embedding_model.embed(batch_texts))

        points = []

        for offset, vector in enumerate(batch_vectors):
            row_id = start + offset
            doc = batch_docs[offset]

            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            else:
                vector = list(vector)

            points.append(
                models.PointStruct(
                    id=row_id,
                    vector=vector,
                    payload=make_payload(doc, row_id),
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

    info = client.get_collection(collection_name=COLLECTION_NAME)

    # qdrant-client versions expose slightly different CollectionInfo fields.
    # points_count is the important one for this project.
    points_count = getattr(info, "points_count", None)
    vectors_count = getattr(info, "vectors_count", None)
    indexed_vectors_count = getattr(info, "indexed_vectors_count", None)
    status = getattr(info, "status", None)

    section("INDEX CREATED")

    summary = pd.DataFrame([{
        "qdrant_path": str(QDRANT_PATH),
        "collection_name": COLLECTION_NAME,
        "points_count": points_count,
        "vectors_count": vectors_count,
        "indexed_vectors_count": indexed_vectors_count,
        "status": str(status),
        "expected_documents": len(docs),
    }])

    print_df(summary)

    if points_count is not None and points_count != len(docs):
        raise ValueError(
            f"Qdrant point count mismatch: expected {len(docs)}, got {points_count}"
        )

    print("\nIndex build completed successfully.")
# ============================================================
# SEARCH
# ============================================================

def search_attack_rag(
    client: QdrantClient,
    embedding_model: TextEmbedding,
    query: str,
    top_k: int = 5,
    object_type: Optional[str] = None,
    tactic: Optional[str] = None,
    platform: Optional[str] = None,
) -> pd.DataFrame:
    query_vector = list(embedding_model.embed([query]))[0]

    if hasattr(query_vector, "tolist"):
        query_vector = query_vector.tolist()
    else:
        query_vector = list(query_vector)

    must_filters = []

    if object_type:
        must_filters.append(
            models.FieldCondition(
                key="object_type",
                match=models.MatchValue(value=object_type),
            )
        )

    if tactic:
        must_filters.append(
            models.FieldCondition(
                key="tactics",
                match=models.MatchValue(value=tactic),
            )
        )

    if platform:
        must_filters.append(
            models.FieldCondition(
                key="platforms",
                match=models.MatchValue(value=platform),
            )
        )

    query_filter = models.Filter(must=must_filters) if must_filters else None

    try:
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        hits = result.points
    except Exception:
        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

    rows = []

    for hit in hits:
        payload = hit.payload or {}

        tactics = payload.get("tactics", [])
        platforms = payload.get("platforms", [])

        rows.append({
            "score": round(float(hit.score), 4),
            "object_type": payload.get("object_type"),
            "attack_id": payload.get("attack_id"),
            "name": payload.get("name"),
            "tactics": ", ".join(tactics) if isinstance(tactics, list) else "",
            "platforms": ", ".join(platforms) if isinstance(platforms, list) else "",
            "doc_id": payload.get("doc_id"),
            "url": payload.get("url"),
            "text_preview": (payload.get("text_for_embedding") or "")[:400],
        })

    return pd.DataFrame(rows)


# ============================================================
# TEST SEARCHES
# ============================================================

def run_test_searches(client: QdrantClient, embedding_model: TextEmbedding) -> None:
    section("TEST SEARCH 1: T1059 / COMMAND AND SCRIPTING INTERPRETER")
    result = search_attack_rag(
        client=client,
        embedding_model=embedding_model,
        query="T1059 command and scripting interpreter PowerShell bash cmd execution",
        top_k=5,
    )
    print_df(result)

    section("TEST SEARCH 2: MIMIKATZ")
    result = search_attack_rag(
        client=client,
        embedding_model=embedding_model,
        query="Mimikatz credential dumping Windows passwords LSASS",
        top_k=5,
    )
    print_df(result)

    section("TEST SEARCH 3: CREDENTIAL ACCESS TECHNIQUES ONLY")
    result = search_attack_rag(
        client=client,
        embedding_model=embedding_model,
        query="techniques for stealing credentials and dumping passwords",
        top_k=10,
        object_type="attack-pattern",
        tactic="credential-access",
    )
    print_df(result)

    section("TEST SEARCH 4: INITIAL ACCESS TECHNIQUES ONLY")
    result = search_attack_rag(
        client=client,
        embedding_model=embedding_model,
        query="how adversaries gain initial access through phishing or public facing applications",
        top_k=10,
        object_type="attack-pattern",
        tactic="initial-access",
    )
    print_df(result)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    section("MITRE ATT&CK RAG VECTOR INDEX BUILDER")

    print(f"Base directory: {BASE_DIR}")
    print(f"Corpus path:    {CORPUS_PATH}")
    print(f"Qdrant path:    {QDRANT_PATH}")
    print(f"HF_HOME:        {os.environ.get('HF_HOME')}")
    print(f"Python:         {sys.version}")

    docs = load_jsonl(CORPUS_PATH)
    docs_df = build_docs_df(docs)

    section("CORPUS SUMMARY")
    summary = pd.DataFrame([{
        "documents": len(docs),
        "unique_doc_ids": docs_df["doc_id"].nunique(),
        "unique_stix_ids": docs_df["stix_id"].nunique(),
        "object_types": docs_df["object_type"].nunique(),
        "min_text_chars": int(docs_df["text_chars"].min()),
        "median_text_chars": int(docs_df["text_chars"].median()),
        "max_text_chars": int(docs_df["text_chars"].max()),
    }])
    print_df(summary)

    section("DOCUMENTS BY TYPE")
    docs_by_type = (
        docs_df["object_type"]
        .value_counts()
        .rename_axis("object_type")
        .reset_index(name="count")
    )
    print_df(docs_by_type, max_rows=50)

    validate_docs(docs_df)

    embedding_model, vector_size = load_embedding_model()

    client = create_qdrant_collection(vector_size=vector_size)

    build_vector_index(
        client=client,
        embedding_model=embedding_model,
        docs=docs,
    )

    run_test_searches(
        client=client,
        embedding_model=embedding_model,
    )

    section("DONE")
    print("Vector index is ready.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Path:       {QDRANT_PATH}")
    print("\nYou can now use this Qdrant collection in your RAG chatbot.")


if __name__ == "__main__":
    main()