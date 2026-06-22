import argparse
import json
import statistics
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List


REQUIRED_BASE_FIELDS = {
    "doc_id",
    "stix_id",
    "object_type",
    "name",
    "status",
    "text_for_embedding",
}

REQUIRED_BY_TYPE = {
    "attack-pattern": {
        "attack_id",
        "tactics",
        "platforms",
        "description",
        "related",
    },
    "intrusion-set": {
        "attack_id",
        "description",
        "related",
    },
    "malware": {
        "attack_id",
        "description",
        "related",
    },
    "tool": {
        "attack_id",
        "description",
        "related",
    },
    "campaign": {
        "attack_id",
        "description",
        "related",
    },
    "course-of-action": {
        "attack_id",
        "description",
        "related",
    },
    "x-mitre-detection-strategy": {
        "attack_id",
        "related",
    },
    "x-mitre-analytic": {
        "attack_id",
    },
    "x-mitre-data-source": {
        "attack_id",
    },
    "x-mitre-data-component": {
        "attack_id",
    },
    "x-mitre-tactic": {
        "attack_id",
        "description",
    },
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    docs = []
    errors = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append(
                    {
                        "line": line_number,
                        "error": str(e),
                    }
                )

    if errors:
        print("\nJSONL PARSE ERRORS")
        for err in errors[:20]:
            print(f"  line {err['line']}: {err['error']}")
        raise SystemExit("Invalid JSONL file.")

    return docs


def text_len(doc: Dict[str, Any]) -> int:
    return len(doc.get("text_for_embedding") or "")


def token_estimate(text: str) -> int:
    # Rough English-token estimate. Good enough for corpus QA.
    return max(1, len(text) // 4)


def is_empty(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        return value.strip() == ""

    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0

    return False


def check_required_fields(docs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    missing = defaultdict(list)

    for doc in docs:
        object_type = doc.get("object_type")
        required = set(REQUIRED_BASE_FIELDS)
        required.update(REQUIRED_BY_TYPE.get(object_type, set()))

        for field in required:
            if field not in doc or is_empty(doc.get(field)):
                missing[field].append(
                    {
                        "doc_id": doc.get("doc_id"),
                        "object_type": object_type,
                        "attack_id": doc.get("attack_id"),
                        "name": doc.get("name"),
                    }
                )

    return missing


def check_duplicates(docs: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups = defaultdict(list)

    for doc in docs:
        value = doc.get(key)
        if value:
            groups[value].append(doc)

    return {
        value: group
        for value, group in groups.items()
        if len(group) > 1
    }


def check_duplicate_attack_ids(docs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups = defaultdict(list)

    for doc in docs:
        object_type = doc.get("object_type")
        attack_id = doc.get("attack_id")

        if object_type and attack_id:
            key = f"{object_type}:{attack_id}"
            groups[key].append(doc)

    return {
        key: group
        for key, group in groups.items()
        if len(group) > 1
    }


def check_related_integrity(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    problems = []

    for doc in docs:
        related = doc.get("related", {})

        if not isinstance(related, dict):
            problems.append(
                {
                    "doc_id": doc.get("doc_id"),
                    "problem": "related is not a dictionary",
                }
            )
            continue

        for rel_name, items in related.items():
            if not isinstance(items, list):
                problems.append(
                    {
                        "doc_id": doc.get("doc_id"),
                        "problem": f"related.{rel_name} is not a list",
                    }
                )
                continue

            for item in items:
                if not isinstance(item, dict):
                    problems.append(
                        {
                            "doc_id": doc.get("doc_id"),
                            "problem": f"related.{rel_name} contains non-dict item",
                        }
                    )
                    continue

                if not item.get("stix_id"):
                    problems.append(
                        {
                            "doc_id": doc.get("doc_id"),
                            "problem": f"related.{rel_name} item missing stix_id",
                        }
                    )

                if not item.get("object_type"):
                    problems.append(
                        {
                            "doc_id": doc.get("doc_id"),
                            "problem": f"related.{rel_name} item missing object_type",
                        }
                    )

                if not item.get("name"):
                    problems.append(
                        {
                            "doc_id": doc.get("doc_id"),
                            "problem": f"related.{rel_name} item missing name",
                        }
                    )

    return problems


def print_counter(title: str, counter: Counter, limit: int = 50) -> None:
    print(f"\n{title}")

    if not counter:
        print("  None")
        return

    for key, count in counter.most_common(limit):
        print(f"  {str(key):40} {count}")


def print_sample_docs(title: str, docs: List[Dict[str, Any]], limit: int = 10) -> None:
    print(f"\n{title}")

    if not docs:
        print("  None")
        return

    for doc in docs[:limit]:
        print(
            f"  {doc.get('object_type')} | "
            f"{doc.get('attack_id')} | "
            f"{doc.get('name')} | "
            f"{doc.get('doc_id')}"
        )


def analyze(path: Path) -> None:
    docs = load_jsonl(path)

    if not docs:
        raise SystemExit("The JSONL file is empty.")

    print("\n=== RAG CORPUS SUMMARY ===")
    print(f"File: {path}")
    print(f"Documents: {len(docs)}")

    object_type_counts = Counter(doc.get("object_type") for doc in docs)
    status_counts = Counter(doc.get("status") for doc in docs)

    print_counter("Documents by type", object_type_counts)
    print_counter("Documents by status", status_counts)

    deprecated_docs = [
        doc for doc in docs
        if doc.get("status") != "active"
        or doc.get("revoked") is True
        or doc.get("deprecated") is True
    ]

    print_sample_docs("Deprecated/revoked documents found", deprecated_docs)

    doc_id_duplicates = check_duplicates(docs, "doc_id")
    stix_id_duplicates = check_duplicates(docs, "stix_id")
    typed_attack_id_duplicates = check_duplicate_attack_ids(docs)

    print("\n=== DUPLICATE CHECKS ===")
    print(f"Duplicate doc_id values: {len(doc_id_duplicates)}")
    print(f"Duplicate stix_id values: {len(stix_id_duplicates)}")
    print(f"Duplicate object_type:attack_id values: {len(typed_attack_id_duplicates)}")

    if typed_attack_id_duplicates:
        print("\nSample duplicate object_type:attack_id values:")
        for key, group in list(typed_attack_id_duplicates.items())[:20]:
            print(f"  {key}")
            for doc in group[:5]:
                print(f"    {doc.get('name')} | {doc.get('doc_id')}")

    missing = check_required_fields(docs)

    print("\n=== REQUIRED FIELD CHECKS ===")

    if not missing:
        print("No required-field problems found.")
    else:
        for field, items in sorted(missing.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"Missing/empty {field}: {len(items)}")
            for item in items[:5]:
                print(
                    f"  {item.get('object_type')} | "
                    f"{item.get('attack_id')} | "
                    f"{item.get('name')}"
                )

    lengths = [text_len(doc) for doc in docs]
    token_estimates = [token_estimate(doc.get("text_for_embedding") or "") for doc in docs]

    print("\n=== EMBEDDING TEXT LENGTH ===")
    print(f"Min chars:    {min(lengths)}")
    print(f"Median chars: {int(statistics.median(lengths))}")
    print(f"Mean chars:   {int(statistics.mean(lengths))}")
    print(f"Max chars:    {max(lengths)}")
    print(f"Median tokens estimate: {int(statistics.median(token_estimates))}")
    print(f"Max tokens estimate:    {max(token_estimates)}")

    empty_text_docs = [
        doc for doc in docs
        if not (doc.get("text_for_embedding") or "").strip()
    ]

    too_short_docs = [
        doc for doc in docs
        if 0 < text_len(doc) < 120
    ]

    very_long_docs = [
        doc for doc in docs
        if token_estimate(doc.get("text_for_embedding") or "") > 3000
    ]

    print_sample_docs("Documents with empty text_for_embedding", empty_text_docs)
    print_sample_docs("Suspiciously short embedding texts under 120 chars", too_short_docs)
    print_sample_docs("Very long embedding texts over about 3000 tokens", very_long_docs)

    attack_docs = [
        doc for doc in docs
        if doc.get("object_type") == "attack-pattern"
    ]

    attack_without_tactics = [
        doc for doc in attack_docs
        if not doc.get("tactics")
    ]

    attack_without_description = [
        doc for doc in attack_docs
        if not (doc.get("description") or "").strip()
    ]

    subtech_without_parent = [
        doc for doc in attack_docs
        if doc.get("is_subtechnique") is True and not doc.get("parent_technique")
    ]

    parent_techniques = [
        doc for doc in attack_docs
        if doc.get("is_subtechnique") is False
    ]

    subtechniques = [
        doc for doc in attack_docs
        if doc.get("is_subtechnique") is True
    ]

    tactic_counts = Counter()
    platform_counts = Counter()

    for doc in attack_docs:
        for tactic in doc.get("tactics", []):
            tactic_counts[tactic] += 1

        for platform in doc.get("platforms", []):
            platform_counts[platform] += 1

    print("\n=== TECHNIQUE CHECKS ===")
    print(f"Attack-pattern documents: {len(attack_docs)}")
    print(f"Parent techniques:        {len(parent_techniques)}")
    print(f"Sub-techniques:           {len(subtechniques)}")

    print_sample_docs("Techniques without tactics", attack_without_tactics)
    print_sample_docs("Techniques without description", attack_without_description)
    print_sample_docs("Sub-techniques without parent_technique", subtech_without_parent)

    print_counter("Technique counts by tactic", tactic_counts)
    print_counter("Technique counts by platform", platform_counts)

    relation_problems = check_related_integrity(docs)

    print("\n=== RELATED FIELD INTEGRITY ===")
    print(f"Related-field structural problems: {len(relation_problems)}")

    for problem in relation_problems[:20]:
        print(f"  {problem.get('doc_id')}: {problem.get('problem')}")

    related_density = Counter()

    for doc in docs:
        related = doc.get("related", {})
        if not isinstance(related, dict):
            continue

        for rel_name, items in related.items():
            if isinstance(items, list):
                related_density[rel_name] += len(items)

    print_counter("Total related-object links by relation field", related_density)

    print("\n=== SAMPLE RAG DOCUMENTS ===")

    for wanted_type in [
        "attack-pattern",
        "intrusion-set",
        "malware",
        "tool",
        "course-of-action",
        "x-mitre-detection-strategy",
    ]:
        sample = next(
            (doc for doc in docs if doc.get("object_type") == wanted_type),
            None,
        )

        if sample:
            print(f"\n--- {wanted_type} sample ---")
            print(f"ID:   {sample.get('attack_id')}")
            print(f"Name: {sample.get('name')}")
            print(f"Doc:  {sample.get('doc_id')}")
            print("Text preview:")
            print((sample.get("text_for_embedding") or "")[:1200])

    print("\n=== FINAL ASSESSMENT ===")

    hard_failures = []
    warnings = []

    if doc_id_duplicates:
        hard_failures.append("Duplicate doc_id values found.")

    if stix_id_duplicates:
        hard_failures.append("Duplicate stix_id values found.")

    if empty_text_docs:
        hard_failures.append("Some documents have empty text_for_embedding.")

    if attack_without_tactics:
        hard_failures.append("Some attack-pattern documents have no tactics.")

    if subtech_without_parent:
        hard_failures.append("Some sub-techniques have no parent_technique.")

    if deprecated_docs:
        warnings.append("Deprecated/revoked documents are present.")

    if too_short_docs:
        warnings.append("Some documents have very short embedding text.")

    if very_long_docs:
        warnings.append("Some documents are very long and may need chunking.")

    if relation_problems:
        warnings.append("Some related fields have structural issues.")

    if not hard_failures and not warnings:
        print("PASS: Corpus looks ready for embedding and vector indexing.")
    elif not hard_failures:
        print("PASS WITH WARNINGS: Corpus is usable, but review warnings before production.")
    else:
        print("FAIL: Fix hard failures before embedding.")

    if hard_failures:
        print("\nHard failures:")
        for item in hard_failures:
            print(f"  - {item}")

    if warnings:
        print("\nWarnings:")
        for item in warnings:
            print(f"  - {item}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze enriched MITRE ATT&CK JSONL RAG corpus."
    )

    parser.add_argument(
        "--input",
        default="rag_output/enterprise_attack_rag_corpus.jsonl",
        help="Path to enriched JSONL corpus.",
    )

    args = parser.parse_args()

    analyze(Path(args.input))


if __name__ == "__main__":
    main()