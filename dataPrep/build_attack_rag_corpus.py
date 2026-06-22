import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional


INDEXABLE_TYPES = {
    "attack-pattern",
    "x-mitre-tactic",
    "intrusion-set",
    "malware",
    "tool",
    "campaign",
    "course-of-action",
    "x-mitre-data-source",
    "x-mitre-data-component",
    "x-mitre-detection-strategy",
    "x-mitre-analytic",
    "x-mitre-matrix",
    "x-mitre-collection",
}


MITRE_REFERENCE_SOURCES = {
    "mitre-attack",
    "mitre-mobile-attack",
    "mitre-ics-attack",
}


TACTIC_ALIASES = {
    "stealth": ["defense-evasion", "defense evasion", "evasion"],
    "defense-impairment": ["defense-evasion", "defense evasion", "impair defenses"],
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def is_deprecated_or_revoked(obj: Dict[str, Any]) -> bool:
    return obj.get("revoked") is True or obj.get("x_mitre_deprecated") is True


def get_attack_id(obj: Dict[str, Any]) -> Optional[str]:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") in MITRE_REFERENCE_SOURCES:
            return ref.get("external_id")
    return None


def get_primary_url(obj: Dict[str, Any]) -> Optional[str]:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") in MITRE_REFERENCE_SOURCES and ref.get("url"):
            return ref.get("url")

    for ref in obj.get("external_references", []):
        if ref.get("url"):
            return ref.get("url")

    return None


def get_external_references(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    refs = []

    for ref in obj.get("external_references", []):
        refs.append(
            {
                "source_name": ref.get("source_name"),
                "external_id": ref.get("external_id"),
                "url": ref.get("url"),
                "description": clean_text(ref.get("description")),
            }
        )

    return refs


def get_tactics(obj: Dict[str, Any]) -> List[str]:
    tactics = []

    for phase in obj.get("kill_chain_phases", []):
        if phase.get("kill_chain_name") in {
            "mitre-attack",
            "mitre-mobile-attack",
            "mitre-ics-attack",
        }:
            phase_name = phase.get("phase_name")
            if phase_name:
                tactics.append(phase_name)

    return sorted(set(tactics))


def get_tactic_search_aliases(tactics: List[str]) -> List[str]:
    aliases = set()

    for tactic in tactics:
        aliases.add(tactic)
        aliases.add(tactic.replace("-", " "))

        for alias in TACTIC_ALIASES.get(tactic, []):
            aliases.add(alias)

    return sorted(aliases)


def summarize_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "stix_id": obj.get("id"),
        "object_type": obj.get("type"),
        "attack_id": get_attack_id(obj),
        "name": obj.get("name"),
        "url": get_primary_url(obj),
    }


def summarize_many(objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = [summarize_obj(obj) for obj in objs]
    summaries.sort(
        key=lambda x: (
            x.get("attack_id") or "",
            x.get("name") or "",
            x.get("stix_id") or "",
        )
    )
    return summaries


def list_names(objs: List[Dict[str, Any]]) -> List[str]:
    names = []

    for obj in objs:
        attack_id = get_attack_id(obj)
        name = obj.get("name")

        if attack_id and name:
            names.append(f"{attack_id} {name}")
        elif name:
            names.append(name)
        elif attack_id:
            names.append(attack_id)

    return sorted(set(names))


def resolve_refs(obj: Dict[str, Any], objects_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    resolved = {}

    for key, value in obj.items():
        if key.endswith("_refs") and isinstance(value, list):
            hits = []

            for stix_id in value:
                target = objects_by_id.get(stix_id)
                if target:
                    hits.append(summarize_obj(target))

            if hits:
                resolved[key] = hits

    return resolved


def add_text_section(lines: List[str], title: str, value: Any) -> None:
    if value is None:
        return

    if isinstance(value, list):
        if not value:
            return
        value = ", ".join(str(x) for x in value)

    value = clean_text(value)

    if value:
        lines.append(f"{title}: {value}")


def add_related_section(lines: List[str], title: str, objs: List[Dict[str, Any]]) -> None:
    names = list_names(objs)

    if names:
        lines.append(f"{title}: {', '.join(names)}")


def build_text_for_embedding(doc: Dict[str, Any]) -> str:
    lines = []

    object_type = doc.get("object_type")
    attack_id = doc.get("attack_id")
    name = doc.get("name")

    if attack_id and name:
        lines.append(f"MITRE ATT&CK {object_type}: {attack_id} {name}")
    elif name:
        lines.append(f"MITRE ATT&CK {object_type}: {name}")
    else:
        lines.append(f"MITRE ATT&CK {object_type}: {doc.get('stix_id')}")

    add_text_section(lines, "Object type", object_type)
    add_text_section(lines, "ATT&CK ID", attack_id)
    add_text_section(lines, "Name", name)
    add_text_section(lines, "Status", doc.get("status"))
    add_text_section(lines, "Tactics / kill-chain phases", doc.get("tactics"))
    add_text_section(lines, "Tactic search aliases", doc.get("tactic_aliases"))
    add_text_section(lines, "Platforms", doc.get("platforms"))
    add_text_section(lines, "Domains", doc.get("domains"))

    if "is_subtechnique" in doc:
        add_text_section(lines, "Is sub-technique", doc.get("is_subtechnique"))

    parent = doc.get("parent_technique")
    if parent:
        add_text_section(
            lines,
            "Parent technique",
            f"{parent.get('attack_id') or ''} {parent.get('name') or ''}".strip(),
        )

    add_text_section(lines, "Description", doc.get("description"))
    add_text_section(lines, "Detection", doc.get("detection"))

    related = doc.get("related", {})
    for key, value in related.items():
        title = key.replace("_", " ").title()
        add_related_section(lines, title, value)

    refs = doc.get("external_references", [])
    ref_text = []

    for ref in refs:
        source = ref.get("source_name")
        external_id = ref.get("external_id")
        url = ref.get("url")

        parts = [p for p in [source, external_id, url] if p]
        if parts:
            ref_text.append(" | ".join(parts))

    if ref_text:
        lines.append("External references: " + "; ".join(ref_text[:20]))

    return "\n".join(lines).strip()


def make_base_doc(
    obj: Dict[str, Any],
    bundle_id: str,
    objects_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    attack_id = get_attack_id(obj)

    doc = {
        "doc_id": obj.get("id"),
        "stix_id": obj.get("id"),
        "object_type": obj.get("type"),
        "attack_id": attack_id,
        "name": obj.get("name"),
        "status": "deprecated_or_revoked" if is_deprecated_or_revoked(obj) else "active",
        "revoked": obj.get("revoked", False),
        "deprecated": obj.get("x_mitre_deprecated", False),
        "description": clean_text(obj.get("description")),
        "created": obj.get("created"),
        "modified": obj.get("modified"),
        "x_mitre_version": obj.get("x_mitre_version"),
        "domains": obj.get("x_mitre_domains", []),
        "platforms": obj.get("x_mitre_platforms", []),
        "external_references": get_external_references(obj),
        "url": get_primary_url(obj),
        "bundle_id": bundle_id,
        "resolved_ref_fields": resolve_refs(obj, objects_by_id),
        "related": {},
    }

    detection = clean_text(obj.get("x_mitre_detection"))
    if detection:
        doc["detection"] = detection

    return doc


def build_relation_indexes(
    objects: List[Dict[str, Any]],
    objects_by_id: Dict[str, Dict[str, Any]],
    active_only: bool = True,
):
    outgoing = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    incoming = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    relationships = [obj for obj in objects if obj.get("type") == "relationship"]

    skipped = Counter()
    used_relationships = []

    for rel in relationships:
        if is_deprecated_or_revoked(rel):
            skipped["relationship_deprecated_or_revoked"] += 1
            continue

        source_ref = rel.get("source_ref")
        target_ref = rel.get("target_ref")
        relationship_type = rel.get("relationship_type")

        source = objects_by_id.get(source_ref)
        target = objects_by_id.get(target_ref)

        if not source or not target:
            skipped["broken_reference"] += 1
            continue

        if active_only and is_deprecated_or_revoked(source):
            skipped["deprecated_or_revoked_source"] += 1
            continue

        if active_only and is_deprecated_or_revoked(target):
            skipped["deprecated_or_revoked_target"] += 1
            continue

        source_type = source.get("type")
        target_type = target.get("type")

        outgoing[source_ref][relationship_type][target_type].append(target)
        incoming[target_ref][relationship_type][source_type].append(source)

        used_relationships.append(rel)

    return outgoing, incoming, used_relationships, skipped


def get_related(
    rel_index: Dict[str, Any],
    stix_id: str,
    relationship_type: str,
    object_type: str,
) -> List[Dict[str, Any]]:
    return rel_index.get(stix_id, {}).get(relationship_type, {}).get(object_type, [])


def get_propagated_tactics_and_platforms(related_techniques: List[Dict[str, Any]]) -> tuple[List[str], List[str]]:
    tactics = set()
    platforms = set()
    for tech in related_techniques:
        for tactic in get_tactics(tech):
            tactics.add(tactic)
        for platform in tech.get("x_mitre_platforms", []):
            platforms.add(platform)
    return sorted(tactics), sorted(platforms)


def enrich_doc(
    doc: Dict[str, Any],
    obj: Dict[str, Any],
    outgoing,
    incoming,
) -> Dict[str, Any]:
    stix_id = obj.get("id")
    obj_type = obj.get("type")
    related = doc["related"]

    if obj_type == "attack-pattern":
        tactics = get_tactics(obj)

        doc["tactics"] = tactics
        doc["tactic_aliases"] = get_tactic_search_aliases(tactics)
        doc["is_subtechnique"] = bool(obj.get("x_mitre_is_subtechnique", False))

        parents = get_related(outgoing, stix_id, "subtechnique-of", "attack-pattern")
        subtechniques = get_related(incoming, stix_id, "subtechnique-of", "attack-pattern")

        doc["parent_technique"] = summarize_obj(parents[0]) if parents else None
        related["subtechniques"] = summarize_many(subtechniques)

        related["used_by_groups"] = summarize_many(
            get_related(incoming, stix_id, "uses", "intrusion-set")
        )
        related["used_by_malware"] = summarize_many(
            get_related(incoming, stix_id, "uses", "malware")
        )
        related["used_by_tools"] = summarize_many(
            get_related(incoming, stix_id, "uses", "tool")
        )
        related["used_by_campaigns"] = summarize_many(
            get_related(incoming, stix_id, "uses", "campaign")
        )
        related["mitigations"] = summarize_many(
            get_related(incoming, stix_id, "mitigates", "course-of-action")
        )
        related["detection_strategies"] = summarize_many(
            get_related(incoming, stix_id, "detects", "x-mitre-detection-strategy")
        )

    elif obj_type == "course-of-action":
        techs = get_related(outgoing, stix_id, "mitigates", "attack-pattern")
        related["mitigates_techniques"] = summarize_many(techs)
        tactics, platforms = get_propagated_tactics_and_platforms(techs)
        doc["tactics"] = tactics
        doc["platforms"] = platforms

    elif obj_type == "intrusion-set":
        techs = get_related(outgoing, stix_id, "uses", "attack-pattern")
        related["uses_techniques"] = summarize_many(techs)
        related["uses_malware"] = summarize_many(
            get_related(outgoing, stix_id, "uses", "malware")
        )
        related["uses_tools"] = summarize_many(
            get_related(outgoing, stix_id, "uses", "tool")
        )
        related["attributed_campaigns"] = summarize_many(
            get_related(incoming, stix_id, "attributed-to", "campaign")
        )
        tactics, platforms = get_propagated_tactics_and_platforms(techs)
        doc["tactics"] = tactics
        doc["platforms"] = platforms

    elif obj_type == "malware":
        techs = get_related(outgoing, stix_id, "uses", "attack-pattern")
        related["uses_techniques"] = summarize_many(techs)
        related["used_by_groups"] = summarize_many(
            get_related(incoming, stix_id, "uses", "intrusion-set")
        )
        related["used_by_campaigns"] = summarize_many(
            get_related(incoming, stix_id, "uses", "campaign")
        )
        tactics, platforms = get_propagated_tactics_and_platforms(techs)
        doc["tactics"] = tactics
        orig_platforms = obj.get("x_mitre_platforms", [])
        doc["platforms"] = sorted(set(orig_platforms) | set(platforms))

    elif obj_type == "tool":
        techs = get_related(outgoing, stix_id, "uses", "attack-pattern")
        related["uses_techniques"] = summarize_many(techs)
        related["used_by_groups"] = summarize_many(
            get_related(incoming, stix_id, "uses", "intrusion-set")
        )
        related["used_by_campaigns"] = summarize_many(
            get_related(incoming, stix_id, "uses", "campaign")
        )
        tactics, platforms = get_propagated_tactics_and_platforms(techs)
        doc["tactics"] = tactics
        orig_platforms = obj.get("x_mitre_platforms", [])
        doc["platforms"] = sorted(set(orig_platforms) | set(platforms))

    elif obj_type == "campaign":
        techs = get_related(outgoing, stix_id, "uses", "attack-pattern")
        related["uses_techniques"] = summarize_many(techs)
        related["uses_malware"] = summarize_many(
            get_related(outgoing, stix_id, "uses", "malware")
        )
        related["uses_tools"] = summarize_many(
            get_related(outgoing, stix_id, "uses", "tool")
        )
        related["attributed_to_groups"] = summarize_many(
            get_related(outgoing, stix_id, "attributed-to", "intrusion-set")
        )
        tactics, platforms = get_propagated_tactics_and_platforms(techs)
        doc["tactics"] = tactics
        doc["platforms"] = platforms

    elif obj_type == "x-mitre-detection-strategy":
        techs = get_related(outgoing, stix_id, "detects", "attack-pattern")
        related["detects_techniques"] = summarize_many(techs)
        tactics, platforms = get_propagated_tactics_and_platforms(techs)
        doc["tactics"] = tactics
        doc["platforms"] = platforms

    return doc


def build_corpus(
    input_path: Path,
    output_path: Path,
    stats_path: Path,
    include_deprecated: bool = False,
) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    objects = bundle.get("objects", [])
    bundle_id = bundle.get("id")

    objects_by_id = {
        obj["id"]: obj
        for obj in objects
        if "id" in obj
    }

    outgoing, incoming, active_relationships, skipped_relationships = build_relation_indexes(
        objects=objects,
        objects_by_id=objects_by_id,
        active_only=not include_deprecated,
    )

    docs = []
    skipped_objects = Counter()

    for obj in objects:
        obj_type = obj.get("type")

        if obj_type not in INDEXABLE_TYPES:
            skipped_objects[f"not_indexable:{obj_type}"] += 1
            continue

        if not include_deprecated and is_deprecated_or_revoked(obj):
            skipped_objects[f"deprecated_or_revoked:{obj_type}"] += 1
            continue

        doc = make_base_doc(
            obj=obj,
            bundle_id=bundle_id,
            objects_by_id=objects_by_id,
        )

        doc = enrich_doc(
            doc=doc,
            obj=obj,
            outgoing=outgoing,
            incoming=incoming,
        )

        doc["text_for_embedding"] = build_text_for_embedding(doc)

        docs.append(doc)

    docs.sort(
        key=lambda d: (
            d.get("object_type") or "",
            d.get("attack_id") or "",
            d.get("name") or "",
            d.get("stix_id") or "",
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    object_type_counts = Counter(obj.get("type") for obj in objects)
    doc_type_counts = Counter(doc.get("object_type") for doc in docs)

    active_object_count = sum(
        1 for obj in objects
        if not is_deprecated_or_revoked(obj)
    )

    relationship_type_counts = Counter(
        rel.get("relationship_type")
        for rel in active_relationships
    )

    empty_embedding_text = [
        doc.get("doc_id")
        for doc in docs
        if not clean_text(doc.get("text_for_embedding"))
    ]

    missing_description_by_type = Counter(
        doc.get("object_type")
        for doc in docs
        if not clean_text(doc.get("description"))
    )

    stats = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "bundle_id": bundle_id,
        "total_objects": len(objects),
        "active_objects": active_object_count,
        "all_object_types": dict(object_type_counts.most_common()),
        "docs_written": len(docs),
        "docs_by_type": dict(doc_type_counts.most_common()),
        "relationships_used": len(active_relationships),
        "relationships_used_by_type": dict(relationship_type_counts.most_common()),
        "relationships_skipped": dict(skipped_relationships.most_common()),
        "objects_skipped": dict(skipped_objects.most_common()),
        "missing_description_by_doc_type": dict(missing_description_by_type.most_common()),
        "empty_embedding_text_docs": empty_embedding_text,
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Stats:  {stats_path}")
    print(f"Docs written: {len(docs)}")
    print("\nDocs by type:")

    for obj_type, count in doc_type_counts.most_common():
        print(f"  {obj_type:35} {count}")

    print("\nRelationships used:")

    for rel_type, count in relationship_type_counts.most_common():
        print(f"  {rel_type:35} {count}")

    print("\nMissing description by doc type:")

    for obj_type, count in missing_description_by_type.most_common():
        print(f"  {obj_type:35} {count}")

    print("\nPreview first document:")
    if docs:
        preview = {
            "doc_id": docs[0].get("doc_id"),
            "object_type": docs[0].get("object_type"),
            "attack_id": docs[0].get("attack_id"),
            "name": docs[0].get("name"),
            "text_for_embedding": docs[0].get("text_for_embedding", "")[:1000],
        }
        print(json.dumps(preview, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="Build an enriched JSONL RAG corpus from MITRE ATT&CK STIX data."
    )

    parser.add_argument(
        "--input",
        default="attack-stix-data/enterprise-attack/enterprise-attack.json",
        help="Path to ATT&CK STIX bundle JSON.",
    )

    parser.add_argument(
        "--output",
        default="rag_output/enterprise_attack_rag_corpus.jsonl",
        help="Output JSONL corpus path.",
    )

    parser.add_argument(
        "--stats",
        default="rag_output/enterprise_attack_rag_stats.json",
        help="Output stats JSON path.",
    )

    parser.add_argument(
        "--include-deprecated",
        action="store_true",
        help="Include deprecated/revoked STIX objects. Default is active objects only.",
    )

    args = parser.parse_args()

    build_corpus(
        input_path=Path(args.input),
        output_path=Path(args.output),
        stats_path=Path(args.stats),
        include_deprecated=args.include_deprecated,
    )


if __name__ == "__main__":
    main()