# inspect_attack_data.py

import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("attack-stix-data")
BUNDLE_PATH = ROOT / "enterprise-attack" / "enterprise-attack.json"

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
}

def get_attack_id(obj):
    for ref in obj.get("external_references", []):
        if ref.get("source_name") in {"mitre-attack", "mitre-mobile-attack", "mitre-ics-attack"}:
            return ref.get("external_id")
    return None

def is_deprecated_or_revoked(obj):
    return obj.get("revoked") is True or obj.get("x_mitre_deprecated") is True

def main():
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Missing file: {BUNDLE_PATH}")

    with open(BUNDLE_PATH, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    objects = bundle.get("objects", [])
    by_id = {obj["id"]: obj for obj in objects if "id" in obj}

    print("\n=== Bundle summary ===")
    print("File:", BUNDLE_PATH)
    print("Bundle type:", bundle.get("type"))
    print("Bundle ID:", bundle.get("id"))
    print("Object count:", len(objects))

    print("\n=== STIX object types ===")
    type_counts = Counter(obj.get("type", "UNKNOWN") for obj in objects)
    for t, c in type_counts.most_common():
        print(f"{t:35} {c}")

    print("\n=== Active vs deprecated/revoked ===")
    deprecated_counts = Counter(
        obj.get("type") for obj in objects if is_deprecated_or_revoked(obj)
    )
    for t, c in deprecated_counts.most_common():
        print(f"{t:35} {c}")

    print("\n=== ATT&CK techniques ===")
    techniques = [
        obj for obj in objects
        if obj.get("type") == "attack-pattern"
    ]
    active_techniques = [
        obj for obj in techniques
        if not is_deprecated_or_revoked(obj)
    ]

    subtechniques = [
        obj for obj in active_techniques
        if obj.get("x_mitre_is_subtechnique") is True
    ]
    parent_techniques = [
        obj for obj in active_techniques
        if obj.get("x_mitre_is_subtechnique") is not True
    ]

    print("All attack-pattern objects:", len(techniques))
    print("Active techniques/sub-techniques:", len(active_techniques))
    print("Active parent techniques:", len(parent_techniques))
    print("Active sub-techniques:", len(subtechniques))

    print("\n=== Tactics / kill-chain phases used by techniques ===")
    tactic_counts = Counter()
    techniques_without_tactics = []

    for obj in active_techniques:
        phases = obj.get("kill_chain_phases", [])
        if not phases:
            techniques_without_tactics.append(obj)
        for phase in phases:
            if phase.get("kill_chain_name") == "mitre-attack":
                tactic_counts[phase.get("phase_name")] += 1

    for tactic, count in tactic_counts.most_common():
        print(f"{tactic:30} {count}")

    print("\nTechniques without ATT&CK tactic:", len(techniques_without_tactics))
    for obj in techniques_without_tactics[:10]:
        print(" -", get_attack_id(obj), obj.get("name"))

    print("\n=== Platforms ===")
    platform_counts = Counter()
    for obj in active_techniques:
        for platform in obj.get("x_mitre_platforms", []):
            platform_counts[platform] += 1

    for platform, count in platform_counts.most_common():
        print(f"{platform:30} {count}")

    print("\n=== Relationships ===")
    relationships = [
        obj for obj in objects
        if obj.get("type") == "relationship"
    ]

    rel_type_counts = Counter(obj.get("relationship_type") for obj in relationships)
    for rel_type, count in rel_type_counts.most_common():
        print(f"{rel_type:30} {count}")

    print("\n=== Relationship source -> target type pairs ===")
    pair_counts = Counter()
    broken_relationships = []

    for rel in relationships:
        src = by_id.get(rel.get("source_ref"))
        tgt = by_id.get(rel.get("target_ref"))

        if src is None or tgt is None:
            broken_relationships.append(rel)
            continue

        pair = (
            src.get("type"),
            rel.get("relationship_type"),
            tgt.get("type"),
        )
        pair_counts[pair] += 1

    for (src_type, rel_type, tgt_type), count in pair_counts.most_common(30):
        print(f"{src_type:25} --{rel_type:15}-> {tgt_type:25} {count}")

    print("\nBroken relationships:", len(broken_relationships))
    for rel in broken_relationships[:10]:
        print(" -", rel.get("id"), rel.get("source_ref"), "->", rel.get("target_ref"))

    print("\n=== Duplicate ATT&CK external IDs ===")
    external_id_map = defaultdict(list)

    for obj in objects:
        attack_id = get_attack_id(obj)
        if attack_id:
            external_id_map[attack_id].append(obj)

    duplicates = {
        attack_id: objs
        for attack_id, objs in external_id_map.items()
        if len(objs) > 1
    }

    print("Duplicate external IDs:", len(duplicates))
    for attack_id, objs in list(duplicates.items())[:20]:
        print(f" - {attack_id}:")
        for obj in objs:
            status = "deprecated/revoked" if is_deprecated_or_revoked(obj) else "active"
            print(f"   {obj.get('type')} | {obj.get('name')} | {status}")

    print("\n=== Indexable objects missing important fields ===")
    missing_description = []
    missing_name = []
    missing_external_id = []

    for obj in objects:
        if obj.get("type") not in INDEXABLE_TYPES:
            continue

        if is_deprecated_or_revoked(obj):
            continue

        if not obj.get("name"):
            missing_name.append(obj)

        if not obj.get("description"):
            missing_description.append(obj)

        if obj.get("type") != "x-mitre-tactic" and not get_attack_id(obj):
            missing_external_id.append(obj)

    print("Missing name:", len(missing_name))
    print("Missing description:", len(missing_description))
    print("Missing ATT&CK external ID:", len(missing_external_id))

    for obj in missing_description[:10]:
        print(" - missing description:", obj.get("type"), get_attack_id(obj), obj.get("name"))

    print("\n=== Sample technique document for RAG ===")
    sample = next(
        obj for obj in active_techniques
        if get_attack_id(obj)
    )

    sample_doc = {
        "stix_id": sample.get("id"),
        "attack_id": get_attack_id(sample),
        "type": sample.get("type"),
        "name": sample.get("name"),
        "tactics": [
            p.get("phase_name")
            for p in sample.get("kill_chain_phases", [])
            if p.get("kill_chain_name") == "mitre-attack"
        ],
        "platforms": sample.get("x_mitre_platforms", []),
        "is_subtechnique": sample.get("x_mitre_is_subtechnique", False),
        "description": sample.get("description", "")[:500] + "...",
        "detection": sample.get("x_mitre_detection", "")[:500] + "...",
    }

    print(json.dumps(sample_doc, indent=2))

    print("\n=== Recommendation ===")
    print("For RAG, index active objects only by default.")
    print("Keep deprecated/revoked objects in a separate metadata table, not in the main retrieval corpus.")
    print("Do not embed raw relationship objects directly; use them to enrich technique/group/malware/mitigation documents.")

if __name__ == "__main__":
    main()