"""
build_atlas.py

CLI for building atlas JSON files for the viewer, plus a JSON manifest
(atlas_registry.json) that an HTML page can fetch to know which
atlases exist, without hardcoding atlas names/keys in JS.

Usage
-----
    python build_atlas.py list
    python build_atlas.py build <atlas_key>
    python build_atlas.py build-all
    python build_atlas.py custom <url> [name] [description] [citation]
    python build_atlas.py registry          # (re)write atlas_registry.json only
"""

import json
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    # Allow direct execution from inside scripts/atlas_viewer/ as:
    #     python build_atlas.py list
    # while still importing the atlas_viewer package from the parent
    # scripts directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_viewer import full_registry, build_atlas_json
from atlas_viewer.sources import CustomURLAtlasSource

OUTPUT_DIR = "./atlas_data"
REGISTRY_FILE = os.path.join(OUTPUT_DIR, "atlas_registry.json")
CUSTOM_ATLASES_FILE = os.path.join(OUTPUT_DIR, "custom_atlases_registry.json")


def data_filename(key):
    return f"{key}.json"


def data_path(key):
    return os.path.join(OUTPUT_DIR, data_filename(key))


def _entry_from_source(src):
    """Build one manifest entry from an AtlasSource, regardless of
    whether it's come from the Python registry or a persisted custom
    atlas record."""
    return {
        "key": src.key,
        "name": src.name,
        "description": src.description,
        "citation": src.citation,
        "license_note": src.license_note,
        "resolution": src.resolution,
        "tags": src.tags,
        "data_file": data_filename(src.key),
        "available": os.path.exists(data_path(src.key)),
    }


def _load_custom_atlas_records():
    """Custom (URL-based) atlases built via `custom` aren't part of the
    static Python registry, so their metadata is persisted here across
    separate CLI invocations."""
    if not os.path.exists(CUSTOM_ATLASES_FILE):
        return {}
    with open(CUSTOM_ATLASES_FILE) as f:
        return json.load(f)


def _save_custom_atlas_record(src):
    records = _load_custom_atlas_records()
    records[src.key] = {
        "key": src.key,
        "name": src.name,
        "description": src.description,
        "citation": src.citation,
        "license_note": src.license_note,
        "resolution": src.resolution,
        "tags": list(src.tags) + ["custom"],
        "url": src.url,
    }
    with open(CUSTOM_ATLASES_FILE, "w") as f:
        json.dump(records, f, indent=2)


def write_registry_manifest():
    """Write atlas_registry.json: one entry per atlas in the Python
    registry, plus any previously-built custom atlases, each flagged
    with whether its data file currently exists on disk."""
    entries = [_entry_from_source(src) for src in full_registry().values()]

    for record in _load_custom_atlas_records().values():
        entries.append({
            "key": record["key"],
            "name": record["name"],
            "description": record["description"],
            "citation": record["citation"],
            "license_note": record.get("license_note"),
            "resolution": record.get("resolution", "fsaverage"),
            "tags": record.get("tags", ["custom"]),
            "data_file": data_filename(record["key"]),
            "available": os.path.exists(data_path(record["key"])),
        })

    entries.sort(key=lambda e: e["name"])

    print(f"Writing registry manifest to {REGISTRY_FILE} "
          f"({sum(e['available'] for e in entries)}/{len(entries)} built)...")
    with open(REGISTRY_FILE, "w") as f:
        json.dump({"atlases": entries}, f, indent=2)


def cmd_list():
    print("Available atlases:")
    for key, src in full_registry().items():
        print(f"  {key}: {src.name} - {src.description}")


def cmd_build(key):
    registry = full_registry()
    if key not in registry:
        print(f"Atlas '{key}' not found. Available atlases:")
        for k in registry:
            print(f"  {k}")
        sys.exit(1)

    src = registry[key]
    data = build_atlas_json(src)
    output_file = data_path(key)
    print(f"Writing data to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(data, f)
    print("Done!")
    write_registry_manifest()


def cmd_build_all():
    for key, src in full_registry().items():
        print(f"\nBuilding {src.name}...")
        try:
            data = build_atlas_json(src)
            output_file = data_path(key)
            print(f"Writing data to {output_file}...")
            with open(output_file, "w") as f:
                json.dump(data, f)
            print("Done!")
        except Exception as e:
            print(f"Failed to build {src.name}: {e}")
    write_registry_manifest()


def cmd_custom(url, name=None, description=None, citation=None):
    key = (name or "custom_atlas").replace(" ", "_").lower()
    src = CustomURLAtlasSource(
        key=key,
        name=name or key,
        description=description or f"{key} atlas",
        citation=citation or "Unknown",
        url=url,
    )
    try:
        data = build_atlas_json(src)
    except Exception as e:
        print(f"Failed to process custom atlas: {e}")
        sys.exit(1)

    output_file = data_path(key)
    print(f"Writing data to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(data, f)
    print("Done!")

    _save_custom_atlas_record(src)
    write_registry_manifest()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python build_atlas.py list")
        print("  python build_atlas.py build <atlas_key>")
        print("  python build_atlas.py build-all")
        print("  python build_atlas.py custom <url> [name] [description] [citation]")
        print("  python build_atlas.py registry")
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        cmd_list()
    elif command == "build":
        if len(sys.argv) < 3:
            print("Usage: python build_atlas.py build <atlas_key>")
            sys.exit(1)
        cmd_build(sys.argv[2])
    elif command == "build-all":
        cmd_build_all()
    elif command == "custom":
        if len(sys.argv) < 3:
            print("Usage: python build_atlas.py custom <url> [name] [description] [citation]")
            sys.exit(1)
        cmd_custom(
            sys.argv[2],
            sys.argv[3] if len(sys.argv) > 3 else None,
            sys.argv[4] if len(sys.argv) > 4 else None,
            sys.argv[5] if len(sys.argv) > 5 else None,
        )
    elif command == "registry":
        write_registry_manifest()
    else:
        print("Unknown command. Use 'list', 'build <key>', 'build-all', 'custom <url>', or 'registry'")
        sys.exit(1)