#!/usr/bin/env python3
"""Write dataset_manifest.json (path you choose; parent dirs are created).

The manifest may live beside the bag (sidecar) so ``ros2 bag record -o <dir>``
can create ``<dir>`` itself; do not place the manifest under ``<dir>`` before recording.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


def load_topics(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topics-file", type=Path, required=True)
    ap.add_argument("--bag-directory", type=Path, required=True, help="Same path passed to ros2 bag record -o")
    ap.add_argument("--manifest-path", type=Path, required=True)
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()

    topics = load_topics(args.topics_file)
    manifest = {
        "schema_version": "planner_record_v1",
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "bag_directory": str(args.bag_directory.resolve()),
        "topics_file": str(args.topics_file.resolve()),
        "topics": topics,
        "note": args.note,
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
