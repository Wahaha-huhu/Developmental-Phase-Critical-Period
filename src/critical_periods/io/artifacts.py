from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


MANIFEST_COLUMNS = [
    "timestamp_utc",
    "experiment_id",
    "artifact_type",
    "path",
    "thesis_section",
    "caption_draft",
    "source_data",
    "code_entrypoint",
    "status",
    "notes",
]


@dataclass(frozen=True)
class ArtifactRecord:
    experiment_id: str
    artifact_type: str
    path: str | Path
    thesis_section: str = ""
    caption_draft: str = ""
    source_data: str | Path = ""
    code_entrypoint: str = ""
    status: str = "draft"
    notes: str = ""

    def to_row(self) -> dict[str, str]:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "experiment_id": self.experiment_id,
            "artifact_type": self.artifact_type,
            "path": str(self.path),
            "thesis_section": self.thesis_section,
            "caption_draft": self.caption_draft,
            "source_data": str(self.source_data),
            "code_entrypoint": self.code_entrypoint,
            "status": self.status,
            "notes": self.notes,
        }


class ArtifactRegistry:
    """Append-only manifest for thesis-useful artifacts."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            with self.manifest_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
                writer.writeheader()

    def append(self, record: ArtifactRecord) -> None:
        row = record.to_row()
        with self.manifest_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            writer.writerow(row)

    def append_many(self, records: Iterable[ArtifactRecord]) -> None:
        for record in records:
            self.append(record)


def ensure_experiment_dirs(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    subdirs = {
        "root": root,
        "raw": root / "raw",
        "processed": root / "processed",
        "figures": root / "figures",
        "tables": root / "tables",
        "reports": root / "reports",
        "manifests": root / "manifests",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs
