"""
Append-only audit trail for Commerce Action Center activity.

Stored as JSON Lines at data/commerce_audit_log.jsonl — consistent with
this project's existing file-based (no database) conventions. Each line is
one complete audit record: timestamp, recommendation, reason, merchant
approval status, amount, payment attempt, and result.

`path` is accepted as an optional parameter on every function purely to
make this module testable in isolation (see scripts/test_commerce.py)
without touching the real audit log; normal app usage never needs to pass it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_AUDIT_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "commerce_audit_log.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_audit_record(record: Dict[str, Any], path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Appends one record to the audit trail, stamping it with a timestamp if
    not already present. Never raises — a filesystem failure is caught and
    reported back via the returned record's '_persisted' flag, so a disk
    hiccup never crashes the Commerce Action Center.
    """
    target_path = path or DEFAULT_AUDIT_LOG_PATH
    record = dict(record)
    record.setdefault("timestamp", _now_iso())

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        record["_persisted"] = True
    except OSError:
        record["_persisted"] = False

    return record


def load_audit_log(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Loads all audit records from disk, oldest first. Returns [] if the
    file doesn't exist yet or can't be read — never raises."""
    target_path = path or DEFAULT_AUDIT_LOG_PATH
    if not target_path.exists():
        return []

    records = []
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip a corrupted line rather than fail the whole load
    except OSError:
        return []
    return records


def update_audit_record_result(
    reference_id: str, new_fields: Dict[str, Any], path: Optional[Path] = None
) -> bool:
    """
    Used by the manual "Check Status" action: updates the most recent
    record matching reference_id with fresh status fields, then rewrites
    the file. A simple read-all/rewrite-all approach — perfectly fine at
    demo scale (at most a few dozen records) and avoids adding a database
    dependency. Returns True if a matching record was found and updated.
    """
    target_path = path or DEFAULT_AUDIT_LOG_PATH
    records = load_audit_log(target_path)

    updated = False
    for record in reversed(records):
        if record.get("reference_id") == reference_id:
            record.update(new_fields)
            record["status_checked_at"] = _now_iso()
            updated = True
            break

    if not updated:
        return False

    try:
        with open(target_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        return False

    return True
