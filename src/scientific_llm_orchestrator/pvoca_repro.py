from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_jsonl(path: Path) -> dict:
    rows = 0
    ids: set[str] = set()
    domains: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows += 1
        item = row.get("item") if isinstance(row.get("item"), dict) else row
        if "id" in item:
            ids.add(str(item["id"]))
        elif "item_id" in item:
            ids.add(str(item["item_id"]))
        if "domain" in item:
            domains.add(str(item["domain"]))
    return {
        "rows": rows,
        "unique_ids": len(ids),
        "unique_domains": len(domains),
        "sha256": sha256_file(path),
    }


def sanitize_ollama_model(model: str, tags_payload: dict, show_payload: dict) -> dict:
    """Keep only publication-safe reproducibility fields.

    Raw templates, system prompts, Modelfiles, local paths, and parameter blobs
    are intentionally excluded.
    """
    models = list(tags_payload.get("models") or [])
    tag = next(
        (
            row for row in models
            if str(row.get("name") or row.get("model") or "") == model
        ),
        {},
    )
    details = dict(show_payload.get("details") or {})
    allowed_details = {
        key: details.get(key)
        for key in (
            "parent_model",
            "format",
            "family",
            "families",
            "parameter_size",
            "quantization_level",
        )
        if key in details
    }
    return {
        "model": model,
        "digest": tag.get("digest"),
        "size_bytes": tag.get("size"),
        "modified_at": tag.get("modified_at"),
        "details": allowed_details,
        "capabilities": list(show_payload.get("capabilities") or []),
    }


def sanitize_gpu_rows(rows: Iterable[dict]) -> list[dict]:
    safe = []
    for row in rows:
        safe.append(
            {
                "name": str(row.get("name") or ""),
                "driver_version": str(row.get("driver_version") or ""),
                "memory_total_mib": int(float(row.get("memory_total_mib") or 0)),
            }
        )
    return safe
