#!/usr/bin/env python3
"""Scan the candidate tree and build a deterministic public manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple

MANIFEST_NAME = "publication-manifest.json"
GENERATED_OUTPUTS = frozenset({"review-bundle.json"})
MAX_FILE_BYTES = 1024 * 1024
IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
PRIVATE_URL = re.compile(
    r"https?://github\.com/[^\s/]+/(?:private|internal|confidential)"
    r"(?:[-_.][^\s/#?]+)?(?:[/?#]|\b)",
    re.IGNORECASE,
)
ABSOLUTE_PATH = re.compile(r"(?:\b[A-Za-z]:[\\/]|\\\\(?:localhost|[A-Za-z0-9_.-]+)[\\/]|/(?:Users|home|private|mnt)/)")
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
SECRET = re.compile(r"(?:sk-[A-Za-z0-9]{12,}|ghp_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{12,}|Bearer\s+[A-Za-z0-9._-]{12,}|-----BEGIN\s+(?:RSA|OPENSSH|EC|PRIVATE)\s+KEY)", re.IGNORECASE)
HIGH_ENTROPY_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9+/=_-]{32,}(?![A-Za-z0-9])")
SHA256_HEX = re.compile(r"[0-9a-f]{64}", re.IGNORECASE)
RAW_MODEL = re.compile(r"\braw_(?:prompt|response)\b|\b(?:prompt|response)_(?:text|body)\b", re.IGNORECASE)
UNSUPPORTED_CLAIM = re.compile(r"\b(?:outperform(?:s|ed)?|always cheaper|replaces frontier models|proves scientific correctness)\b", re.IGNORECASE)


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
        if path.suffix.lower() in IGNORED_SUFFIXES or path.name == MANIFEST_NAME:
            continue
        if relative.as_posix() in GENERATED_OUTPUTS:
            continue
        yield path


def _shannon_entropy(value: str) -> float:
    counts = {character: value.count(character) for character in set(value)}
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _high_entropy_credential_candidates(text: str) -> List[str]:
    candidates = []
    for match in HIGH_ENTROPY_TOKEN.finditer(text):
        token = match.group(0)
        # Manifest and other public receipts commonly contain 64-hex SHA-256
        # values. They are hashes, not credential candidates for this scan.
        if SHA256_HEX.fullmatch(token):
            continue
        categories = sum(
            (
                any(character.islower() for character in token),
                any(character.isupper() for character in token),
                any(character.isdigit() for character in token),
                any(character in "+/=_-" for character in token),
            )
        )
        if categories >= 3 and _shannon_entropy(token) >= 4.0:
            candidates.append(token)
    return candidates


def _classification(relative: str) -> str:
    suffix = Path(relative).suffix.lower()
    if relative.startswith("src/"):
        return "source-code"
    if relative.startswith("tests/"):
        return "test-code"
    if relative.startswith("scripts/"):
        return "qa-script"
    if relative.startswith("benchmarks/fixtures/"):
        return "synthetic-fixture"
    if relative.startswith("benchmarks/"):
        return "benchmark-method"
    if relative.startswith("schemas/"):
        return "schema"
    if relative.startswith("configs/"):
        return "example-config"
    if relative.startswith("docs/"):
        return "documentation"
    if relative.startswith(".github/"):
        return "ci-configuration"
    if suffix in {".md", ".cff", ".txt", ".toml", ".json"}:
        return "project-metadata"
    return "project-file"


def _license_status(relative: str) -> str:
    if relative in {"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"}:
        return "Apache-2.0-or-notice"
    return "Apache-2.0"


def _find_issues(relative: str, data: bytes) -> List[str]:
    issues: List[str] = []
    if len(data) > MAX_FILE_BYTES:
        issues.append("file exceeds 1 MiB public-MVP limit")
    if b"\x00" in data:
        issues.append("binary content is not allowed")
    if Path(relative).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".zip", ".gz", ".gguf", ".bin", ".sqlite", ".db"}:
        issues.append("binary/archive/model/database file is not allowed")
    text = data.decode("utf-8", errors="replace")
    checks = (
        (PRIVATE_URL, "private repository URL"),
        (ABSOLUTE_PATH, "local absolute path or machine path"),
        (EMAIL, "personal email address"),
        (SECRET, "credential or authentication token"),
        (RAW_MODEL, "raw prompt/response or model record"),
    )
    if relative != "scripts/scan_publication_boundary.py":
        checks = checks + ((UNSUPPORTED_CLAIM, "unsupported performance or scientific claim"),)
    for pattern, message in checks:
        if pattern.search(text):
            issues.append(message)
    if _high_entropy_credential_candidates(text):
        issues.append("high-entropy credential candidate")
    return sorted(set(issues))


def build_manifest(root: Path) -> Tuple[Dict, List[Dict[str, object]]]:
    entries: List[Dict[str, object]] = []
    for path in iter_files(root):
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        issues = _find_issues(relative, data)
        entries.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
                "classification": _classification(relative),
                "license_status": _license_status(relative),
                "reviewer_status": "pending-semantic-review",
                "public_safe_decision": "pass-deterministic-only" if not issues else "exclude",
                "reason": "public-safe candidate; human semantic review remains required" if not issues else "; ".join(issues),
            }
        )
    entries.sort(key=lambda item: str(item["path"]))
    manifest = {
        "manifest_version": "1.0",
        "self_hash_policy": "excluded-from-entries",
        "manifest_sha256": None,
        "deterministic_scan": {
            "status": "PASS" if all(item["public_safe_decision"] != "exclude" for item in entries) else "FAIL",
            "network_used": False,
            "credentials_used": False,
            "semantic_review_required": True,
        },
        "excluded_generated_outputs": sorted(GENERATED_OUTPUTS),
        "entries": entries,
    }
    return manifest, entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest, entries = build_manifest(root)
    manifest_path = root / MANIFEST_NAME
    if args.write_manifest:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif manifest_path.exists():
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != manifest:
            print("publication_scan: FAIL manifest is stale; rerun with --write-manifest")
            return 1
    else:
        print("publication_scan: FAIL publication-manifest.json is missing")
        return 1
    failures = [item for item in entries if item["public_safe_decision"] == "exclude"]
    status = "PASS" if not failures else "FAIL"
    print(f"publication_scan: {status} files={len(entries)} manifest_self_hash=excluded semantic_review=required")
    if failures:
        for item in failures:
            print(f"exclude: {item['path']} :: {item['reason']}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
