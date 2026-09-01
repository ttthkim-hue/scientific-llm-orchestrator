#!/usr/bin/env python3
"""Optional hosted retrieval demo using OpenAI Responses API + file_search.

The core repository remains credential-free. This script is an optional example
and intentionally does not hard-code a model ID: set OPENAI_MODEL to a currently
supported model after checking the official OpenAI documentation.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "corpus" / "evidence.jsonl"


def load_records() -> List[dict]:
    records: List[dict] = []
    with CORPUS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def require_environment() -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for the hosted demo")
    model = os.environ.get("OPENAI_MODEL")
    if not model:
        raise SystemExit(
            "OPENAI_MODEL is required. Use a current supported model ID from the official OpenAI docs."
        )
    return model


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <question>")
    question = " ".join(sys.argv[1:]).strip()
    model = require_environment()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The optional hosted demo requires the `openai` package; the core project does not."
        ) from exc

    client = OpenAI()
    records = load_records()
    vector_store = client.vector_stores.create(name="scientific-materials-evidence-rag-demo")
    uploaded_file_ids: List[str] = []

    try:
        with tempfile.TemporaryDirectory(prefix="scientific-evidence-rag-") as temp_dir:
            files = []
            for record in records:
                path = Path(temp_dir) / f"{record['evidence_id']}.md"
                path.write_text(
                    "\n".join(
                        [
                            f"# {record['evidence_id']} — {record['title']}",
                            "",
                            f"source_type: {record['source_type']}",
                            "",
                            record["text"],
                        ]
                    ),
                    encoding="utf-8",
                )
                files.append(path)

            # File-batch upload is used so ingestion is polled before retrieval.
            handles = [path.open("rb") for path in files]
            try:
                batch = client.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vector_store.id,
                    files=handles,
                )
            finally:
                for handle in handles:
                    handle.close()

            if getattr(batch, "status", None) not in {"completed", None}:
                raise RuntimeError(f"vector-store ingestion status: {batch.status}")

        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from retrieved evidence. Cite stable evidence IDs such as [E003]. "
                        "If the retrieved evidence is insufficient, say exactly: INSUFFICIENT_EVIDENCE."
                    ),
                },
                {"role": "user", "content": question},
            ],
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store.id],
                    "max_num_results": 5,
                }
            ],
            include=["file_search_call.results"],
        )

        print(response.output_text)
        print("\n--- retrieval trace ---")
        for item in getattr(response, "output", []):
            if getattr(item, "type", None) != "file_search_call":
                continue
            results = getattr(item, "results", None) or []
            for result in results:
                filename = getattr(result, "filename", "")
                score = getattr(result, "score", None)
                print(f"{filename}\tscore={score}")
        return 0
    finally:
        # Cleanup is best-effort; API method names can change, so failures here
        # must not hide the retrieval result itself.
        try:
            client.vector_stores.delete(vector_store.id)
        except Exception as exc:  # pragma: no cover - network/API dependent
            print(f"warning: vector-store cleanup failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
