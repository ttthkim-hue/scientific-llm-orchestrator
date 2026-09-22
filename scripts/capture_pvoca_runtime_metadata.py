from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientific_llm_orchestrator.pvoca_repro import (  # noqa: E402
    sanitize_gpu_rows,
    sanitize_ollama_model,
    sha256_file,
    summarize_jsonl,
)


def get_json(url: str, *, payload: dict | None = None, timeout: float = 10.0) -> dict:
    body = None
    method = "GET"
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=body, headers=headers, method=method)
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def gpu_metadata() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    rows = []
    reader = csv.reader(io.StringIO(completed.stdout))
    for fields in reader:
        if len(fields) != 3:
            continue
        rows.append(
            {
                "name": fields[0].strip(),
                "driver_version": fields[1].strip(),
                "memory_total_mib": fields[2].strip(),
            }
        )
    return sanitize_gpu_rows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture publication-safe P-VoCA runtime reproducibility metadata."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    version = get_json(endpoint + "/api/version")
    tags = get_json(endpoint + "/api/tags")
    show = get_json(endpoint + "/api/show", payload={"model": args.model})

    receipt = {
        "schema": "pvoca.runtime-repro.v3",
        "model": sanitize_ollama_model(args.model, tags, show),
        "ollama_version": version.get("version"),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "gpus": gpu_metadata(),
        "dataset": {
            "path_basename": args.dataset.name,
            **summarize_jsonl(args.dataset),
        },
        "trace": (
            {
                "path_basename": args.trace.name,
                **summarize_jsonl(args.trace),
            }
            if args.trace is not None and args.trace.exists()
            else None
        ),
        "research_git_commit": args.git_commit,
        "privacy_boundary": (
            "No raw prompt/response archive, local absolute path, credential, "
            "Ollama Modelfile, or hidden chain-of-thought is stored."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
