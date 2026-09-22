from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from urllib import request

ROOT = Path(__file__).resolve().parents[1]


def get_json(url: str, *, timeout: float = 10.0) -> dict:
    req = request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def installed_models(endpoint: str) -> set[str]:
    payload = get_json(endpoint.rstrip("/") + "/api/tags")
    names = set()
    for row in payload.get("models") or []:
        for key in ("name", "model"):
            if row.get(key):
                names.add(str(row[key]))
    return names


def run_checked(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def run_spec(
    *,
    model: str,
    benchmark: str,
    task_family: str,
    dataset: Path,
    results_root: Path,
    endpoint: str,
    git_commit: str,
) -> dict:
    slug = f"{benchmark.lower()}__{model.replace(':', '_').replace('/', '_')}"
    run_dir = results_root / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    traces = run_dir / "traces.jsonl"
    trace_summary = run_dir / "trace-summary.json"
    controller = run_dir / "controller-v3.json"
    runtime = run_dir / "runtime-repro.json"

    run_checked(
        [
            sys.executable,
            "scripts/run_pvoca_cascade_v3.py",
            "--dataset",
            str(dataset),
            "--output",
            str(traces),
            "--summary",
            str(trace_summary),
            "--endpoint",
            endpoint,
            "--model",
            model,
        ]
    )
    run_checked(
        [
            sys.executable,
            "scripts/train_pvoca_cascade_v3.py",
            "--traces",
            str(traces),
            "--output",
            str(controller),
            "--model-label",
            model,
            "--task-family",
            task_family,
            "--benchmark",
            benchmark,
        ]
    )
    run_checked(
        [
            sys.executable,
            "scripts/capture_pvoca_runtime_metadata.py",
            "--model",
            model,
            "--dataset",
            str(dataset),
            "--trace",
            str(traces),
            "--git-commit",
            git_commit,
            "--endpoint",
            endpoint,
            "--output",
            str(runtime),
        ]
    )
    return {
        "model": model,
        "benchmark": benchmark,
        "task_family": task_family,
        "dataset": dataset.name,
        "run_dir": run_dir.name,
        "controller": controller.name,
        "runtime_receipt": runtime.name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the isolated P-VoCA v3 experiment matrix with checkpoint/resume. "
            "This script never downloads models or changes production/control-plane state."
        )
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--matscibench", type=Path, required=True)
    parser.add_argument("--scibench", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--strong-extension",
        action="store_true",
        help="Also run the selected second model family extension.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate matrix, datasets, and model availability without generating traces.",
    )
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if matrix.get("schema") != "pvoca.experiment-matrix.v3":
        raise ValueError("expected pvoca.experiment-matrix.v3")
    if matrix.get("global_control_plane_mutation_allowed") is not False:
        raise ValueError("matrix must keep global control-plane mutation disabled")
    if matrix.get("production_deployment_allowed") is not False:
        raise ValueError("matrix must keep production deployment disabled")

    datasets = {
        "MatSciBench": args.matscibench,
        "SciBench": args.scibench,
    }
    for benchmark, path in datasets.items():
        if not path.is_file():
            raise FileNotFoundError(f"{benchmark} dataset is missing: {path}")

    minimum = matrix["minimum_matrix"]
    models = [str(row["label"]) for row in minimum["models"]]
    specs = []
    for benchmark_row in minimum["benchmarks"]:
        benchmark = str(benchmark_row["name"])
        for model in models:
            specs.append(
                {
                    "model": model,
                    "benchmark": benchmark,
                    "task_family": str(benchmark_row["task_family"]),
                    "dataset": datasets[benchmark],
                }
            )

    if args.strong_extension:
        selected = (matrix.get("strong_submission_extension") or {}).get(
            "selected_second_family"
        ) or {}
        model = str(selected.get("label") or "")
        if not model:
            raise ValueError("strong extension has no selected second-family model")
        for benchmark_row in minimum["benchmarks"]:
            benchmark = str(benchmark_row["name"])
            specs.append(
                {
                    "model": model,
                    "benchmark": benchmark,
                    "task_family": str(benchmark_row["task_family"]),
                    "dataset": datasets[benchmark],
                }
            )

    available = installed_models(args.endpoint)
    missing = sorted({spec["model"] for spec in specs if spec["model"] not in available})
    preflight = {
        "schema": "pvoca.experiment-matrix.preflight.v3",
        "requested_runs": len(specs),
        "models_required": sorted({spec["model"] for spec in specs}),
        "models_installed": sorted(set(available).intersection({spec["model"] for spec in specs})),
        "models_missing": missing,
        "automatic_model_download": False,
        "global_control_plane_mutation": False,
        "production_deployment": False,
    }
    args.results_root.mkdir(parents=True, exist_ok=True)
    (args.results_root / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(preflight, ensure_ascii=False, indent=2))

    if missing:
        raise SystemExit(
            "Missing required local models; install explicitly outside this research runner: "
            + ", ".join(missing)
        )
    if args.dry_run:
        return 0

    receipts = []
    for spec in specs:
        receipts.append(
            run_spec(
                model=spec["model"],
                benchmark=spec["benchmark"],
                task_family=spec["task_family"],
                dataset=spec["dataset"],
                results_root=args.results_root,
                endpoint=args.endpoint,
                git_commit=args.git_commit,
            )
        )

    receipt = {
        "schema": "pvoca.experiment-matrix.receipt.v3",
        "status": "COMPLETE",
        "runs": receipts,
        "global_control_plane_mutation": False,
        "production_deployment": False,
    }
    (args.results_root / "matrix-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
