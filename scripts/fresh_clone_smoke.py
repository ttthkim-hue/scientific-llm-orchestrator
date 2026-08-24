#!/usr/bin/env python3
"""Run the public QA contract from a network-free local fresh clone."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def _run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("fresh_clone_smoke: run " + " ".join(command))
    completed = subprocess.run(command, cwd=str(cwd), env=env, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {' '.join(command)}")


def _require_standalone_git_root(root: Path) -> None:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("--root must be a Git repository top-level")
    top_level = Path(completed.stdout.strip()).resolve()
    if top_level != root:
        raise RuntimeError(
            "refusing nested/private repository path: --root itself must be the Git top-level"
        )


def run(root: Path) -> None:
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError("--root must be an existing directory")
    _require_standalone_git_root(root)

    with tempfile.TemporaryDirectory(prefix="scientific-llm-orchestrator-smoke-") as temporary:
        temporary_root = Path(temporary)
        clone_root = temporary_root / "clone"
        # --local is a filesystem clone; it does not contact a provider or the network.
        _run(["git", "clone", "--local", "--no-hardlinks", str(root), str(clone_root)], root)

        # Run source-tree checks before the compatibility install creates
        # disposable build and egg-info artifacts inside the temporary clone.
        commands = (
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            [sys.executable, "scripts/run_static_qa.py", "--root", "."],
            [sys.executable, "scripts/run_synthetic_benchmark.py", "--root", "."],
            [sys.executable, "scripts/scan_publication_boundary.py", "--root", "."],
        )
        for command in commands:
            _run(command, clone_root)

        install_target = temporary_root / "installed"
        install_record = temporary_root / "install-record.txt"
        # Use the setuptools install command already present in the Python
        # environment. This stays offline even when the optional wheel package
        # is absent, while the normal package metadata remains in pyproject.toml.
        _run(
            [
                sys.executable,
                "setup.py",
                "install",
                "--install-lib",
                str(install_target),
                "--single-version-externally-managed",
                "--record",
                str(install_record),
            ],
            clone_root,
        )

        smoke_env = os.environ.copy()
        existing_pythonpath = smoke_env.get("PYTHONPATH")
        paths = [str(install_target)]
        if existing_pythonpath:
            paths.append(existing_pythonpath)
        smoke_env["PYTHONPATH"] = os.pathsep.join(paths)
        _run(
            [sys.executable, "-c", "import scientific_llm_orchestrator; print(scientific_llm_orchestrator.__version__)"],
            clone_root,
            smoke_env,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.root)
    except Exception as exc:
        print(f"fresh_clone_smoke: FAIL {type(exc).__name__}: {exc}")
        return 1
    print("fresh_clone_smoke: PASS local-clone=offline package=installed import=passed qa=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
