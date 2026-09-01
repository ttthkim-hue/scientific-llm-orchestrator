from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "scientific_evidence_rag" / "offline_eval.py"


def test_scientific_evidence_rag_offline_eval() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"supported_queries": 30' in result.stdout
    assert '"unsupported_queries": 6' in result.stdout
    assert '"hit_at_1": 1.0' in result.stdout
    assert '"abstention_accuracy": 1.0' in result.stdout
    assert "scientific_evidence_rag_offline_eval: PASS" in result.stdout
