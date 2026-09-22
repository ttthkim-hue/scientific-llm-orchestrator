import json
from pathlib import Path
import tempfile
import unittest

from scientific_llm_orchestrator.pvoca_repro import (
    sanitize_gpu_rows,
    sanitize_ollama_model,
    summarize_jsonl,
)


class PvocaReproTests(unittest.TestCase):
    def test_ollama_metadata_excludes_raw_configuration(self):
        result = sanitize_ollama_model(
            "model:1",
            {"models": [{"name": "model:1", "digest": "abc", "size": 123}]},
            {
                "details": {
                    "family": "family-x",
                    "parameter_size": "27B",
                    "quantization_level": "Q4_K_M",
                },
                "capabilities": ["completion"],
                "modelfile": "SECRET_LOCAL_CONFIGURATION",
                "template": "RAW_TEMPLATE",
            },
        )
        serialized = json.dumps(result)
        self.assertNotIn("SECRET_LOCAL_CONFIGURATION", serialized)
        self.assertNotIn("RAW_TEMPLATE", serialized)
        self.assertEqual(result["digest"], "abc")
        self.assertEqual(result["details"]["quantization_level"], "Q4_K_M")

    def test_jsonl_summary_hashes_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(
                '{"id":"a","domain":"A"}\n{"id":"b","domain":"B"}\n',
                encoding="utf-8",
            )
            summary = summarize_jsonl(path)
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["unique_ids"], 2)
            self.assertEqual(summary["unique_domains"], 2)
            self.assertEqual(len(summary["sha256"]), 64)

    def test_gpu_metadata_is_bounded(self):
        rows = sanitize_gpu_rows(
            [{"name": "GPU", "driver_version": "1.2", "memory_total_mib": "49140", "serial": "private"}]
        )
        self.assertEqual(rows[0]["name"], "GPU")
        self.assertNotIn("serial", rows[0])


if __name__ == "__main__":
    unittest.main()
