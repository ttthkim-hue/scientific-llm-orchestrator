import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import scripts.run_pvoca_experiment_matrix_v3 as matrix_runner


class PvocaMatrixRunnerTests(unittest.TestCase):
    def test_installed_models_reads_bounded_names(self):
        with mock.patch.object(
            matrix_runner,
            "get_json",
            return_value={
                "models": [
                    {"name": "qwen3.8:27b"},
                    {"model": "gemma3:27b"},
                ]
            },
        ):
            models = matrix_runner.installed_models("http://127.0.0.1:11434")
        self.assertIn("qwen3.8:27b", models)
        self.assertIn("gemma3:27b", models)

    def test_run_checked_propagates_failure(self):
        with mock.patch.object(
            matrix_runner.subprocess,
            "run",
            return_value=mock.Mock(returncode=7),
        ):
            with self.assertRaises(SystemExit) as context:
                matrix_runner.run_checked(["python", "-V"])
        self.assertEqual(context.exception.code, 7)

    def test_matrix_contract_requires_isolation_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = {
                "schema": "pvoca.experiment-matrix.v3",
                "global_control_plane_mutation_allowed": False,
                "production_deployment_allowed": False,
                "minimum_matrix": {
                    "models": [{"label": "m1"}],
                    "benchmarks": [
                        {"name": "MatSciBench", "task_family": "materials_numeric"},
                        {"name": "SciBench", "task_family": "college_science_numeric"},
                    ],
                },
            }
            path = root / "matrix.json"
            path.write_text(json.dumps(matrix), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertIs(loaded["global_control_plane_mutation_allowed"], False)
            self.assertIs(loaded["production_deployment_allowed"], False)


if __name__ == "__main__":
    unittest.main()
