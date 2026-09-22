import unittest

from scientific_llm_orchestrator.pvoca_scibench import (
    normalize_scibench_rows,
    parse_scibench_answer,
)


class PvocaSciBenchTests(unittest.TestCase):
    def test_parse_numeric_answer(self):
        self.assertEqual(parse_scibench_answer(" 0.9522 "), 0.9522)
        self.assertEqual(parse_scibench_answer("-2.5"), -2.5)
        self.assertIsNone(parse_scibench_answer("1 or 2"))

    def test_deduplicates_solution_variants_without_copying_solution(self):
        rows = [
            {
                "problem_text": "Compute x.",
                "answer_number": "2.0",
                "unit": "m",
                "source": "fund",
                "problemid": "1",
                "solution": "",
            },
            {
                "problem_text": "Compute x.",
                "answer_number": "2.0",
                "unit": "m",
                "source": "fund",
                "problemid": "1",
                "solution": "worked solution that must not leak",
            },
        ]
        normalized, skipped = normalize_scibench_rows(rows)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(skipped["duplicate"], 1)
        self.assertNotIn("solution", normalized[0])
        self.assertEqual(normalized[0]["task_family"], "college_science_numeric")


if __name__ == "__main__":
    unittest.main()
