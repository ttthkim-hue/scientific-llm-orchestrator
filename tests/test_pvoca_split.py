import unittest

from scientific_llm_orchestrator.pvoca_split import (
    stable_group_folds,
    summarize_group_folds,
    validate_group_holdout,
)


class PvocaSplitTests(unittest.TestCase):
    def test_group_is_never_split_across_folds(self):
        groups = []
        for i in range(19):
            groups.extend([f"D{i}"] * (i + 3))
        assignment = stable_group_folds(groups, folds=5)
        validate_group_holdout(groups, assignment, folds=5)
        self.assertEqual(len(assignment), 19)
        self.assertEqual(set(assignment.values()), set(range(5)))

    def test_assignment_is_deterministic(self):
        groups = ["A"] * 20 + ["B"] * 9 + ["C"] * 7 + ["D"] * 5 + ["E"] * 4
        self.assertEqual(
            stable_group_folds(groups, folds=3, seed=7),
            stable_group_folds(groups, folds=3, seed=7),
        )

    def test_summary_counts_all_items(self):
        groups = ["A", "A", "B", "C", "C", "C"]
        assignment = stable_group_folds(groups, folds=2)
        summary = summarize_group_folds(groups, assignment)
        self.assertEqual(sum(row.item_count for row in summary), len(groups))


if __name__ == "__main__":
    unittest.main()
