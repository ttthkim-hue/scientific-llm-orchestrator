import unittest

from scientific_llm_orchestrator.pvoca import (
    Action,
    ActionResult,
    Generation,
    Item,
    choose_oracle,
    extract_last_number,
    run_item,
    balanced_pilot_sample,
    summarize,
)


class FakeProvider:
    def __init__(self):
        self.calls = 0

    def generate(self, *, messages, enable_thinking):
        self.calls += 1
        if self.calls == 1:
            return Generation("10", 10, 2, 5.0)
        if self.calls == 2:
            return Generation("12", 10, 20, 20.0)
        return Generation("12", 12, 3, 7.0)


class PvocaTests(unittest.TestCase):
    def test_extract_last_number(self):
        self.assertEqual(extract_last_number("answer = 1.25e2 MPa"), 125.0)

    def test_run_item_and_verify_cost(self):
        item = Item("x", "ceramics", "easy", "dummy", 12.0)
        rows = run_item(FakeProvider(), item)
        by_action = {r.action: r for r in rows}
        self.assertFalse(by_action[Action.STOP].correct)
        self.assertTrue(by_action[Action.THINK].correct)
        self.assertTrue(by_action[Action.VERIFY].correct)
        self.assertEqual(by_action[Action.VERIFY].calls, 2)
        self.assertEqual(by_action[Action.VERIFY].total_tokens, 27)

    def test_oracle_chooses_cheapest_correct(self):
        rows = [
            ActionResult(Action.STOP, "1", 1.0, False, 10, 5.0, 1),
            ActionResult(Action.THINK, "2", 2.0, True, 40, 20.0, 1),
            ActionResult(Action.VERIFY, "2", 2.0, True, 25, 12.0, 2),
        ]
        self.assertEqual(choose_oracle(rows), Action.VERIFY)

    def test_balanced_pilot_sample_108(self):
        items = []
        for d in range(19):
            for difficulty in ("easy", "medium", "hard"):
                for i in range(3):
                    items.append(Item(f"{d}-{difficulty}-{i}", f"D{d}", difficulty, "q", float(i)))
        sample = balanced_pilot_sample(items)
        self.assertEqual(len(sample), 108)
        for difficulty in ("easy", "medium", "hard"):
            rows = [item for item in sample if item.difficulty == difficulty]
            self.assertEqual(len(rows), 36)
            self.assertGreaterEqual(len({item.domain for item in rows}), 18)

    def test_endpoint_normalization(self):
        from scientific_llm_orchestrator.pvoca import OpenAICompatibleLocalProvider
        self.assertEqual(
            OpenAICompatibleLocalProvider(endpoint="http://127.0.0.1:8000", model="m").endpoint,
            "http://127.0.0.1:8000/v1/chat/completions",
        )
        self.assertEqual(
            OpenAICompatibleLocalProvider(endpoint="http://127.0.0.1:8000/v1", model="m").endpoint,
            "http://127.0.0.1:8000/v1/chat/completions",
        )

    def test_summary(self):
        records = []
        for i in range(4):
            rows = [
                ActionResult(Action.STOP, "1", 1.0, i < 2, 10, 5.0, 1),
                ActionResult(Action.THINK, "1", 1.0, True, 40, 20.0, 1),
                ActionResult(Action.VERIFY, "1", 1.0, i >= 2, 25, 12.0, 2),
            ]
            records.append({"item": Item(str(i), "D", "easy", "q", 1.0), "results": rows, "oracle_action": choose_oracle(rows)})
        summary = summarize(records)
        self.assertEqual(summary["oracle_counts"]["STOP"], 2)
        self.assertEqual(summary["oracle_counts"]["VERIFY"], 2)
        self.assertTrue(summary["heterogeneous_actions"])
        self.assertGreater(summary["token_savings_vs_always_think"], 0.0)


if __name__ == "__main__":
    unittest.main()
