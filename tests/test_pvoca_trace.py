import unittest

from scientific_llm_orchestrator.pvoca import Generation, Item
from scientific_llm_orchestrator.pvoca_trace import gate_labels, run_cascade_trace, trace_to_jsonable


class FakeProvider:
    def __init__(self):
        self.i = 0
        self.outputs = [
            "10",
            "formula note",
            "12",
            "audit note",
            "12",
        ]

    def generate(self, *, messages, enable_thinking):
        text = self.outputs[self.i]
        self.i += 1
        return Generation(
            text=text,
            prompt_tokens=10,
            completion_tokens=2,
            latency_ms=5.0,
            calls=1,
        )


class PvocaTraceTests(unittest.TestCase):
    def test_staged_trace_uses_incremental_costs(self):
        trace = run_cascade_trace(
            FakeProvider(),
            Item("x", "ceramics", "easy", "dummy", 12.0),
        )
        rows = trace.by_action()
        self.assertFalse(rows[next(a for a in rows if a.value == "STOP")].correct)
        self.assertTrue(rows[next(a for a in rows if a.value == "THINK")].correct)
        self.assertTrue(rows[next(a for a in rows if a.value == "VERIFY")].correct)
        self.assertEqual(rows[next(a for a in rows if a.value == "STOP")].incremental_tokens, 12)
        self.assertEqual(rows[next(a for a in rows if a.value == "THINK")].incremental_tokens, 24)
        self.assertEqual(rows[next(a for a in rows if a.value == "VERIFY")].incremental_tokens, 24)

    def test_gate_labels(self):
        trace = run_cascade_trace(
            FakeProvider(),
            Item("x", "ceramics", "easy", "dummy", 12.0),
        )
        labels = gate_labels(trace)
        self.assertEqual(labels["gate_s_rescue"], 1)
        self.assertEqual(labels["gate_v_rescue"], 0)
        self.assertEqual(labels["gate_v_harm"], 0)

    def test_json_schema_marker(self):
        trace = run_cascade_trace(
            FakeProvider(),
            Item("x", "ceramics", "easy", "dummy", 12.0),
        )
        row = trace_to_jsonable(trace)
        self.assertEqual(row["schema"], "pvoca.cascade.trace.v3")
        self.assertEqual(row["cost_semantics"], "incremental-stage-cost")
        self.assertEqual(len(row["stages"]), 3)


if __name__ == "__main__":
    unittest.main()
