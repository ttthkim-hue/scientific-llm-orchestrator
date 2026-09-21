import unittest

from scientific_llm_orchestrator.pvoca import Action
from scientific_llm_orchestrator.pvoca_controller import (
    ControllerExample,
    SoftmaxController,
    cross_validate,
    evaluate_policy,
    majority_action,
    question_features,
)


def example(question, oracle):
    outcomes = {
        Action.STOP: (oracle == Action.STOP, 10),
        Action.THINK: (True, 40),
        Action.VERIFY: (oracle == Action.VERIFY, 25),
    }
    return ControllerExample(question, oracle, outcomes)


class PvocaControllerTests(unittest.TestCase):
    def test_question_features_are_deterministic(self):
        a = question_features("Calculate 2 + 2 = ?")
        b = question_features("Calculate 2 + 2 = ?")
        self.assertEqual(a, b)
        self.assertAlmostEqual(sum(x * x for x in a), 1.0)

    def test_majority_action(self):
        rows = [
            example("easy one", Action.STOP),
            example("easy two", Action.STOP),
            example("hard", Action.THINK),
        ]
        self.assertEqual(majority_action(rows), Action.STOP)

    def test_evaluate_policy(self):
        rows = [
            example("a", Action.STOP),
            example("b", Action.VERIFY),
        ]
        accuracy, tokens = evaluate_policy(rows, [Action.STOP, Action.VERIFY])
        self.assertEqual(accuracy, 1.0)
        self.assertEqual(tokens, 35)

    def test_softmax_learns_simple_pattern(self):
        rows = []
        for i in range(20):
            rows.append(example(f"simple definition {i}", Action.STOP))
            rows.append(example(f"calculate complex equation {i} + {i} * 3", Action.THINK))
            rows.append(example(f"check sign unit error {i}", Action.VERIFY))
        model = SoftmaxController(len(question_features("x")))
        model.fit(rows, epochs=350)
        predictions = [
            model.predict(question_features("simple definition example")),
            model.predict(question_features("calculate complex equation 4 + 8 * 3")),
            model.predict(question_features("check sign unit error result")),
        ]
        self.assertEqual(predictions, [Action.STOP, Action.THINK, Action.VERIFY])

    def test_cross_validate_returns_cost_metrics(self):
        rows = []
        for i in range(18):
            rows.append(example(f"simple definition {i}", Action.STOP))
            rows.append(example(f"calculate complex equation {i} + 2", Action.THINK))
            rows.append(example(f"check sign unit error {i}", Action.VERIFY))
        metrics = cross_validate(rows, folds=3)
        self.assertEqual(metrics["items"], len(rows))
        self.assertIn("realized_accuracy", metrics)
        self.assertIn("token_savings_vs_always_think", metrics)
        self.assertGreater(metrics["always_think_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
