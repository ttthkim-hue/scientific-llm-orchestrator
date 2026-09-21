from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
import re
from typing import Iterable, Sequence

from .pvoca import Action


TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|[^\sA-Za-z0-9]")
ACTIONS = (Action.STOP, Action.THINK, Action.VERIFY)
ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}


@dataclass(frozen=True)
class ControllerExample:
    question: str
    oracle_action: Action
    outcomes: dict[Action, tuple[bool, int]]


@dataclass(frozen=True)
class FoldMetrics:
    items: int
    action_accuracy: float
    realized_accuracy: float
    realized_tokens: int
    always_think_accuracy: float
    always_think_tokens: int
    oracle_accuracy: float
    oracle_tokens: int


def _stable_bucket(token: str, buckets: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % buckets


def question_features(question: str, *, buckets: int = 96) -> list[float]:
    """Public, pre-action features only.

    No benchmark difficulty label, domain label, solution text, model hidden
    state, or chain-of-thought is used by the controller.
    """
    tokens = TOKEN_RE.findall(question.lower())
    features = [0.0] * (buckets + 8)
    for token in tokens:
        features[_stable_bucket(token, buckets)] += 1.0

    length = max(len(question), 1)
    digits = sum(ch.isdigit() for ch in question)
    operators = sum(ch in "+-*/=^" for ch in question)
    parentheses = sum(ch in "()[]{}" for ch in question)
    question_marks = question.count("?")
    features[buckets + 0] = math.log1p(len(tokens))
    features[buckets + 1] = math.log1p(length)
    features[buckets + 2] = math.log1p(digits)
    features[buckets + 3] = math.log1p(operators)
    features[buckets + 4] = math.log1p(parentheses)
    features[buckets + 5] = math.log1p(question_marks)
    features[buckets + 6] = 1.0 if any(k in question.lower() for k in ("calculate", "compute", "determine", "derive")) else 0.0
    features[buckets + 7] = 1.0
    norm = math.sqrt(sum(value * value for value in features)) or 1.0
    return [value / norm for value in features]


class SoftmaxController:
    def __init__(self, dimension: int):
        self.weights = [[0.0] * dimension for _ in ACTIONS]

    def logits(self, features: Sequence[float]) -> list[float]:
        return [sum(w * x for w, x in zip(row, features)) for row in self.weights]

    def predict(self, features: Sequence[float]) -> Action:
        logits = self.logits(features)
        index = max(range(len(logits)), key=lambda i: (logits[i], -i))
        return ACTIONS[index]

    def fit(
        self,
        examples: Sequence[ControllerExample],
        *,
        epochs: int = 250,
        learning_rate: float = 0.18,
        l2: float = 1e-3,
        seed: int = 20260921,
        buckets: int = 96,
    ) -> None:
        rng = random.Random(seed)
        order = list(range(len(examples)))
        for epoch in range(epochs):
            rng.shuffle(order)
            rate = learning_rate / math.sqrt(1.0 + epoch / 30.0)
            for idx in order:
                example = examples[idx]
                x = question_features(example.question, buckets=buckets)
                logits = self.logits(x)
                peak = max(logits)
                exp_values = [math.exp(value - peak) for value in logits]
                total = sum(exp_values)
                probs = [value / total for value in exp_values]
                target = ACTION_INDEX[example.oracle_action]
                for k, row in enumerate(self.weights):
                    error = probs[k] - (1.0 if k == target else 0.0)
                    for j, value in enumerate(x):
                        row[j] -= rate * (error * value + l2 * row[j])


def majority_action(examples: Sequence[ControllerExample]) -> Action:
    counts = {action: 0 for action in ACTIONS}
    for example in examples:
        counts[example.oracle_action] += 1
    return max(ACTIONS, key=lambda action: (counts[action], -ACTION_INDEX[action]))


def evaluate_policy(examples: Sequence[ControllerExample], choices: Sequence[Action]) -> tuple[float, int]:
    if len(examples) != len(choices):
        raise ValueError("examples and choices must have the same length")
    correct = 0
    tokens = 0
    for example, action in zip(examples, choices):
        outcome_correct, outcome_tokens = example.outcomes[action]
        correct += int(outcome_correct)
        tokens += int(outcome_tokens)
    return correct / len(examples), tokens


def stratified_folds(
    examples: Sequence[ControllerExample],
    *,
    folds: int = 5,
    seed: int = 20260921,
) -> list[list[int]]:
    if folds < 2:
        raise ValueError("folds must be at least two")
    rng = random.Random(seed)
    by_action = {action: [] for action in ACTIONS}
    for index, example in enumerate(examples):
        by_action[example.oracle_action].append(index)
    split = [[] for _ in range(folds)]
    for indices in by_action.values():
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            split[offset % folds].append(index)
    return split


def cross_validate(
    examples: Sequence[ControllerExample],
    *,
    folds: int = 5,
    seed: int = 20260921,
    buckets: int = 96,
) -> dict:
    if len(examples) < folds:
        raise ValueError("not enough examples for requested folds")
    fold_indices = stratified_folds(examples, folds=folds, seed=seed)
    results: list[FoldMetrics] = []

    for fold_id, test_indices in enumerate(fold_indices):
        test_set = set(test_indices)
        train = [ex for i, ex in enumerate(examples) if i not in test_set]
        test = [examples[i] for i in test_indices]
        if not train or not test:
            continue

        dimension = buckets + 8
        model = SoftmaxController(dimension)
        model.fit(train, seed=seed + fold_id, buckets=buckets)
        predicted = [model.predict(question_features(ex.question, buckets=buckets)) for ex in test]
        action_accuracy = sum(p == ex.oracle_action for p, ex in zip(predicted, test)) / len(test)
        realized_accuracy, realized_tokens = evaluate_policy(test, predicted)

        think = [Action.THINK] * len(test)
        always_think_accuracy, always_think_tokens = evaluate_policy(test, think)
        oracle = [ex.oracle_action for ex in test]
        oracle_accuracy, oracle_tokens = evaluate_policy(test, oracle)

        results.append(
            FoldMetrics(
                items=len(test),
                action_accuracy=action_accuracy,
                realized_accuracy=realized_accuracy,
                realized_tokens=realized_tokens,
                always_think_accuracy=always_think_accuracy,
                always_think_tokens=always_think_tokens,
                oracle_accuracy=oracle_accuracy,
                oracle_tokens=oracle_tokens,
            )
        )

    total_items = sum(row.items for row in results)
    if not total_items:
        raise ValueError("cross-validation produced no test items")

    weighted = lambda attr: sum(getattr(row, attr) * row.items for row in results) / total_items
    realized_tokens = sum(row.realized_tokens for row in results)
    think_tokens = sum(row.always_think_tokens for row in results)
    oracle_tokens = sum(row.oracle_tokens for row in results)

    return {
        "items": total_items,
        "folds": len(results),
        "action_accuracy": weighted("action_accuracy"),
        "realized_accuracy": weighted("realized_accuracy"),
        "realized_tokens": realized_tokens,
        "always_think_accuracy": weighted("always_think_accuracy"),
        "always_think_tokens": think_tokens,
        "oracle_accuracy": weighted("oracle_accuracy"),
        "oracle_tokens": oracle_tokens,
        "token_savings_vs_always_think": 1.0 - (realized_tokens / think_tokens) if think_tokens else 0.0,
        "oracle_token_savings_vs_always_think": 1.0 - (oracle_tokens / think_tokens) if think_tokens else 0.0,
    }
