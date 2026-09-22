from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
import re
from statistics import NormalDist
from typing import Sequence


TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|[^\sA-Za-z0-9]")


def _bucket(token: str, buckets: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % buckets


def _hashed(tokens: Sequence[str], *, buckets: int) -> list[float]:
    out = [0.0] * buckets
    for token in tokens:
        out[_bucket(token, buckets)] += 1.0
    return out


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def gate_s_features(trace: dict, *, buckets: int = 96) -> list[float]:
    stages = {row["action"]: row for row in trace["stages"]}
    stop = stages["STOP"]
    question = str(trace["item"]["question"])
    tokens = ["Q:" + tok for tok in TOKEN_RE.findall(question.lower())]
    x = _hashed(tokens, buckets=buckets)
    predicted = stop.get("predicted_value")
    x.extend(
        [
            math.log1p(len(question)),
            math.log1p(sum(ch.isdigit() for ch in question)),
            math.log1p(int(stop.get("incremental_tokens") or 0)),
            math.log1p(float(stop.get("incremental_latency_ms") or 0.0)) / 10.0,
            1.0 if predicted is None else 0.0,
            0.0 if predicted in (None, 0) else (1.0 if float(predicted) > 0 else -1.0),
            0.0 if predicted is None else math.log1p(abs(float(predicted))) / 20.0,
            1.0,
        ]
    )
    return _normalize(x)


def gate_v_features(trace: dict, *, buckets: int = 128) -> list[float]:
    stages = {row["action"]: row for row in trace["stages"]}
    stop = stages["STOP"]
    think = stages["THINK"]
    question = str(trace["item"]["question"])
    note = str(think.get("work_note") or "")
    tokens = (
        ["Q:" + tok for tok in TOKEN_RE.findall(question.lower())]
        + ["N:" + tok for tok in TOKEN_RE.findall(note.lower())]
    )
    x = _hashed(tokens, buckets=buckets)
    sp = stop.get("predicted_value")
    tp = think.get("predicted_value")
    if sp is None or tp is None:
        agreement = 0.0
        missing = 1.0
    else:
        scale = max(abs(float(sp)), abs(float(tp)), 1e-12)
        agreement = abs(float(sp) - float(tp)) / scale
        missing = 0.0
    x.extend(
        [
            math.log1p(len(question)),
            math.log1p(len(note)),
            math.log1p(int(stop.get("incremental_tokens") or 0)),
            math.log1p(int(think.get("incremental_tokens") or 0)),
            math.log1p(float(think.get("incremental_latency_ms") or 0.0)) / 10.0,
            math.log1p(agreement),
            missing,
            1.0,
        ]
    )
    return _normalize(x)


@dataclass
class BinaryLogit:
    weights: list[float]

    @classmethod
    def create(cls, dimension: int) -> "BinaryLogit":
        return cls([0.0] * dimension)

    def predict_proba(self, features: Sequence[float]) -> float:
        z = max(-30.0, min(30.0, sum(w * x for w, x in zip(self.weights, features))))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[int],
        *,
        epochs: int = 220,
        learning_rate: float = 0.15,
        l2: float = 1e-3,
        seed: int = 20260922,
    ) -> None:
        if not features or len(features) != len(labels):
            raise ValueError("features and labels must be non-empty and aligned")
        positives = sum(int(v) for v in labels)
        negatives = len(labels) - positives
        pos_weight = len(labels) / (2 * positives) if positives else 1.0
        neg_weight = len(labels) / (2 * negatives) if negatives else 1.0
        rng = random.Random(seed)
        order = list(range(len(labels)))
        for epoch in range(epochs):
            rng.shuffle(order)
            rate = learning_rate / math.sqrt(1.0 + epoch / 40.0)
            for index in order:
                x = features[index]
                y = int(labels[index])
                p = self.predict_proba(x)
                class_weight = pos_weight if y else neg_weight
                error = (p - y) * class_weight
                for j, value in enumerate(x):
                    self.weights[j] -= rate * (error * value + l2 * self.weights[j])


def wilson_upper(successes: int, trials: int, *, confidence: float = 0.95) -> float:
    if trials <= 0:
        return 1.0
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt((p * (1.0 - p) / trials) + z * z / (4.0 * trials * trials)) / denom
    return min(1.0, center + half)


def select_safe_stop_threshold(
    probabilities: Sequence[float],
    rescue_labels: Sequence[int],
    *,
    max_unsafe_stop_rate: float = 0.01,
    min_stop_decisions: int = 100,
    confidence: float = 0.95,
) -> float | None:
    """Largest threshold whose missed-rescue risk is statistically bounded.

    Returns None when training evidence cannot justify any STOP shortcut.
    """
    if len(probabilities) != len(rescue_labels):
        raise ValueError("probabilities and labels must align")
    candidates = sorted(set(float(p) for p in probabilities))
    safe: list[float] = []
    for threshold in candidates:
        selected = [i for i, p in enumerate(probabilities) if p <= threshold]
        if len(selected) < min_stop_decisions:
            continue
        missed = sum(int(rescue_labels[i]) for i in selected)
        if wilson_upper(missed, len(selected), confidence=confidence) <= max_unsafe_stop_rate:
            safe.append(threshold)
    return max(safe) if safe else None


def select_verify_margin(
    rescue_probabilities: Sequence[float],
    harm_probabilities: Sequence[float],
    labels_rescue: Sequence[int],
    labels_harm: Sequence[int],
    incremental_verify_tokens: Sequence[int],
    *,
    margins: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
) -> float:
    """Pick a conservative VERIFY margin using training data only.

    The objective minimizes VERIFY token use among margins that do not create
    more harmful than rescuing VERIFY decisions. If none qualify, return 1.0,
    effectively disabling VERIFY.
    """
    n = len(rescue_probabilities)
    if not (n == len(harm_probabilities) == len(labels_rescue) == len(labels_harm) == len(incremental_verify_tokens)):
        raise ValueError("verify arrays must align")
    best: tuple[int, float] | None = None
    for margin in margins:
        selected = [
            i
            for i in range(n)
            if rescue_probabilities[i] - harm_probabilities[i] >= margin
        ]
        rescues = sum(int(labels_rescue[i]) for i in selected)
        harms = sum(int(labels_harm[i]) for i in selected)
        if harms > rescues:
            continue
        cost = sum(int(incremental_verify_tokens[i]) for i in selected)
        key = (cost, -margin)
        if best is None or key < best:
            best = (cost, margin)
    return best[1] if best is not None else 1.0


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    bins: int = 10,
) -> float:
    """Standard binary ECE on [0, 1].

    Empty bins contribute zero. This is used only as an evaluation diagnostic,
    never as a training feature.
    """
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must align")
    if bins < 2:
        raise ValueError("bins must be at least two")
    if not probabilities:
        raise ValueError("at least one probability is required")
    total = len(probabilities)
    error = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        members = [
            i for i, p in enumerate(probabilities)
            if (lo <= float(p) < hi) or (b == bins - 1 and float(p) == 1.0)
        ]
        if not members:
            continue
        confidence = sum(float(probabilities[i]) for i in members) / len(members)
        accuracy = sum(int(labels[i]) for i in members) / len(members)
        error += (len(members) / total) * abs(confidence - accuracy)
    return error
