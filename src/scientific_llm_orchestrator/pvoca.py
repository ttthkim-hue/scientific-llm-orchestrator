from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
import random
import re
import time
from typing import Iterable, Protocol, Sequence
from urllib import request


class Action(str, Enum):
    STOP = "STOP"
    THINK = "THINK"
    VERIFY = "VERIFY"


@dataclass(frozen=True)
class Item:
    item_id: str
    domain: str
    difficulty: str
    question: str
    answer: float
    modality: str = "text"


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    calls: int = 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class ActionResult:
    action: Action
    text: str
    predicted_value: float | None
    correct: bool
    total_tokens: int
    latency_ms: float
    calls: int


class Provider(Protocol):
    def generate(self, *, messages: Sequence[dict[str, str]], enable_thinking: bool) -> Generation:
        ...


_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def extract_last_number(text: str) -> float | None:
    matches = _NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    try:
        value = float(matches[-1])
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def numeric_correct(predicted: float | None, expected: float, *, rtol: float = 5e-2, atol: float = 1e-12) -> bool:
    if predicted is None or not math.isfinite(expected):
        return False
    return math.isclose(predicted, expected, rel_tol=rtol, abs_tol=atol)


def _result(action: Action, generation: Generation, expected: float, *, rtol: float, atol: float) -> ActionResult:
    value = extract_last_number(generation.text)
    return ActionResult(
        action=action,
        text=generation.text,
        predicted_value=value,
        correct=numeric_correct(value, expected, rtol=rtol, atol=atol),
        total_tokens=generation.total_tokens,
        latency_ms=generation.latency_ms,
        calls=generation.calls,
    )


def run_item(provider: Provider, item: Item, *, rtol: float = 5e-2, atol: float = 1e-12) -> list[ActionResult]:
    direct_messages = [
        {"role": "system", "content": "Answer the numeric materials-science question directly. Return only the final numeric answer, with units if useful. Do not show reasoning."},
        {"role": "user", "content": item.question},
    ]
    stop_gen = provider.generate(messages=direct_messages, enable_thinking=False)
    stop = _result(Action.STOP, stop_gen, item.answer, rtol=rtol, atol=atol)

    think_messages = [
        {"role": "system", "content": "Solve the numeric materials-science question carefully. Use deeper internal reasoning, but return only the final numeric answer, with units if useful."},
        {"role": "user", "content": item.question},
    ]
    think_gen = provider.generate(messages=think_messages, enable_thinking=True)
    think = _result(Action.THINK, think_gen, item.answer, rtol=rtol, atol=atol)

    verify_messages = [
        {"role": "system", "content": "Check the proposed numeric answer for arithmetic, unit, sign, and interpretation errors. If it is wrong, correct it. Return only the final numeric answer."},
        {"role": "user", "content": f"Question: {item.question}\nProposed answer: {stop_gen.text}"},
    ]
    verify_second = provider.generate(messages=verify_messages, enable_thinking=False)
    verify_gen = Generation(
        text=verify_second.text,
        prompt_tokens=stop_gen.prompt_tokens + verify_second.prompt_tokens,
        completion_tokens=stop_gen.completion_tokens + verify_second.completion_tokens,
        latency_ms=stop_gen.latency_ms + verify_second.latency_ms,
        calls=2,
    )
    verify = _result(Action.VERIFY, verify_gen, item.answer, rtol=rtol, atol=atol)
    return [stop, think, verify]


def choose_oracle(results: Sequence[ActionResult]) -> Action | None:
    correct = [r for r in results if r.correct]
    if not correct:
        return None
    rank = {Action.STOP: 0, Action.VERIFY: 1, Action.THINK: 2}
    winner = min(correct, key=lambda r: (r.total_tokens, r.latency_ms, r.calls, rank[r.action]))
    return winner.action


def balanced_pilot_sample(
    items: Iterable[Item],
    *,
    difficulties: Sequence[str] = ("easy", "medium", "hard"),
    per_difficulty: int = 36,
    seed: int = 20260921,
) -> list[Item]:
    """Difficulty-balanced pilot with broad category coverage.

    MatSciBench currently exposes 19 primary_category values, not six disjoint
    domains. For each difficulty, this sampler shuffles category buckets and
    takes one item per category in rounds until the quota is filled. This keeps
    the 108-item pilot (36 x 3) without inventing a six-domain taxonomy.
    """
    eligible = [i for i in items if i.modality.lower() == "text" and math.isfinite(i.answer)]
    rng = random.Random(seed)
    sampled: list[Item] = []
    for difficulty in difficulties:
        buckets: dict[str, list[Item]] = {}
        for item in eligible:
            if item.difficulty == difficulty:
                buckets.setdefault(item.domain, []).append(item)
        if sum(len(v) for v in buckets.values()) < per_difficulty:
            raise ValueError(
                f"insufficient items for difficulty={difficulty!r}: "
                f"{sum(len(v) for v in buckets.values())} < {per_difficulty}"
            )
        domains = sorted(buckets)
        rng.shuffle(domains)
        for values in buckets.values():
            rng.shuffle(values)

        chosen: list[Item] = []
        round_index = 0
        while len(chosen) < per_difficulty:
            progressed = False
            for domain in domains:
                bucket = buckets[domain]
                if round_index < len(bucket):
                    chosen.append(bucket[round_index])
                    progressed = True
                    if len(chosen) == per_difficulty:
                        break
            if not progressed:
                raise ValueError(f"could not fill quota for difficulty={difficulty!r}")
            round_index += 1
        sampled.extend(chosen)
    return sampled

def summarize(records: Sequence[dict]) -> dict:
    if not records:
        raise ValueError("at least one record is required")
    actions = [Action.STOP, Action.THINK, Action.VERIFY]
    accuracy = {}
    tokens = {}
    for action in actions:
        rows = [r for record in records for r in record["results"] if r.action == action]
        accuracy[action.value] = sum(r.correct for r in rows) / len(rows)
        tokens[action.value] = sum(r.total_tokens for r in rows)
    oracle_counts = {a.value: 0 for a in actions}
    solvable_oracle_counts = {a.value: 0 for a in actions}
    unsolved = 0
    oracle_tokens = 0
    rank = {Action.STOP: 0, Action.VERIFY: 1, Action.THINK: 2}
    for record in records:
        results = list(record["results"])
        oracle = record["oracle_action"]
        if oracle is None:
            unsolved += 1
            # A real policy still pays for an action even when every action is
            # wrong. Charge the cheapest observed action rather than zero.
            chosen = min(results, key=lambda r: (r.total_tokens, r.latency_ms, r.calls, rank[r.action]))
        else:
            solvable_oracle_counts[oracle.value] += 1
            chosen = next(r for r in results if r.action == oracle)
        oracle_counts[chosen.action.value] += 1
        oracle_tokens += chosen.total_tokens
    solvable = len(records) - unsolved
    oracle_accuracy = solvable / len(records)
    think_tokens = tokens[Action.THINK.value]
    savings_vs_think = 1.0 - (oracle_tokens / think_tokens) if think_tokens else 0.0
    shares = {k: (v / len(records) if records else 0.0) for k, v in oracle_counts.items()}
    heterogeneous = sum(share >= 0.15 for share in shares.values()) >= 2
    return {
        "items": len(records),
        "accuracy": accuracy,
        "tokens": tokens,
        "oracle_counts": oracle_counts,
        "solvable_oracle_counts": solvable_oracle_counts,
        "oracle_shares": shares,
        "unsolved": unsolved,
        "oracle_accuracy": oracle_accuracy,
        "oracle_tokens": oracle_tokens,
        "token_savings_vs_always_think": savings_vs_think,
        "heterogeneous_actions": heterogeneous,
    }


def records_to_jsonable(records: Sequence[dict]) -> list[dict]:
    out = []
    for record in records:
        row = {
            "item": asdict(record["item"]),
            "oracle_action": record["oracle_action"].value if record["oracle_action"] is not None else None,
            "results": [],
        }
        for result in record["results"]:
            data = asdict(result)
            data["action"] = result.action.value
            row["results"].append(data)
        out.append(row)
    return out


class OpenAICompatibleLocalProvider:
    """Minimal standard-library adapter for a local OpenAI-compatible endpoint.

    The request includes Qwen/vLLM-style chat_template_kwargs so STOP and THINK
    can differ by thinking mode without exposing chain-of-thought text.
    """

    def __init__(self, *, endpoint: str, model: str, timeout_s: float = 120.0, max_tokens: int = 2048):
        base = endpoint.rstrip("/")
        self.endpoint = base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens

    def generate(self, *, messages: Sequence[dict[str, str]], enable_thinking: bool) -> Generation:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(self.endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
        start = time.perf_counter()
        with request.urlopen(req, timeout=self.timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
        latency_ms = (time.perf_counter() - start) * 1000.0
        usage = data.get("usage") or {}
        text = data["choices"][0]["message"]["content"]
        return Generation(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
            calls=1,
        )
