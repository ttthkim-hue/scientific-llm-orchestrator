#!/usr/bin/env python3
"""Deterministic lexical retrieval baseline for the scientific-evidence RAG demo.

This is intentionally transparent and standard-library only. It is a lower-bound
retriever used to prove the eval harness, not a recommended production retriever.
"""

from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "corpus" / "evidence.jsonl"
EVAL_PATH = ROOT / "evals" / "retrieval_eval.jsonl"
ABSTAIN_THRESHOLD = 0.45
TOP_K = 3

STOPWORDS = frozenset(
    """
    a an the is are was were be been being of to in on at for from with without
    and or but if then than as by into over under this that these those it its
    what which who whom whose why how does do did can could should would may
    might must have has had about exact highest project author operator year 2025
    2026
    """.split()
)


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL {path}:{line_number}: {exc}") from exc
    return rows


def tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


class LexicalRetriever:
    def __init__(self, corpus: Sequence[dict]) -> None:
        self._records: Dict[str, dict] = {}
        self._tokens: Dict[str, List[str]] = {}
        document_frequency: collections.Counter[str] = collections.Counter()

        for record in corpus:
            evidence_id = str(record["evidence_id"])
            if evidence_id in self._records:
                raise ValueError(f"duplicate evidence_id: {evidence_id}")
            self._records[evidence_id] = record
            tokens = tokenize(f"{record.get('title', '')} {record.get('text', '')}")
            self._tokens[evidence_id] = tokens
            document_frequency.update(set(tokens))

        document_count = len(self._records)
        if not document_count:
            raise ValueError("empty corpus")
        self._idf = {
            token: math.log((document_count + 1) / (frequency + 1)) + 1.0
            for token, frequency in document_frequency.items()
        }

    def score(self, query: str, evidence_id: str) -> float:
        query_tokens = tokenize(query)
        if not query_tokens:
            return 0.0
        document_counts = collections.Counter(self._tokens[evidence_id])
        denominator = sum(self._idf.get(token, 2.5) for token in query_tokens) or 1.0
        numerator = 0.0
        for token in query_tokens:
            frequency = document_counts.get(token, 0)
            if frequency:
                numerator += self._idf.get(token, 2.5) * (1.0 + math.log(frequency))
        return numerator / denominator

    def rank(self, query: str) -> List[Tuple[str, float]]:
        ranked = [
            (evidence_id, self.score(query, evidence_id))
            for evidence_id in self._records
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked


def reciprocal_rank(ranked_ids: Sequence[str], expected_ids: Iterable[str]) -> float:
    expected = set(expected_ids)
    for position, evidence_id in enumerate(ranked_ids, start=1):
        if evidence_id in expected:
            return 1.0 / position
    return 0.0


def main() -> int:
    corpus = read_jsonl(CORPUS_PATH)
    eval_rows = read_jsonl(EVAL_PATH)
    retriever = LexicalRetriever(corpus)

    supported = [row for row in eval_rows if not row.get("unsupported", False)]
    unsupported = [row for row in eval_rows if row.get("unsupported", False)]

    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_ranks: List[float] = []
    abstention_correct = 0
    failures: List[str] = []

    for row in supported:
        ranked = retriever.rank(str(row["query"]))
        ranked_ids = [evidence_id for evidence_id, _ in ranked]
        expected_ids = [str(value) for value in row.get("expected_ids", [])]
        if ranked_ids and ranked_ids[0] in expected_ids:
            hit_at_1 += 1
        if any(evidence_id in expected_ids for evidence_id in ranked_ids[:TOP_K]):
            hit_at_3 += 1
        rr = reciprocal_rank(ranked_ids, expected_ids)
        reciprocal_ranks.append(rr)
        if rr == 0.0:
            failures.append(f"retrieval miss: {row['query']!r}")

    for row in unsupported:
        ranked = retriever.rank(str(row["query"]))
        top_score = ranked[0][1] if ranked else 0.0
        if top_score < ABSTAIN_THRESHOLD:
            abstention_correct += 1
        else:
            failures.append(
                f"unsupported query did not abstain: {row['query']!r} top_score={top_score:.3f}"
            )

    supported_count = len(supported) or 1
    unsupported_count = len(unsupported) or 1
    metrics = {
        "supported_queries": len(supported),
        "unsupported_queries": len(unsupported),
        "hit_at_1": hit_at_1 / supported_count,
        "hit_at_3": hit_at_3 / supported_count,
        "mrr": sum(reciprocal_ranks) / supported_count,
        "abstention_accuracy": abstention_correct / unsupported_count,
        "abstain_threshold": ABSTAIN_THRESHOLD,
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))

    # These thresholds qualify the harness/corpus only. They are not claims about
    # production retrieval quality or generalization to other scientific corpora.
    passed = (
        metrics["hit_at_1"] >= 0.95
        and metrics["hit_at_3"] >= 0.98
        and metrics["mrr"] >= 0.97
        and metrics["abstention_accuracy"] >= 0.95
    )
    if not passed:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("scientific_evidence_rag_offline_eval: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
