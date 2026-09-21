from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from typing import Iterable, Sequence


@dataclass(frozen=True)
class GroupFold:
    fold: int
    groups: tuple[str, ...]
    item_count: int


def stable_group_folds(
    groups: Iterable[str],
    *,
    folds: int = 5,
    seed: int = 20260922,
) -> dict[str, int]:
    """Assign each whole group to exactly one approximately balanced fold.

    This is deliberately group-level rather than item-level. Every item from a
    material category must therefore stay on the same side of a held-out fold,
    preventing category leakage between train and test.
    """
    if folds < 2:
        raise ValueError("folds must be at least two")
    counts: dict[str, int] = defaultdict(int)
    for group in groups:
        counts[str(group)] += 1
    if len(counts) < folds:
        raise ValueError("number of distinct groups must be at least folds")

    def tie_key(name: str) -> str:
        return hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()

    ordered = sorted(counts, key=lambda g: (-counts[g], tie_key(g), g))
    loads = [0] * folds
    assignment: dict[str, int] = {}
    for group in ordered:
        fold = min(range(folds), key=lambda i: (loads[i], i))
        assignment[group] = fold
        loads[fold] += counts[group]
    return assignment


def summarize_group_folds(groups: Sequence[str], assignment: dict[str, int]) -> list[GroupFold]:
    if not groups:
        return []
    fold_groups: dict[int, set[str]] = defaultdict(set)
    fold_counts: dict[int, int] = defaultdict(int)
    for group in groups:
        group = str(group)
        if group not in assignment:
            raise ValueError(f"missing assignment for group={group!r}")
        fold = int(assignment[group])
        fold_groups[fold].add(group)
        fold_counts[fold] += 1
    return [
        GroupFold(
            fold=fold,
            groups=tuple(sorted(fold_groups[fold])),
            item_count=fold_counts[fold],
        )
        for fold in sorted(fold_groups)
    ]


def validate_group_holdout(groups: Sequence[str], assignment: dict[str, int], *, folds: int) -> None:
    if set(assignment.values()) != set(range(folds)):
        raise ValueError("every fold must receive at least one group")
    for group in set(map(str, groups)):
        if group not in assignment:
            raise ValueError(f"group {group!r} has no fold")
