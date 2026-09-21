from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .pvoca import Action, Generation, Item, Provider, extract_last_number, numeric_correct


@dataclass(frozen=True)
class StageResult:
    action: Action
    final_text: str
    work_note: str
    predicted_value: float | None
    correct: bool
    incremental_tokens: int
    incremental_latency_ms: float
    incremental_calls: int


@dataclass(frozen=True)
class CascadeTrace:
    item: Item
    stages: tuple[StageResult, StageResult, StageResult]

    def by_action(self) -> dict[Action, StageResult]:
        return {stage.action: stage for stage in self.stages}


def _stage(
    action: Action,
    calls: Sequence[Generation],
    *,
    final_text: str,
    work_note: str,
    expected: float,
    rtol: float,
    atol: float,
) -> StageResult:
    predicted = extract_last_number(final_text)
    return StageResult(
        action=action,
        final_text=final_text,
        work_note=work_note,
        predicted_value=predicted,
        correct=numeric_correct(predicted, expected, rtol=rtol, atol=atol),
        incremental_tokens=sum(call.total_tokens for call in calls),
        incremental_latency_ms=sum(call.latency_ms for call in calls),
        incremental_calls=sum(call.calls for call in calls),
    )


def run_cascade_trace(
    provider: Provider,
    item: Item,
    *,
    rtol: float = 5e-2,
    atol: float = 1e-12,
) -> CascadeTrace:
    """Collect a full STOP -> THINK -> VERIFY counterfactual trace.

    Every stage is executed so later training has labels for marginal cognitive
    value. THINK and VERIFY use explicit observable work notes; correctness does
    not depend on access to hidden chain-of-thought.
    """
    final_only = (
        "Return exactly one numeric value only. "
        "No explanation and no unit label."
    )

    stop_call = provider.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the numeric materials-science problem directly. "
                    + final_only
                ),
            },
            {"role": "user", "content": item.question},
        ],
        enable_thinking=False,
    )
    stop = _stage(
        Action.STOP,
        [stop_call],
        final_text=stop_call.text,
        work_note="",
        expected=item.answer,
        rtol=rtol,
        atol=atol,
    )

    scaffold_call = provider.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "Create a compact observable calculation scaffold for this "
                    "materials-science problem: governing relation, converted "
                    "givens, and rough result. Maximum six short lines."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {item.question}\n"
                    f"Fast answer from STOP: {stop_call.text}"
                ),
            },
        ],
        enable_thinking=False,
    )
    think_final_call = provider.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "Recompute independently from the question and the supplied "
                    "work note. Correct any mistake in the fast answer or work "
                    "note. " + final_only
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {item.question}\n"
                    f"Fast answer: {stop_call.text}\n"
                    f"Work note:\n{scaffold_call.text}"
                ),
            },
        ],
        enable_thinking=False,
    )
    think = _stage(
        Action.THINK,
        [scaffold_call, think_final_call],
        final_text=think_final_call.text,
        work_note=scaffold_call.text,
        expected=item.answer,
        rtol=rtol,
        atol=atol,
    )

    audit_call = provider.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "Audit the current numeric answer independently. Check the "
                    "governing relation, substitutions, unit conversions, sign, "
                    "and order of magnitude. Maximum six short lines."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {item.question}\n"
                    f"STOP answer: {stop_call.text}\n"
                    f"THINK answer: {think_final_call.text}"
                ),
            },
        ],
        enable_thinking=False,
    )
    verify_final_call = provider.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "Use the independent audit to produce the corrected final "
                    "numeric answer. " + final_only
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {item.question}\n"
                    f"Current THINK answer: {think_final_call.text}\n"
                    f"Audit note:\n{audit_call.text}"
                ),
            },
        ],
        enable_thinking=False,
    )
    verify = _stage(
        Action.VERIFY,
        [audit_call, verify_final_call],
        final_text=verify_final_call.text,
        work_note=audit_call.text,
        expected=item.answer,
        rtol=rtol,
        atol=atol,
    )

    return CascadeTrace(item=item, stages=(stop, think, verify))


def gate_labels(trace: CascadeTrace) -> dict[str, int]:
    rows = trace.by_action()
    stop = rows[Action.STOP].correct
    think = rows[Action.THINK].correct
    verify = rows[Action.VERIFY].correct
    return {
        "gate_s_rescue": int((not stop) and (think or verify)),
        "gate_v_rescue": int(verify and not think),
        "gate_v_harm": int(think and not verify),
    }


def trace_to_jsonable(trace: CascadeTrace) -> dict:
    return {
        "schema": "pvoca.cascade.trace.v3",
        "item": asdict(trace.item),
        "stages": [
            {
                **asdict(stage),
                "action": stage.action.value,
            }
            for stage in trace.stages
        ],
        "labels": gate_labels(trace),
        "cost_semantics": "incremental-stage-cost",
    }
