"""Oracle-hint condition for MIRA-Math.

The standard benchmark makes Agent A (1) localize the missing constraint,
(2) phrase a request that Agent B accepts, and (3) solve the completed system.
`acc_final` therefore confounds all three abilities.

The oracle condition removes (1) and (2). The resolving hint is delivered to
Agent A as a synthetic request/offer pair in ``shared_history``, reproducing the
state of a real run just after B has made an offer; Agent B is never called. The
hint text is Agent B's own constraint string, chosen by
:func:`select_b_resolving_constraint` -- not ``apply_hint``'s rendering, which
differs from B's wording in some families. ``build_oracle_instance`` is the
verification gate behind that choice: it proves the hint flips A's view from
locally ill-posed to uniquely solvable. What remains is a pure measurement of
(3), and an upper bound on the accuracy the protocol could reach if the request
channel were free.

Folding the hint into A's private constraints instead does not work: A's prompt
still carries the "find what to request" scaffolding, so A re-requests the value
it was just handed.

Accuracy measured this way is an unconditional counterfactual over the whole
dataset. It is *not* the same as accuracy conditioned on B having made an offer,
which is biased by selection on A's own request success.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

from mira_math.validate import FAMILY_VALIDATORS

#: Per-instance metrics that carry no meaning without a request phase.
DEGENERATE_INSTANCE_KEYS: Tuple[str, ...] = (
    "request_attempts",
    "hit_rate",
    "hints_used",
    "hint_overuse",
    "decline_count",
    "first_request_success",
    "requests_before_offer",
    "rounds_to_solve",
    "rounds_to_final",
    "rounds_to_correct_final",
)

#: Aggregate metrics that carry no meaning without a request phase.
DEGENERATE_AGGREGATE_KEYS: Tuple[str, ...] = (
    "avg_request_attempts",
    "avg_hit_rate",
    "avg_hints_used",
    "avg_hint_overuse",
    "avg_declines",
    "total_declines",
    "first_request_success_rate",
    "avg_requests_before_offer",
    "avg_rounds_to_solve",
)

CONDITION_NAME = "oracle_hint"


def _get_view(instance: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    """Return the agent view with the given id.

    Args:
        instance: A mira_math instance dict.
        agent_id: Agent identifier, e.g. ``"A"``.

    Returns:
        The matching agent view dict.

    Raises:
        KeyError: If no view with that agent id exists.
    """
    for view in instance["agent_views"]:
        if view["agent_id"] == agent_id:
            return view
    raise KeyError(agent_id)


def build_oracle_instance(
    instance: Dict[str, Any],
    agent_id: str = "A",
    verify: bool = True,
) -> Tuple[Dict[str, Any], List[str]]:
    """Return a copy of `instance` with the atomic hints folded into A's view.

    Every atomic hint that lists `agent_id` among its consumers is applied to
    that agent's view via the family's own ``apply_hint``, so the injected text
    uses the family's canonical notation rather than a paraphrase.

    The runner does not prompt Agent A with this augmented view. It is used as a
    verification gate -- the hint must flip the view from locally ill-posed to
    uniquely solvable -- and the returned `injected_texts` seed the search in
    :func:`select_b_resolving_constraint`.

    Args:
        instance: A mira_math instance dict.
        agent_id: The consuming agent whose view is augmented. Defaults to "A".
        verify: If True, assert that the augmented view is uniquely solvable
            and that the original view was locally ill-posed.

    Returns:
        A tuple ``(oracle_instance, injected_texts)`` where `injected_texts`
        lists the constraint strings added to the view.

    Raises:
        KeyError: If the family is not registered in `FAMILY_VALIDATORS`.
        ValueError: If `verify` is set and the augmented view is not uniquely
            solvable, or no hint was applicable to `agent_id`.
    """
    family = instance["family"]
    if family not in FAMILY_VALIDATORS:
        raise KeyError(f"unknown family: {family}")
    is_illposed, is_unique, apply_hint, _solve_global = FAMILY_VALIDATORS[family]

    oracle_instance = copy.deepcopy(instance)
    original_view = _get_view(oracle_instance, agent_id)
    before = list(original_view["private_data"]["constraints_text"])

    if verify and not is_illposed(instance, original_view):
        raise ValueError(
            f"{instance.get('id', '?')}: agent {agent_id} view is already well-posed; "
            "the oracle condition would be vacuous"
        )

    view = original_view
    applied = 0
    for hint in instance["minimal_hint_spec"]["atomic_hints"]:
        if agent_id not in hint.get("consumers", []):
            continue
        view = apply_hint(view, hint)
        applied += 1

    if applied == 0:
        raise ValueError(
            f"{instance.get('id', '?')}: no atomic hint lists agent {agent_id} as a consumer"
        )

    if verify and not is_unique(instance, view):
        raise ValueError(
            f"{instance.get('id', '?')}: augmented agent {agent_id} view is still not "
            "uniquely solvable after applying its atomic hints"
        )

    # Splice the augmented view back in, preserving agent order.
    oracle_instance["agent_views"] = [
        view if v["agent_id"] == agent_id else v
        for v in oracle_instance["agent_views"]
    ]

    after = view["private_data"]["constraints_text"]
    injected = [c for c in after if c not in before]
    return oracle_instance, injected


def strip_instance_metrics(
    metrics: Dict[str, Any],
    injected: List[str],
) -> Dict[str, Any]:
    """Remove request-channel metrics and tag the record as an oracle run.

    The keys are deleted rather than set to ``None`` because
    :func:`mira_math.scoring.aggregate_metrics` reads them with ``.get(key, 0)``,
    which does not substitute a default for a present-but-None value.

    Args:
        metrics: A per-instance metrics dict from `score_transcript`.
        injected: Constraint strings that were injected into A's view.

    Returns:
        The same dict, mutated in place and returned for convenience.
    """
    for key in DEGENERATE_INSTANCE_KEYS:
        metrics.pop(key, None)
    metrics["condition"] = CONDITION_NAME
    metrics["hints_injected"] = len(injected)
    metrics["injected_constraints"] = injected
    return metrics


def strip_aggregate_metrics(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    """Remove aggregate fields that are meaningless without a request phase.

    Args:
        aggregate: An aggregate metrics dict from `aggregate_metrics`.

    Returns:
        The same dict, mutated in place and returned for convenience.
    """
    for key in DEGENERATE_AGGREGATE_KEYS:
        aggregate.pop(key, None)
    if aggregate:
        aggregate["condition"] = CONDITION_NAME
    return aggregate


def select_b_resolving_constraint(
    instance: Dict[str, Any],
    injected: List[str],
    agent_id: str = "A",
) -> Tuple[List[str], str]:
    """Pick the constraint(s) Agent B would quote when relaying the atomic hint.

    Agent B's private constraints and the text produced by ``apply_hint`` are
    two renderings of the same fact, and they are not always equivalent in
    difficulty. In `deconvolution`, for example, ``apply_hint`` writes the hint
    in measurement space (``y[0] = -9``) while B holds it in solution space
    (``3x0 = -9``), which is one division from the unknown. Delivering B's own
    text keeps the oracle at least as informative as a rule-following B.

    Selection order:
        1. If an ``apply_hint`` rendering appears verbatim among the constraints
           B holds but A does not, use it. This disambiguates families where B
           holds several constraints (e.g. `rankdef_linear_shared`).
        2. Otherwise, if exactly one constraint is unique to B, use it.
        3. Otherwise fall back to the ``apply_hint`` rendering.

    Args:
        instance: A mira_math instance dict.
        injected: Constraint strings produced by `build_oracle_instance`.
        agent_id: The consuming agent. Defaults to "A".

    Returns:
        A tuple ``(texts, source)`` where `source` is one of
        ``"b_exact"``, ``"b_unique"`` or ``"apply_hint_fallback"``.
    """
    views = {v["agent_id"]: v["private_data"]["constraints_text"] for v in instance["agent_views"]}
    consumer = views.get(agent_id, [])
    candidates = [c for v_id, texts in views.items() if v_id != agent_id
                  for c in texts if c not in consumer]

    exact = [c for c in candidates if c in injected]
    if exact:
        return exact, "b_exact"
    if len(candidates) == 1:
        return candidates, "b_unique"
    return list(injected), "apply_hint_fallback"
