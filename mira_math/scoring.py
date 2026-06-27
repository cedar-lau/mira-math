"""Scoring module for mira_math v2.

Metrics tracked:
- acc_final: Whether final answer is correct (0 or 1)
- hints_used: Number of hints received from B
- hint_overuse: Hints beyond minimum needed (max(0, hints_used - k_min))
- hit_rate: Fraction of requests that resulted in offers
- rounds_to_solve: Round number when correct final was submitted
- rounds_to_final: Round number when any final was submitted
- rounds_to_correct_final: Same as rounds_to_solve (explicit name)
- request_attempts: Total number of requests made by A (lower is better)
- token_cost: Estimated token usage
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _normalize_answer(answer: Any, answer_type: str) -> Any:
    """Normalize answer to comparable format."""
    if answer is None:
        return None
    
    if answer_type == "int":
        if isinstance(answer, dict) and "value" in answer:
            if answer["value"] is None:
                return None
            return int(answer["value"])
        try:
            return int(answer)
        except (TypeError, ValueError):
            return None
    
    if answer_type == "dict_int":
        if not isinstance(answer, dict):
            return None
        try:
            return {str(k): int(v) for k, v in answer.items()}
        except (TypeError, ValueError):
            return None
    
    return answer


def _answers_match(answer1: Any, answer2: Any) -> bool:
    """Check if two answers match."""
    if answer1 is None or answer2 is None:
        return False
    return answer1 == answer2


def _count_tokens(text: str) -> int:
    """Rough token count estimate (words * 1.3)."""
    return int(len(text.split()) * 1.3)


def score_transcript(
    instance: Dict[str, Any],
    transcript: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Score a mira_math v2 transcript.

    Returns dict with metrics:
    - family               : problem family name
    - difficulty           : difficulty level (1/2/3)
    - acc_final            : 1 = correct final answer, 0 = wrong or missing
    - request_attempts     : total requests A made to B (lower = better)
    - hit_rate             : fraction of requests B accepted (offers / total responses)
    - hints_used           : number of B's offers A received
    - hint_overuse         : hints beyond the minimum needed — max(0, hints_used - k_min)
    - decline_count        : number of times B declined A's request
    - first_request_success: 1 if A received an offer on the very first request, else 0
    - requests_before_offer: requests made before B's first offer (None if B never offered)
    - rounds_to_solve      : round number when A submitted the correct answer (None if never)
    - rounds_to_final      : round number when A submitted any final answer (None if none)
    - token_cost           : estimated token count (word-count x 1.3)
    """
    answer_type = instance.get("answer_type", "dict_int")
    global_solution = instance.get("global_solution")
    k_min = instance.get("minimal_hint_spec", {}).get("k_min", 1)

    # Normalize expected answer
    expected = _normalize_answer(global_solution, answer_type)

    # Initialize counters
    request_count = 0
    offer_count = 0
    decline_count = 0
    final_answer = None
    final_round = None
    correct_final_round = None
    total_tokens = 0
    first_offer_at_request: Optional[int] = None  # request_count value when first offer arrived

    for entry in transcript:
        msg_type = entry.get("type", "")
        msg_text = entry.get("message", "")
        parsed = entry.get("parsed", {})
        round_num = entry.get("round", 0)

        # Token counting
        total_tokens += _count_tokens(msg_text)

        if msg_type == "request":
            request_count += 1
        elif msg_type == "offer":
            if offer_count == 0:
                first_offer_at_request = request_count
            offer_count += 1
        elif msg_type == "decline":
            decline_count += 1
        elif msg_type == "final":
            if final_round is None:
                final_round = round_num
                final_answer = parsed.get("answer")

                # Check correctness
                normalized = _normalize_answer(final_answer, answer_type)
                if _answers_match(normalized, expected):
                    correct_final_round = round_num

    # Compute metrics
    acc_final = 1 if correct_final_round is not None else 0
    hints_used = offer_count
    hint_overuse = max(0, hints_used - k_min)

    # Hit rate: fraction of requests that got offers
    total_responses = offer_count + decline_count
    hit_rate = offer_count / total_responses if total_responses > 0 else 0.0

    # first_request_success: B offered on the very first request (no prior declines)
    first_request_success = 1 if first_offer_at_request == 1 else 0

    # requests_before_offer: how many requests until B's first offer
    # (0 means first request was accepted, None means B never offered)
    requests_before_offer: Optional[int] = (
        (first_offer_at_request - 1) if first_offer_at_request is not None else None
    )

    return {
        "family": instance.get("family", "unknown"),
        "difficulty": instance.get("difficulty", 1),
        "acc_final": acc_final,
        "request_attempts": request_count,
        "hit_rate": hit_rate,
        "hints_used": hints_used,
        "hint_overuse": hint_overuse,
        "decline_count": decline_count,
        "first_request_success": first_request_success,
        "requests_before_offer": requests_before_offer,
        "rounds_to_solve": correct_final_round,
        "rounds_to_final": final_round,
        "rounds_to_correct_final": correct_final_round,
        "token_cost": total_tokens,
        # Diagnostic fields
        "got_correct_answer": acc_final == 1,
        "expected_answer": expected,
        "actual_answer": _normalize_answer(final_answer, answer_type) if final_answer else None,
    }


def aggregate_metrics(
    all_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate metrics across multiple instances."""
    if not all_metrics:
        return {}

    n = len(all_metrics)

    # Sum-based metrics
    total_correct = sum(m.get("acc_final", 0) for m in all_metrics)
    total_hints = sum(m.get("hints_used", 0) for m in all_metrics)
    total_overuse = sum(m.get("hint_overuse", 0) for m in all_metrics)
    total_requests = sum(m.get("request_attempts", 0) for m in all_metrics)
    total_tokens = sum(m.get("token_cost", 0) for m in all_metrics)
    total_declines = sum(m.get("decline_count", 0) for m in all_metrics)
    total_first_req_success = sum(m.get("first_request_success", 0) for m in all_metrics)

    # Averages
    avg_hit_rate = sum(m.get("hit_rate", 0) for m in all_metrics) / n

    # Rounds to solve (only for solved instances)
    solved = [m for m in all_metrics if m.get("rounds_to_solve") is not None]
    avg_rounds_to_solve = sum(m["rounds_to_solve"] for m in solved) / len(solved) if solved else None

    # requests_before_offer (only for instances where B offered at all)
    offered = [m for m in all_metrics if m.get("requests_before_offer") is not None]
    avg_requests_before_offer = (
        sum(m["requests_before_offer"] for m in offered) / len(offered) if offered else None
    )

    return {
        "n_instances": n,
        "accuracy": total_correct / n,
        "total_correct": total_correct,
        "avg_request_attempts": total_requests / n,
        "avg_hit_rate": avg_hit_rate,
        "avg_hints_used": total_hints / n,
        "avg_hint_overuse": total_overuse / n,
        "avg_declines": total_declines / n,
        "total_declines": total_declines,
        "first_request_success_rate": total_first_req_success / n,
        "avg_requests_before_offer": avg_requests_before_offer,
        "avg_rounds_to_solve": avg_rounds_to_solve,
        "avg_tokens": total_tokens / n,
        "total_tokens": total_tokens,
    }
