#!/usr/bin/env python3
"""mira_math Dataset Runner.

Runs all instances in a JSONL dataset file and produces aggregate metrics.
Supports multiple agent methods: llm, react, reflexion

FIXES:
- datetime.utcnow() → datetime.now(timezone.utc)
- Removed all string truncation from display output (quoted, hint, msg, reasoning)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mira_math.scoring import score_transcript, aggregate_metrics
from mira_math.families import FAMILY_TYPES
from run_helpers import print_metric_legend, print_dataset_distribution, print_type_difficulty_summary

import run_single_langgraph as rsl

# Maximum number of retries for transient API errors (content filter, rate limit, etc.)
_API_RETRY_MAX = 3
_API_RETRY_BACKOFF_BASE = 2.0  # seconds; doubles each retry


def _is_retryable_api_error(exc: Exception) -> bool:
    """Return True if the exception is a transient API error worth retrying."""
    cls_name = type(exc).__name__
    return cls_name in (
        "BadRequestError",
        "RateLimitError",
        "APITimeoutError",
        "InternalServerError",
        "APIConnectionError",
    )


def _load_env(path: str = ".env") -> None:
    """Load environment variables from .env file."""
    candidates = [path]
    if not os.path.isabs(path):
        candidates.append(os.path.join(os.path.dirname(__file__), "..", path))
    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            break


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load instances from JSONL file."""
    instances = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    return instances


def get_view_summary(instance: Dict[str, Any], agent_id: str) -> str:
    """Get a brief summary of what an agent sees — full text, no truncation."""
    for v in instance.get("agent_views", []):
        if v.get("agent_id") == agent_id:
            constraints = v.get("private_data", {}).get("constraints_text", [])
            if constraints:
                return constraints[0]
    return "?"


def main() -> None:
    parser = argparse.ArgumentParser(description="mira_math Dataset Runner")
    parser.add_argument("--in", dest="input_file", required=True,
                        help="Path to input JSONL dataset")
    parser.add_argument("--model-a", default=None, help="Model for Agent A")
    parser.add_argument("--model-b", default=None, help="Model for Agent B")
    parser.add_argument("--method", choices=["llm", "react", "reflexion"], default="llm",
                        help="Agent method: llm (single-shot), react (reasoning+acting), reflexion (self-reflection)")
    parser.add_argument("--tools", nargs="*", default=None,
                        help="Tools to give agents (e.g., calculator)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of instances to run")
    parser.add_argument("--start", type=int, default=0,
                        help="Start from this instance index")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed traces of agent reasoning (react/reflexion internals)")
    args = parser.parse_args()

    _load_env()

    model_default = os.environ.get("MIRA_MATH_MODEL", "gpt-4o-mini")
    model_a = args.model_a or os.environ.get("MIRA_MATH_MODEL_TIER_A") or os.environ.get("MIRA_MATH_MODEL_A", model_default)
    model_b = args.model_b or os.environ.get("MIRA_MATH_MODEL_TIER_B") or os.environ.get("MIRA_MATH_MODEL_B", model_default)

    tools_str = ", ".join(args.tools) if args.tools else "None"

    dataset_name = os.path.splitext(os.path.basename(args.input_file))[0]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_a = model_a.replace("/", "_")
    safe_b = model_b.replace("/", "_")
    run_id = args.run_id or f"{safe_a}_vs_{safe_b}_{dataset_name}_{timestamp}"
    os.makedirs(args.log_dir, exist_ok=True)

    logger = logging.getLogger("MIRA_MATH_dataset")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger.handlers = []
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    instances = load_jsonl(args.input_file)

    if args.start > 0:
        instances = instances[args.start:]
    if args.limit:
        instances = instances[:args.limit]

    print(f"Run: {run_id} | Dataset: {args.input_file}")
    print(f"Models: A={model_a} B={model_b}")
    print(f"Method: {args.method} | Tools: {tools_str}")
    print(f"Running {len(instances)} instances")
    print_metric_legend()
    print_dataset_distribution(instances)
    print()

    all_metrics: List[Dict[str, Any]] = []
    all_transcripts: Dict[str, List[Dict[str, Any]]] = {}
    wins = 0

    # Incremental JSONL logs so partial progress survives Ctrl+C / crashes.
    incr_metrics_path = os.path.join(args.log_dir, f"MIRA_MATH_{run_id}_results.jsonl")
    incr_transcripts_path = os.path.join(args.log_dir, f"MIRA_MATH_{run_id}_transcripts.jsonl")

    def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                fh.flush()
        except Exception as exc:  # never let logging errors kill the run
            logger.error("Failed to append to %s: %s", path, exc)

    print(f"Incremental results: {incr_metrics_path}")
    print(f"Incremental transcripts: {incr_transcripts_path}")

    interrupted = False
    last_completed_idx = -1

    try:
        for i, instance in enumerate(instances):
            instance_id = instance.get("id", f"instance-{i}")
            family = instance.get("family", "?")
            family_type = FAMILY_TYPES.get(family, "?")
            difficulty = instance.get("difficulty", 1)

            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(instances)}] {instance_id} ({family}, type={family_type}) diff={difficulty}")
            print(f"  A has: {get_view_summary(instance, 'A')}")
            print(f"  B has: {get_view_summary(instance, 'B')}")

            transcript = None
            last_error = None
            for attempt in range(1, _API_RETRY_MAX + 1):
                try:
                    transcript = rsl.run_instance(
                        instance,
                        model_a,
                        model_b,
                        method=args.method,
                        tool_names=args.tools,
                        temperature=args.temperature,
                        seed=args.seed,
                        logger=logger,
                        verbose=args.verbose,
                    )
                    last_error = None
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    last_error = e
                    if attempt < _API_RETRY_MAX and _is_retryable_api_error(e):
                        wait = _API_RETRY_BACKOFF_BASE ** attempt
                        logger.warning(
                            "Retryable error on %s (attempt %d/%d), retrying in %.1fs: %s",
                            instance_id, attempt, _API_RETRY_MAX, wait, e,
                        )
                        print(f"  --> RETRY {attempt}/{_API_RETRY_MAX} in {wait:.0f}s: {type(e).__name__}")
                        time.sleep(wait)
                    else:
                        break

            if last_error is not None or transcript is None:
                err_str = str(last_error) if last_error is not None else "unknown error (no transcript)"
                logger.error("Error running instance %s: %s", instance_id, err_str)
                print(f"  --> ERROR: {err_str}")
                metrics = {
                    "acc_final": 0,
                    "error": err_str,
                    "family_type": family_type,
                    "family": family,
                    "difficulty": difficulty,
                    "instance_id": instance_id,
                }
                all_metrics.append(metrics)
                _append_jsonl(incr_metrics_path, metrics)
            else:
                try:
                    metrics = score_transcript(instance, transcript)
                except Exception as e:
                    logger.error("Scoring failed for %s: %s", instance_id, e, exc_info=True)
                    metrics = {
                        "acc_final": 0,
                        "error": f"scoring failed: {e}",
                    }
                metrics["family_type"] = family_type
                metrics.setdefault("family", family)
                metrics.setdefault("difficulty", difficulty)
                metrics.setdefault("instance_id", instance_id)
                all_metrics.append(metrics)
                all_transcripts[instance_id] = transcript

                _append_jsonl(incr_metrics_path, metrics)
                _append_jsonl(incr_transcripts_path, {"instance_id": instance_id, "transcript": transcript})

                if metrics.get("acc_final", 0) == 1:
                    wins += 1

                print()
                for entry in transcript:
                    t = entry.get("type", "?")
                    parsed = entry.get("parsed", {})
                    if t == "request":
                        req_text = parsed.get("request", "?")
                        print(f"  A asks: \"{req_text}\"")
                    elif t == "offer":
                        quoted = parsed.get("constraint_quoted", "")
                        hint = parsed.get("hint", "?")
                        has_match = parsed.get("has_exact_match", False)
                        if quoted:
                            print(f"  B quotes: \"{quoted}\"")
                        print(f"  B gives: \"{hint}\" (exact_match={has_match})")
                    elif t == "decline":
                        msg = parsed.get("message", "declined")
                        print(f"  B declines: \"{msg}\"")
                    elif t == "final":
                        reasoning = parsed.get("reasoning", "")
                        ans = parsed.get("answer", "?")
                        if reasoning:
                            reasoning_inline = reasoning.replace("\n", " ").replace("  ", " ")
                            print(f"  A reasons: {reasoning_inline}")
                        print(f"  A answer: {ans}")

                correct = "CORRECT" if metrics.get("acc_final", 0) == 1 else "WRONG"
                req_attempts = metrics.get("request_attempts", 0)
                print(f"  --> {correct} (attempts: {req_attempts}) | Running: {wins}/{i+1}")

            last_completed_idx = i
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n\n!! Interrupted by user (Ctrl+C) after {last_completed_idx + 1}/{len(instances)} instances.")
        print("   Writing partial summary from completed instances...")
        logger.warning("KeyboardInterrupt after %d/%d instances", last_completed_idx + 1, len(instances))
    except BaseException as e:
        interrupted = True
        print(f"\n\n!! Fatal error after {last_completed_idx + 1}/{len(instances)} instances: {type(e).__name__}: {e}")
        print("   Writing partial summary from completed instances...")
        logger.error("Fatal error: %s", e, exc_info=True)

    agg = aggregate_metrics(all_metrics) if all_metrics else {}

    results_path = os.path.join(args.log_dir, f"MIRA_MATH_{run_id}_results.json")
    type_aggregates = {
        ftype: aggregate_metrics([m for m in all_metrics if m.get("family_type") == ftype])
        for ftype in ("A", "B")
        if any(m.get("family_type") == ftype for m in all_metrics)
    }
    all_diffs = sorted({m.get("difficulty") for m in all_metrics if m.get("difficulty") is not None})
    diff_aggregates = {
        str(d): aggregate_metrics([m for m in all_metrics if m.get("difficulty") == d])
        for d in all_diffs
    }
    all_families = sorted({m.get("family") for m in all_metrics if m.get("family")})
    family_aggregates = {
        fam: aggregate_metrics([m for m in all_metrics if m.get("family") == fam])
        for fam in all_families
    }
    # Difficulty x Type cross-tabulation
    diff_x_type_aggregates = {}
    for ftype in ("A", "B"):
        for d in all_diffs:
            subset = [m for m in all_metrics if m.get("family_type") == ftype and m.get("difficulty") == d]
            if subset:
                diff_x_type_aggregates[f"{ftype}_d{d}"] = aggregate_metrics(subset)
    # Difficulty x Family cross-tabulation
    diff_x_family_aggregates = {}
    for fam in all_families:
        for d in all_diffs:
            subset = [m for m in all_metrics if m.get("family") == fam and m.get("difficulty") == d]
            if subset:
                diff_x_family_aggregates[f"{fam}_d{d}"] = aggregate_metrics(subset)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "dataset": args.input_file,
            "model_a": model_a,
            "model_b": model_b,
            "method": args.method,
            "tools": args.tools if args.tools else None,
            "interrupted": interrupted,
            "completed_count": len(all_metrics),
            "total_count": len(instances),
            "aggregate_metrics": agg,
            "type_aggregate_metrics": type_aggregates,
            "difficulty_aggregate_metrics": diff_aggregates,
            "family_aggregate_metrics": family_aggregates,
            "difficulty_x_type_aggregate_metrics": diff_x_type_aggregates,
            "difficulty_x_family_aggregate_metrics": diff_x_family_aggregates,
            "per_instance_metrics": all_metrics,
        }, f, indent=2)

    transcripts_path = os.path.join(args.log_dir, f"MIRA_MATH_{run_id}_transcripts.json")
    with open(transcripts_path, "w", encoding="utf-8") as f:
        json.dump(all_transcripts, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Method: {args.method} | Tools: {tools_str}")
    print(f"Models: A={model_a} B={model_b}")
    print("-" * 60)
    print(f"Instances: {agg.get('n_instances', 0)}")
    print(f"Accuracy: {agg.get('accuracy', 0):.2%} ({agg.get('total_correct', 0)}/{agg.get('n_instances', 0)})")
    print(f"Avg Request Attempts: {agg.get('avg_request_attempts', 0):.2f}")
    print(f"Avg Declines: {agg.get('avg_declines', 0):.2f}")
    print(f"Avg Hit Rate: {agg.get('avg_hit_rate', 0):.2%}")
    print(f"Avg Hints Used: {agg.get('avg_hints_used', 0):.2f}")
    print(f"1st-Request Success: {agg.get('first_request_success_rate', 0):.2%}")
    print(f"Avg Tokens: {agg.get('avg_tokens', 0):.0f}")
    if agg.get('avg_rounds_to_solve') is not None:
        print(f"Avg Rounds to Solve: {agg.get('avg_rounds_to_solve'):.2f}")
    print_type_difficulty_summary(all_metrics)
    print("-" * 60)
    if interrupted:
        print(f"!! RUN WAS INTERRUPTED -- partial results: {len(all_metrics)}/{len(instances)} instances")
    print(f"Results saved: {results_path}")
    print(f"Transcripts saved: {transcripts_path}")
    print(f"Incremental log (per-instance): {incr_metrics_path}")


if __name__ == "__main__":
    main()