#!/usr/bin/env python3
"""mira_math Dataset Runner — Parallel (LangGraph).

Runs all instances in a JSONL dataset file and produces aggregate metrics.
Supports multiple agent methods: llm, react, reflexion.
Parallel execution via ``--workers N`` (asyncio.to_thread wrapping sync LangGraph runner).

Usage:
    # Sequential (default)
    python run_dataset_langgraph_parallel.py --in datasets/test.jsonl --method react

    # Parallel (8 workers)
    python run_dataset_langgraph_parallel.py --in datasets/test.jsonl --method react --workers 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from mira_math.scoring import score_transcript, aggregate_metrics
from mira_math.oracle import strip_instance_metrics, strip_aggregate_metrics
from mira_math.families import FAMILY_TYPES
from run_helpers import print_metric_legend, print_dataset_distribution, print_type_difficulty_summary

import run_single_langgraph as rsl

# Maximum number of retries for transient API errors (content filter, rate limit, etc.)
_API_RETRY_MAX = 5
_API_RETRY_BACKOFF_BASE = 5.0  # seconds; multiplied each retry (5, 25, 125, ...)


def _is_retryable_api_error(exc: Exception) -> bool:
    """Return True if the exception is a transient API error worth retrying."""
    cls_name = type(exc).__name__
    # OpenAI errors
    if cls_name in ("BadRequestError", "RateLimitError", "APITimeoutError",
                    "InternalServerError", "APIConnectionError"):
        return True
    # Gemini / google-genai errors: ServerError covers 429, 500, 503, 504
    if cls_name == "ServerError":
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        return status in (429, 500, 503, 504) if status else True
    # Network-level errors (DNS failures, connection resets, etc.)
    if cls_name in ("ConnectionError", "ReadTimeout", "ConnectTimeout",
                    "ChunkedEncodingError", "RequestException"):
        return True
    # Catch-all for any error whose message mentions transient conditions
    msg = str(exc).lower()
    return any(k in msg for k in ("deadline_exceeded", "unavailable", "connection",
                                   "getaddrinfo", "timeout", "rate limit"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _append_jsonl(path: str, obj: Dict[str, Any], log: Optional[logging.Logger] = None) -> None:
    """Append a single JSON object as one line; flush so Ctrl+C preserves it."""
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fh.flush()
    except Exception as exc:
        if log is not None:
            log.error("Failed to append to %s: %s", path, exc)


def get_view_summary(instance: Dict[str, Any], agent_id: str) -> str:
    """Get a brief summary of what an agent sees — full text, no truncation."""
    for v in instance.get("agent_views", []):
        if v.get("agent_id") == agent_id:
            constraints = v.get("private_data", {}).get("constraints_text", [])
            if constraints:
                return constraints[0]
    return "?"


def _display_transcript(transcript: List[Dict[str, Any]]) -> None:
    """Print per-entry transcript details."""
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


# ---------------------------------------------------------------------------
# Sequential execution  (workers=1, preserves original display behaviour)
# ---------------------------------------------------------------------------

def _run_sequential(
    instances: List[Dict[str, Any]],
    model_a: str,
    model_b: str,
    method: str,
    tool_names: Optional[List[str]],
    temperature: float,
    seed: Optional[int],
    verbose: bool,
    log: logging.Logger,
    oracle: bool = False,
    incr_metrics_path: Optional[str] = None,
    incr_transcripts_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], bool]:
    all_metrics: List[Dict[str, Any]] = []
    all_transcripts: Dict[str, List[Dict[str, Any]]] = {}
    wins = 0
    total = len(instances)
    interrupted = False

    def _emit_partial(metrics_obj: Dict[str, Any], inst_id: str, transcript_obj: Optional[List[Dict[str, Any]]]) -> None:
        if incr_metrics_path:
            _append_jsonl(incr_metrics_path, metrics_obj, log)
        if transcript_obj and incr_transcripts_path:
            _append_jsonl(incr_transcripts_path, {"instance_id": inst_id, "transcript": transcript_obj}, log)

    try:
        for i, instance in enumerate(instances):
            instance_id = instance.get("id", f"instance-{i}")
            family = instance.get("family", "?")
            family_type = FAMILY_TYPES.get(family, "?")
            difficulty = instance.get("difficulty", 1)

            print(f"\n{'='*60}")
            print(f"[{i+1}/{total}] {instance_id} ({family}, type={family_type}) diff={difficulty}")
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
                        method=method,
                        tool_names=tool_names,
                        temperature=temperature,
                        seed=seed,
                        logger=log,
                        verbose=verbose,
                        oracle=oracle,
                    )
                    last_error = None
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    last_error = e
                    if attempt < _API_RETRY_MAX and _is_retryable_api_error(e):
                        wait = _API_RETRY_BACKOFF_BASE ** attempt
                        log.warning(
                            "Retryable error on %s (attempt %d/%d), retrying in %.1fs: %s",
                            instance_id, attempt, _API_RETRY_MAX, wait, e,
                        )
                        print(f"  --> RETRY {attempt}/{_API_RETRY_MAX} in {wait:.0f}s: {type(e).__name__}")
                        time.sleep(wait)
                    else:
                        break

            if last_error is not None or transcript is None:
                err_str = str(last_error) if last_error is not None else "unknown error (no transcript)"
                log.error("Error running instance %s: %s", instance_id, err_str)
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
                _emit_partial(metrics, instance_id, None)
            else:
                try:
                    metrics = score_transcript(instance, transcript)
                    if oracle:
                        metrics = strip_instance_metrics(
                            metrics, transcript[0].get("injected_constraints", []) if transcript else [],
                        )
                except Exception as e:
                    log.error("Scoring failed for %s: %s", instance_id, e, exc_info=True)
                    metrics = {"acc_final": 0, "error": f"scoring failed: {e}"}
                metrics["family_type"] = family_type
                metrics.setdefault("family", family)
                metrics.setdefault("difficulty", difficulty)
                metrics["instance_id"] = instance_id
                all_metrics.append(metrics)
                all_transcripts[instance_id] = transcript
                _emit_partial(metrics, instance_id, transcript)

                if metrics.get("acc_final", 0) == 1:
                    wins += 1

                print()
                _display_transcript(transcript)

                correct = "CORRECT" if metrics.get("acc_final", 0) == 1 else "WRONG"
                req_attempts = metrics.get("request_attempts", 0)
                print(f"  --> {correct} (attempts: {req_attempts}) | Running: {wins}/{i+1}")
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n\n!! Interrupted by user (Ctrl+C) after {len(all_metrics)}/{total} instances.")
        log.warning("KeyboardInterrupt after %d/%d instances", len(all_metrics), total)
    except BaseException as e:
        interrupted = True
        print(f"\n\n!! Fatal error after {len(all_metrics)}/{total} instances: {type(e).__name__}: {e}")
        log.error("Fatal error: %s", e, exc_info=True)

    return all_metrics, all_transcripts, interrupted


# ---------------------------------------------------------------------------
# Parallel execution  (workers>1, asyncio.to_thread)
# ---------------------------------------------------------------------------

async def _run_parallel(
    instances: List[Dict[str, Any]],
    model_a: str,
    model_b: str,
    method: str,
    tool_names: Optional[List[str]],
    temperature: float,
    seed: Optional[int],
    workers: int,
    verbose: bool,
    log: logging.Logger,
    oracle: bool = False,
    instance_timeout: int = 3600,
    incr_metrics_path: Optional[str] = None,
    incr_transcripts_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], bool]:
    """Run instances in parallel with bounded concurrency via asyncio.to_thread."""
    sem = asyncio.Semaphore(workers)
    print_lock = asyncio.Lock()
    total = len(instances)

    # Shared mutable counters — safe because asyncio is single-threaded.
    ctr = {"done": 0, "wins": 0}

    # Pre-allocate per-index slots so results stay in dataset order.
    all_metrics: List[Dict[str, Any]] = [{}] * total
    all_transcripts: Dict[str, List[Dict[str, Any]]] = {}

    async def run_one(idx: int, instance: Dict[str, Any]) -> None:
        instance_id = instance.get("id", f"instance-{idx}")
        family = instance.get("family", "?")
        family_type = FAMILY_TYPES.get(family, "?")

        async with sem:
            t0 = time.monotonic()
            transcript = None  # type: ignore[assignment]
            metrics = None
            for attempt in range(1, _API_RETRY_MAX + 1):
                try:
                    # run_instance is synchronous (LangGraph + sync LLM calls),
                    # so we offload it to a thread pool worker.
                    transcript = await asyncio.wait_for(
                        asyncio.to_thread(
                            rsl.run_instance,
                            instance,
                            model_a,
                            model_b,
                            method,
                            tool_names,
                            temperature,
                            seed,
                            log,
                            False,  # verbose suppressed in parallel to avoid interleaving
                            oracle,
                        ),
                        timeout=instance_timeout,
                    )
                    metrics = score_transcript(instance, transcript)
                    if oracle:
                        metrics = strip_instance_metrics(
                            metrics, transcript[0].get("injected_constraints", []) if transcript else [],
                        )
                    metrics["family_type"] = family_type
                    metrics["instance_id"] = instance_id
                    break
                except asyncio.TimeoutError:
                    log.error("Instance %s timed out after %ds", instance_id, instance_timeout)
                    metrics = {"acc_final": 0, "error": f"timeout after {instance_timeout}s", "family_type": family_type, "instance_id": instance_id}
                    break  # timeouts are not retryable
                except Exception as e:
                    if attempt < _API_RETRY_MAX and _is_retryable_api_error(e):
                        wait = _API_RETRY_BACKOFF_BASE ** attempt
                        log.warning(
                            "Retryable error on %s (attempt %d/%d), retrying in %.1fs: %s",
                            instance_id, attempt, _API_RETRY_MAX, wait, e,
                        )
                        await asyncio.sleep(wait)
                    else:
                        log.error("Error on %s: %s", instance_id, e, exc_info=True)
                        metrics = {"acc_final": 0, "error": str(e), "family_type": family_type, "instance_id": instance_id}
                        break
            elapsed = time.monotonic() - t0

        # Update shared counters & print under lock for clean output
        async with print_lock:
            ctr["done"] += 1
            if metrics.get("acc_final", 0) == 1:
                ctr["wins"] += 1

            done = ctr["done"]
            wins = ctr["wins"]
            err = metrics.get("error")

            if err:
                print(
                    f"  [{done}/{total}] {instance_id} ({family}, type={family_type})"
                    f"  ERROR: {err}  ({elapsed:.1f}s)"
                )
            else:
                solve_lbl = "CORRECT" if metrics.get("acc_final") else "WRONG"
                attempts = metrics.get("request_attempts", 0)
                print(
                    f"  [{done}/{total}] {instance_id} ({family}, type={family_type})"
                    f"  {solve_lbl} (attempts: {attempts})  ({elapsed:.1f}s)"
                    f"  | running: {wins}/{done}"
                )
                if verbose and transcript:
                    _display_transcript(transcript)

        all_metrics[idx] = metrics
        if transcript:
            all_transcripts[instance_id] = transcript

        # Persist incrementally so Ctrl+C / crashes don't lose completed work.
        if incr_metrics_path:
            _append_jsonl(incr_metrics_path, metrics, log)
        if transcript and incr_transcripts_path:
            _append_jsonl(
                incr_transcripts_path,
                {"instance_id": instance_id, "transcript": transcript},
                log,
            )

    tasks = [asyncio.create_task(run_one(i, inst)) for i, inst in enumerate(instances)]
    interrupted = False
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except (KeyboardInterrupt, asyncio.CancelledError):
        interrupted = True
        async with print_lock:
            print(f"\n!! Interrupted -- cancelling outstanding tasks ({ctr['done']}/{total} completed)...")
        for t in tasks:
            if not t.done():
                t.cancel()
        # Drain cancellations so any task that was about to commit its result
        # gets a chance to do so via run_one's normal path.
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except BaseException:
            pass

    return all_metrics, all_transcripts, interrupted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="mira_math Dataset Runner (LangGraph, parallel)")
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
    parser.add_argument("--workers", "-j", type=int, default=1,
                        help="Parallel workers via asyncio.to_thread (default: 1 = sequential)")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of instances to run")
    parser.add_argument("--start", type=int, default=0,
                        help="Start from this instance index")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed traces of agent reasoning (react/reflexion internals)")
    parser.add_argument("--instance-timeout", type=int, default=3600,
                        help="Per-instance wall-clock timeout in seconds for parallel mode (default: 3600)")
    parser.add_argument("--oracle", action="store_true",
                        help="Oracle-hint condition: pre-inject A's atomic hint into its own view "
                             "and skip Agent B entirely. Measures solving ability in isolation and "
                             "gives an upper bound on protocol accuracy. Request-channel metrics "
                             "(hit_rate, first_request_success, ...) are omitted as meaningless.")
    args = parser.parse_args()

    _load_env()

    model_default = os.environ.get("MIRA_MATH_MODEL", "gpt-4o-mini")
    model_a = args.model_a or os.environ.get("MIRA_MATH_MODEL_TIER_A") or os.environ.get("MIRA_MATH_MODEL_A", model_default)
    model_b = args.model_b or os.environ.get("MIRA_MATH_MODEL_TIER_B") or os.environ.get("MIRA_MATH_MODEL_B", model_default)

    tools_str = ", ".join(args.tools) if args.tools else "None"
    workers = max(1, args.workers)

    dataset_name = os.path.splitext(os.path.basename(args.input_file))[0]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_a = model_a.replace("/", "_")
    safe_b = model_b.replace("/", "_")
    condition_tag = "oracle_" if args.oracle else ""
    run_id = args.run_id or f"{condition_tag}{safe_a}_vs_{safe_b}_{dataset_name}_{timestamp}"
    os.makedirs(args.log_dir, exist_ok=True)

    log = logging.getLogger("MIRA_MATH_dataset")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    log.handlers = []
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    log.addHandler(stream_handler)

    instances = load_jsonl(args.input_file)
    if args.start > 0:
        instances = instances[args.start:]
    if args.limit:
        instances = instances[:args.limit]

    incr_metrics_path = os.path.join(args.log_dir, f"MIRA_MATH_{run_id}_results.jsonl")
    incr_transcripts_path = os.path.join(args.log_dir, f"MIRA_MATH_{run_id}_transcripts.jsonl")

    print(f"Run: {run_id} | Dataset: {args.input_file}")
    print(f"Models: A={model_a} B={model_b}")
    print(f"Method: {args.method} | Tools: {tools_str}")
    if args.oracle:
        print("Condition: ORACLE-HINT -- Agent A receives its atomic hint up front; "
              "Agent B is not called. Request-channel metrics are omitted.")
    print(f"Running {len(instances)} instances")
    print(f"Incremental results: {incr_metrics_path}")
    print(f"Incremental transcripts: {incr_transcripts_path}")
    print_metric_legend()
    print_dataset_distribution(instances)
    print()

    interrupted = False
    all_metrics: List[Dict[str, Any]] = []
    all_transcripts: Dict[str, List[Dict[str, Any]]] = {}
    try:
        if workers > 1:
            all_metrics, all_transcripts, interrupted = asyncio.run(
                _run_parallel(
                    instances, model_a, model_b, args.method, args.tools,
                    args.temperature, args.seed, workers, args.verbose, log,
                    oracle=args.oracle,
                    instance_timeout=args.instance_timeout,
                    incr_metrics_path=incr_metrics_path,
                    incr_transcripts_path=incr_transcripts_path,
                )
            )
        else:
            all_metrics, all_transcripts, interrupted = _run_sequential(
                instances, model_a, model_b, args.method, args.tools,
                args.temperature, args.seed, args.verbose, log,
                oracle=args.oracle,
                incr_metrics_path=incr_metrics_path,
                incr_transcripts_path=incr_transcripts_path,
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\n!! Top-level KeyboardInterrupt. Partial progress preserved in incremental logs.")
        log.warning("Top-level KeyboardInterrupt")
    except BaseException as e:
        interrupted = True
        print(f"\n!! Fatal error: {type(e).__name__}: {e}")
        log.error("Fatal error: %s", e, exc_info=True)

    # Drop empty pre-allocated slots from parallel mode if a Ctrl+C cancelled tasks before they ran.
    all_metrics = [m for m in all_metrics if m]

    agg = aggregate_metrics(all_metrics) if all_metrics else {}
    if args.oracle:
        strip_aggregate_metrics(agg)

    results_path = os.path.join(args.log_dir, f"MIRA_MATH_{run_id}_results.jsonl")
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
    if args.oracle:
        for bundle in (type_aggregates, diff_aggregates, family_aggregates,
                       diff_x_type_aggregates, diff_x_family_aggregates):
            for sub_agg in bundle.values():
                strip_aggregate_metrics(sub_agg)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "dataset": args.input_file,
            "condition": "oracle_hint" if args.oracle else "protocol",
            "model_a": model_a,
            # Agent B is never invoked in the oracle condition.
            "model_b": None if args.oracle else model_b,
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

    transcripts_path = os.path.join(args.log_dir, f"MIRA_MATH_{run_id}_transcripts.jsonl")
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
    if agg.get("avg_rounds_to_solve") is not None:
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