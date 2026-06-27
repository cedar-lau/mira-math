#!/usr/bin/env python3
"""mira_math Dataset Runner -- Resilient Parallel (LangGraph).

Like ``run_dataset_langgraph_parallel.py`` but with two resilience features:

1. **Inner retries**: transient API errors are retried within a wave.
2. **Shrinking dataset**: after each wave, successful instances are removed
   from the input JSONL (overwritten in-place) so you can Ctrl+C at any point
   and re-run with the same command to continue where you left off.

Results are appended incrementally to a JSONL log so nothing is lost.

Usage:
    # Basic resilient run
    python run_dataset_langgraph_resilient.py --in datasets/remaining.jsonl --method llm --workers 8

    # Cap retry waves
    python run_dataset_langgraph_resilient.py --in datasets/remaining.jsonl --workers 8 --max-waves 10

    # Wait longer between waves when the API is overloaded
    python run_dataset_langgraph_resilient.py --in datasets/remaining.jsonl --workers 4 --wave-pause 60
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from mira_math.scoring import score_transcript, aggregate_metrics
from mira_math.families import FAMILY_TYPES
from run_helpers import print_metric_legend, print_dataset_distribution, print_type_difficulty_summary

import run_single_langgraph as rsl

# ---------------------------------------------------------------------------
# Retry config (per-attempt, within a single wave)
# ---------------------------------------------------------------------------
_ATTEMPT_RETRY_MAX = 3
_ATTEMPT_BACKOFF_BASE = 5.0  # seconds; multiplied each retry


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
    instances: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    return instances


def save_jsonl(path: str, instances: List[Dict[str, Any]]) -> None:
    """Overwrite a JSONL file with the given instances."""
    with open(path, "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")


def append_results_jsonl(path: str, entries: List[Dict[str, Any]]) -> None:
    """Append result entries to a JSONL log file (one JSON object per line)."""
    with open(path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_view_summary(instance: Dict[str, Any], agent_id: str) -> str:
    """Get a brief summary of what an agent sees."""
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
            print(f"  A asks: \"{parsed.get('request', '?')}\"")
        elif t == "offer":
            quoted = parsed.get("constraint_quoted", "")
            hint = parsed.get("hint", "?")
            has_match = parsed.get("has_exact_match", False)
            if quoted:
                print(f"  B quotes: \"{quoted}\"")
            print(f"  B gives: \"{hint}\" (exact_match={has_match})")
        elif t == "decline":
            print(f"  B declines: \"{parsed.get('message', 'declined')}\"")
        elif t == "final":
            reasoning = parsed.get("reasoning", "")
            ans = parsed.get("answer", "?")
            if reasoning:
                print(f"  A reasons: {reasoning.replace(chr(10), ' ').replace('  ', ' ')}")
            print(f"  A answer: {ans}")


# ---------------------------------------------------------------------------
# Single-instance runner (one wave attempt with inner retries)
# ---------------------------------------------------------------------------

async def _run_one(
    idx: int,
    instance: Dict[str, Any],
    model_a: str,
    model_b: str,
    method: str,
    tool_names: Optional[List[str]],
    temperature: float,
    seed: Optional[int],
    instance_timeout: int,
    log: logging.Logger,
) -> Tuple[int, Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]], Optional[str]]:
    """Run a single instance with inner retries.

    Returns:
        (idx, metrics_or_None, transcript_or_None, error_str_or_None)
    """
    instance_id = instance.get("id", f"instance-{idx}")
    family = instance.get("family", "?")
    family_type = FAMILY_TYPES.get(family, "?")

    for attempt in range(1, _ATTEMPT_RETRY_MAX + 1):
        try:
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
                    False,
                ),
                timeout=instance_timeout,
            )
            metrics = score_transcript(instance, transcript)
            metrics["family_type"] = family_type
            return idx, metrics, transcript, None

        except asyncio.TimeoutError:
            log.error("Instance %s timed out after %ds", instance_id, instance_timeout)
            return idx, None, None, f"timeout after {instance_timeout}s"

        except Exception as e:
            if attempt < _ATTEMPT_RETRY_MAX and _is_retryable_api_error(e):
                wait = _ATTEMPT_BACKOFF_BASE ** attempt + random.uniform(0, 2)
                log.warning(
                    "Retryable error on %s (attempt %d/%d), retrying in %.1fs: %s",
                    instance_id, attempt, _ATTEMPT_RETRY_MAX, wait, e,
                )
                await asyncio.sleep(wait)
            else:
                log.error("Error on %s: %s", instance_id, e, exc_info=True)
                return idx, None, None, str(e)

    # All inner retries exhausted -- surface last error for wave-level retry
    return idx, None, None, f"exhausted {_ATTEMPT_RETRY_MAX} inner retries"


# ---------------------------------------------------------------------------
# Wave runner -- retries failed instances across waves, shrinks dataset
# ---------------------------------------------------------------------------

async def run_waves(
    instances: List[Dict[str, Any]],
    input_file: str,
    results_log_path: str,
    transcripts_log_path: str,
    model_a: str,
    model_b: str,
    method: str,
    tool_names: Optional[List[str]],
    temperature: float,
    seed: Optional[int],
    workers: int,
    instance_timeout: int,
    max_waves: int,
    wave_pause: int,
    verbose: bool,
    log: logging.Logger,
) -> Tuple[int, int]:
    """Run all instances in waves, shrinking the dataset file after each wave.

    Returns:
        (total_completed, total_correct) across all waves.
    """
    total_start = len(instances)
    # pending is a list of instances (actual dicts, not indices)
    pending: List[Dict[str, Any]] = list(instances)

    cumulative_completed = 0
    cumulative_correct = 0

    for wave in range(1, max_waves + 1):
        n_pending = len(pending)
        print(f"\n{'=' * 62}")
        print(f"WAVE {wave}/{max_waves} | {n_pending} pending | "
              f"{cumulative_completed}/{total_start} done | "
              f"{cumulative_correct}/{cumulative_completed} correct")
        print("=" * 62)

        if not pending:
            break

        sem = asyncio.Semaphore(workers)

        async def _bounded_run(idx: int, inst: Dict[str, Any]) -> Tuple[int, Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]], Optional[str]]:
            async with sem:
                return await _run_one(
                    idx, inst, model_a, model_b, method, tool_names,
                    temperature, seed, instance_timeout, log,
                )

        tasks = [asyncio.create_task(_bounded_run(i, inst)) for i, inst in enumerate(pending)]
        print_lock = asyncio.Lock()

        wave_succeeded_indices: List[int] = []  # indices into `pending`
        wave_interrupted = False

        try:
            for coro in asyncio.as_completed(tasks):
                idx, metrics, transcript, error = await coro

                instance = pending[idx]
                instance_id = instance.get("id", f"instance-{idx}")
                family = instance.get("family", "?")
                family_type = FAMILY_TYPES.get(family, "?")

                if error is not None:
                    async with print_lock:
                        print(
                            f"  [{cumulative_completed}/{total_start}] {instance_id} "
                            f"({family}, type={family_type})"
                            f"  FAILED (will retry): {error[:120]}"
                        )
                else:
                    cumulative_completed += 1
                    is_correct = metrics.get("acc_final", 0) == 1
                    if is_correct:
                        cumulative_correct += 1

                    wave_succeeded_indices.append(idx)
                    # Append immediately so Ctrl+C mid-wave still preserves results.
                    try:
                        append_results_jsonl(results_log_path, [{
                            "instance_id": instance_id,
                            "family": family,
                            "family_type": family_type,
                            "difficulty": instance.get("difficulty"),
                            "metrics": metrics,
                        }])
                        if transcript:
                            append_results_jsonl(
                                transcripts_log_path,
                                [{"instance_id": instance_id, "transcript": transcript}],
                            )
                    except Exception as exc:
                        log.error("Failed to append incremental log for %s: %s", instance_id, exc)

                    async with print_lock:
                        solve_lbl = "CORRECT" if is_correct else "WRONG"
                        attempts = metrics.get("request_attempts", 0)
                        print(
                            f"  [{cumulative_completed}/{total_start}] {instance_id} "
                            f"({family}, type={family_type})"
                            f"  {solve_lbl} (attempts: {attempts})"
                            f"  | running: {cumulative_correct}/{cumulative_completed}"
                        )
                        if verbose and transcript:
                            _display_transcript(transcript)
        except (KeyboardInterrupt, asyncio.CancelledError):
            wave_interrupted = True
            print(f"\n  !! Wave {wave} interrupted -- cancelling outstanding tasks and saving progress...")
            for t in tasks:
                if not t.done():
                    t.cancel()
            # Drain cancelled tasks; capture any that finished cleanly before cancel propagated.
            try:
                for t in tasks:
                    if t.done() and not t.cancelled():
                        try:
                            res = t.result()
                            r_idx, r_metrics, r_transcript, r_error = res
                            if r_error is None and r_idx not in wave_succeeded_indices:
                                instance = pending[r_idx]
                                instance_id = instance.get("id", f"instance-{r_idx}")
                                family = instance.get("family", "?")
                                family_type = FAMILY_TYPES.get(family, "?")
                                cumulative_completed += 1
                                if r_metrics.get("acc_final", 0) == 1:
                                    cumulative_correct += 1
                                wave_succeeded_indices.append(r_idx)
                                append_results_jsonl(results_log_path, [{
                                    "instance_id": instance_id,
                                    "family": family,
                                    "family_type": family_type,
                                    "difficulty": instance.get("difficulty"),
                                    "metrics": r_metrics,
                                }])
                                if r_transcript:
                                    append_results_jsonl(
                                        transcripts_log_path,
                                        [{"instance_id": instance_id, "transcript": r_transcript}],
                                    )
                        except Exception:
                            pass
            except Exception:
                pass

        # -- Wave done (or interrupted): persist progress and shrink dataset --
        wave_ok = len(wave_succeeded_indices)
        wave_failed = n_pending - wave_ok
        if wave_interrupted:
            print(f"\n  Wave {wave} interrupted: {wave_ok}/{n_pending} succeeded before interrupt")
        else:
            print(f"\n  Wave {wave} done: {wave_ok}/{n_pending} succeeded, {wave_failed} failed")

        # Remove succeeded instances from pending and overwrite the dataset
        succeeded_set = set(wave_succeeded_indices)
        pending = [inst for i, inst in enumerate(pending) if i not in succeeded_set]

        try:
            save_jsonl(input_file, pending)
            print(f"  Dataset overwritten: {len(pending)} instances remaining in {input_file}")
        except Exception as exc:
            log.error("Failed to overwrite dataset %s: %s", input_file, exc)
            print(f"  !! Failed to overwrite dataset: {exc}")

        if wave_interrupted:
            # Re-raise so main() can show final summary and exit cleanly.
            raise KeyboardInterrupt

        if not pending:
            print("  All instances completed!")
            break

        if wave < max_waves:
            pause = min(wave_pause * wave, 300)
            print(f"  Pausing {pause}s before wave {wave + 1} ({len(pending)} instances to retry)...")
            await asyncio.sleep(pause)
        else:
            print(f"  Reached max waves ({max_waves}). {len(pending)} instances still failed.")

    return cumulative_completed, cumulative_correct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="mira_math Dataset Runner -- Resilient Parallel (shrinks dataset on success)"
    )
    parser.add_argument("--in", dest="input_file", required=True,
                        help="Path to input JSONL dataset (overwritten in-place as instances succeed)")
    parser.add_argument("--model-a", default=None, help="Model for Agent A")
    parser.add_argument("--model-b", default=None, help="Model for Agent B")
    parser.add_argument("--method", choices=["llm", "react", "reflexion"], default="llm",
                        help="Agent method: llm (single-shot), react (reasoning+acting), reflexion (self-reflection)")
    parser.add_argument("--tools", nargs="*", default=None,
                        help="Tools to give agents (e.g., calculator)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", "-j", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed traces of agent reasoning")
    parser.add_argument("--instance-timeout", type=int, default=1200,
                        help="Per-instance wall-clock timeout in seconds (default: 1200)")
    parser.add_argument("--max-waves", type=int, default=20,
                        help="Maximum retry waves before giving up on remaining failures (default: 20)")
    parser.add_argument("--wave-pause", type=int, default=30,
                        help="Base pause in seconds between waves, scales with wave number (default: 30)")
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
    run_id = args.run_id or f"{safe_a}_vs_{safe_b}_{dataset_name}_{timestamp}"
    os.makedirs(args.log_dir, exist_ok=True)

    # Incremental result/transcript logs (JSONL, appended to)
    results_log_path = os.path.join(args.log_dir, f"MIRA_MATH_{run_id}_results.jsonl")
    transcripts_log_path = os.path.join(args.log_dir, f"MIRA_MATH_{run_id}_transcripts.jsonl")

    log = logging.getLogger("MIRA_MATH_resilient")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    log.handlers = []
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    log.addHandler(stream_handler)

    instances = load_jsonl(args.input_file)

    print(f"Run: {run_id} | Dataset: {args.input_file}")
    print(f"Models: A={model_a} B={model_b}")
    print(f"Method: {args.method} | Tools: {tools_str}")
    print(f"Instances: {len(instances)} | Workers: {workers} | Max waves: {args.max_waves}")
    print(f"Results log: {results_log_path}")
    print(f"NOTE: {args.input_file} will be overwritten after each wave (shrinks as instances succeed)")
    print_metric_legend()
    print_dataset_distribution(instances)
    print()

    total_completed = 0
    total_correct = 0
    interrupted = False
    fatal_error: Optional[BaseException] = None
    try:
        total_completed, total_correct = asyncio.run(
            run_waves(
                instances, args.input_file, results_log_path, transcripts_log_path,
                model_a, model_b, args.method, args.tools,
                args.temperature, args.seed, workers, args.instance_timeout,
                args.max_waves, args.wave_pause, args.verbose, log,
            )
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n!! Interrupted by user (Ctrl+C). Partial progress is preserved on disk.")
        log.warning("Run interrupted by KeyboardInterrupt")
    except BaseException as e:
        fatal_error = e
        print(f"\n!! Fatal error: {type(e).__name__}: {e}")
        print("   Partial progress (completed instances) preserved in incremental logs.")
        log.error("Fatal error during run_waves: %s", e, exc_info=True)

    # ----- Final summary (always runs, even after interrupt / crash) -----
    try:
        remaining = load_jsonl(args.input_file)
    except Exception as exc:
        log.error("Could not re-read dataset for final summary: %s", exc)
        remaining = []

    # Recompute totals from the incremental log when interrupted (run_waves's
    # in-memory counters may be stale if it crashed mid-wave).
    if interrupted or fatal_error is not None:
        try:
            with open(results_log_path, "r", encoding="utf-8") as f:
                log_entries = [json.loads(ln) for ln in f if ln.strip()]
            total_completed = len(log_entries)
            total_correct = sum(
                1 for e in log_entries if e.get("metrics", {}).get("acc_final", 0) == 1
            )
        except FileNotFoundError:
            pass
        except Exception as exc:
            log.error("Could not parse incremental results log: %s", exc)

    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print(f"Run ID: {run_id}")
    print(f"Method: {args.method} | Tools: {tools_str}")
    print(f"Models: A={model_a} B={model_b}")
    print("-" * 62)
    if interrupted:
        print("STATUS: INTERRUPTED (Ctrl+C)")
    elif fatal_error is not None:
        print(f"STATUS: FAILED ({type(fatal_error).__name__})")
    print(f"Completed: {total_completed}")
    print(f"Correct: {total_correct}")
    if total_completed > 0:
        print(f"Accuracy (completed only): {total_correct / total_completed:.2%}")
    print(f"Remaining in dataset: {len(remaining)}")
    print("-" * 62)
    print(f"Results log: {results_log_path}")
    print(f"Transcripts log: {transcripts_log_path}")
    if remaining:
        print(f"\nTo continue: re-run with the same --in {args.input_file}")


if __name__ == "__main__":
    main()
