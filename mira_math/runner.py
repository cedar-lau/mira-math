"""Runner skeleton for mira_math."""
from __future__ import annotations

from typing import Any, Dict, List
import argparse
import importlib.util
import json
import os

from mira_math.scoring import score_transcript
from mira_math.schema import load_jsonl


def _load_backend(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    mod_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to import backend: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_dataset(instances: List[Dict[str, Any]], backend, mode: str) -> List[Dict[str, Any]]:
    results = []
    for inst in instances:
        transcript = backend.run_instance(inst, mode=mode)
        metrics = score_transcript(inst, transcript, mode=mode)
        results.append({"id": inst["id"], **metrics})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--agent_backend", required=True)
    parser.add_argument("--mode", choices=["coordinator", "peer"], default="coordinator")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    instances = load_jsonl(args.input_path)
    backend = _load_backend(args.agent_backend)

    results = run_dataset(instances, backend, mode=args.mode)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

