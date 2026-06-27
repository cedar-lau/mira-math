"""Schema utilities for mira_math instances.

The schema is intentionally lightweight and JSON-first. We keep validation
logic simple and rely on family-specific validators for deeper checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json


@dataclass
class AgentView:
    agent_id: str
    private_data: Dict[str, Any]


@dataclass
class MinimalHintSpec:
    k_min: int
    atomic_hints: List[Dict[str, Any]]


@dataclass
class Instance:
    id: str
    family: str
    difficulty: int
    n_agents: int
    answer_type: str
    global_solution: Any
    global_metadata: Dict[str, Any]
    agent_views: List[AgentView]
    minimal_hint_spec: MinimalHintSpec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "difficulty": self.difficulty,
            "n_agents": self.n_agents,
            "answer_type": self.answer_type,
            "global_solution": self.global_solution,
            "global_metadata": self.global_metadata,
            "agent_views": [
                {"agent_id": v.agent_id, "private_data": v.private_data}
                for v in self.agent_views
            ],
            "minimal_hint_spec": {
                "k_min": self.minimal_hint_spec.k_min,
                "atomic_hints": self.minimal_hint_spec.atomic_hints,
            },
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Instance":
        views = [AgentView(**v) for v in d["agent_views"]]
        hint_spec = MinimalHintSpec(**d["minimal_hint_spec"])
        return Instance(
            id=d["id"],
            family=d["family"],
            difficulty=d.get("difficulty", 1),
            n_agents=d.get("n_agents", len(views)),
            answer_type=d.get("answer_type", "dict_int"),
            global_solution=d["global_solution"],
            global_metadata=d.get("global_metadata", {}),
            agent_views=views,
            minimal_hint_spec=hint_spec,
        )


REQUIRED_TOP_LEVEL_FIELDS = {
    "id",
    "family",
    "difficulty",
    "n_agents",
    "answer_type",
    "global_solution",
    "global_metadata",
    "agent_views",
    "minimal_hint_spec",
}


def basic_schema_checks(d: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    missing = REQUIRED_TOP_LEVEL_FIELDS - set(d.keys())
    if missing:
        errors.append(f"missing top-level fields: {sorted(missing)}")

    if "agent_views" in d:
        if not isinstance(d["agent_views"], list) or not d["agent_views"]:
            errors.append("agent_views must be a non-empty list")
        else:
            for i, v in enumerate(d["agent_views"]):
                if "agent_id" not in v or "private_data" not in v:
                    errors.append(f"agent_views[{i}] missing agent_id or private_data")

    if "minimal_hint_spec" in d:
        m = d["minimal_hint_spec"]
        if "k_min" not in m or "atomic_hints" not in m:
            errors.append("minimal_hint_spec missing k_min or atomic_hints")

    return errors


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    instances = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instances.append(json.loads(line))
    return instances


def dump_jsonl(path: str, instances: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(inst, ensure_ascii=True) + "\n")

