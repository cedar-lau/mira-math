"""Discrete Laplace equation on a small grid with missing boundary value."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import random

from mira_math.utils.exact import solve_linear_system


FAMILY_NAME = "laplace_grid"
FAMILY_TYPE = "B"  # Variable hint slot: any boundary cell u[i][j]


def _var_name(i: int, j: int) -> str:
    return f"u_{i}_{j}"


def _variables_for_grid(n: int) -> List[str]:
    return [_var_name(i, j) for i in range(n) for j in range(n)]


def _boundary_cells(n: int) -> List[Tuple[int, int]]:
    cells = []
    for i in range(n):
        for j in range(n):
            if i == 0 or j == 0 or i == n - 1 or j == n - 1:
                cells.append((i, j))
    return cells


def _interior_cells(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(1, n - 1) for j in range(1, n - 1)]


def _laplace_equations(n: int) -> List[Dict[str, Any]]:
    equations = []
    for i, j in _interior_cells(n):
        coeffs = {_var_name(i, j): 4}
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            coeffs[_var_name(ni, nj)] = coeffs.get(_var_name(ni, nj), 0) - 1
        equations.append({"type": "lin_eq", "coeffs": coeffs, "rhs": 0})
    return equations


def _format_laplace_eq(i: int, j: int) -> str:
    center = _var_name(i, j)
    neighbors = [
        _var_name(i - 1, j),
        _var_name(i + 1, j),
        _var_name(i, j - 1),
        _var_name(i, j + 1),
    ]
    return f"4 {center} = {neighbors[0]} + {neighbors[1]} + {neighbors[2]} + {neighbors[3]}"


def _build_linear_system(constraints: List[Dict[str, Any]], variables: List[str]) -> Tuple[List[List[int]], List[int]]:
    A: List[List[int]] = []
    b: List[int] = []
    for c in constraints:
        if c["type"] == "lin_eq":
            row = [int(c["coeffs"].get(v, 0)) for v in variables]
            A.append(row)
            b.append(int(c["rhs"]))
        elif c["type"] == "value":
            row = [0 for _ in variables]
            row[variables.index(c["var"])] = 1
            A.append(row)
            b.append(int(c["value"]))
    return A, b


def _solve_grid(n: int, constraints: List[Dict[str, Any]]) -> Dict[str, int]:
    variables = _variables_for_grid(n)
    A, b = _build_linear_system(constraints, variables)
    result = solve_linear_system(A, b)
    if result["status"] != "unique":
        raise ValueError("grid system not uniquely solvable")
    sol = result["solution"]
    out: Dict[str, int] = {}
    for i, v in enumerate(variables):
        if sol[i].denominator != 1:
            raise ValueError("non-integer grid solution")
        out[v] = int(sol[i])
    return out


def _scale_for_grid(n: int) -> int:
    return {4: 24, 5: 224}[n]


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    n = 4 if difficulty <= 1 else 5
    variables = _variables_for_grid(n)

    scale = _scale_for_grid(n)
    boundary_cells = _boundary_cells(n)

    while True:
        boundary_values = {}
        for i, j in boundary_cells:
            boundary_values[(i, j)] = scale * rng.randint(-2 - difficulty, 2 + difficulty)

        missing_cell = rng.choice(boundary_cells)
        target_cell = rng.choice(_interior_cells(n))

        base_constraints = _laplace_equations(n)
        full_constraints = list(base_constraints)
        for (i, j), val in boundary_values.items():
            full_constraints.append({"type": "value", "var": _var_name(i, j), "value": val})

        try:
            full_solution = _solve_grid(n, full_constraints)
        except Exception:
            continue

        target_var = _var_name(*target_cell)
        target_value = full_solution[target_var]

        # Build agent constraints
        laplace_text = [_format_laplace_eq(i, j) for i, j in _interior_cells(n)]
        base_text = laplace_text

        a_constraints = list(base_constraints)
        a_text = list(base_text)
        for (i, j), val in boundary_values.items():
            if (i, j) == missing_cell:
                continue
            a_constraints.append({"type": "value", "var": _var_name(i, j), "value": val})
            a_text.append(f"{_var_name(i, j)} = {val}")

        b_constraints = list(base_constraints)
        b_text = list(base_text)
        miss_val = boundary_values[missing_cell]
        b_constraints.append({"type": "value", "var": _var_name(*missing_cell), "value": miss_val})
        b_text.append(f"{_var_name(*missing_cell)} = {miss_val}")

        # Check A is ill-posed without hint
        A_mat, b_vec = _build_linear_system(a_constraints, variables)
        if solve_linear_system(A_mat, b_vec)["status"] == "unique":
            continue

        instance = {
            "id": instance_id,
            "family": FAMILY_NAME,
            "difficulty": difficulty,
            "n_agents": 2,
            "answer_type": "int",
            "global_solution": {"value": int(target_value)},
            "global_metadata": {
                "grid_size": n,
                "target_cell": list(target_cell),
                "missing_cell": list(missing_cell),
                "boundary_values": {f"{i},{j}": v for (i, j), v in boundary_values.items()},
            },
            "agent_views": [
                {
                    "agent_id": "A",
                    "private_data": {
                        "unknowns": variables,
                        "externals": [],
                        "constraints_text": a_text,
                        "constraints_machine": a_constraints,
                    },
                },
                {
                    "agent_id": "B",
                    "private_data": {
                        "unknowns": variables,
                        "externals": [],
                        "constraints_text": b_text,
                        "constraints_machine": b_constraints,
                    },
                },
            ],
            "minimal_hint_spec": {
                "k_min": 1,
                "atomic_hints": [
                    {
                        "hint_id": "h1",
                        "kind": "value",
                        "var": _var_name(*missing_cell),
                        "value": miss_val,
                        "providers": ["B"],
                        "consumers": ["A"],
                    }
                ],
            },
        }
        return instance



def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    n = int(instance["global_metadata"]["grid_size"])
    variables = _variables_for_grid(n)
    A, b = _build_linear_system(agent_view["private_data"]["constraints_machine"], variables)
    result = solve_linear_system(A, b)
    return result["status"] != "unique"


def apply_hint(agent_view: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    new_view = {
        "agent_id": agent_view["agent_id"],
        "private_data": {
            **agent_view["private_data"],
            "constraints_machine": list(agent_view["private_data"]["constraints_machine"]),
            "constraints_text": list(agent_view["private_data"]["constraints_text"]),
        },
    }
    if hint["kind"] == "value":
        new_view["private_data"]["constraints_machine"].append(
            {"type": "value", "var": hint["var"], "value": hint["value"]}
        )
        new_view["private_data"]["constraints_text"].append(f"{hint['var']} = {hint['value']}")
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    n = int(instance["global_metadata"]["grid_size"])
    variables = _variables_for_grid(n)
    A, b = _build_linear_system(agent_view["private_data"]["constraints_machine"], variables)
    result = solve_linear_system(A, b)
    return result["status"] == "unique"


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    n = int(instance["global_metadata"]["grid_size"])
    target_cell = tuple(instance["global_metadata"]["target_cell"])
    all_constraints = []
    for v in instance["agent_views"]:
        all_constraints.extend(v["private_data"]["constraints_machine"])
    sol = _solve_grid(n, all_constraints)
    return {"value": int(sol[_var_name(*target_cell)])}
