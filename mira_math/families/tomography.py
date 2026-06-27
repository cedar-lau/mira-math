"""Discrete tomography on a 3x3 binary grid with one extra cell hint."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import random


FAMILY_NAME = "discrete_tomography"
FAMILY_TYPE = "B"  # Variable hint slot: any ambiguous grid cell


def _iter_matrices() -> List[List[List[int]]]:
    mats: List[List[List[int]]] = []
    for mask in range(1 << 9):
        mat = [[0 for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                bit = (mask >> (i * 3 + j)) & 1
                mat[i][j] = bit
        mats.append(mat)
    return mats


def _row_sums(mat: List[List[int]]) -> List[int]:
    return [sum(row) for row in mat]


def _col_sums(mat: List[List[int]]) -> List[int]:
    return [sum(mat[i][j] for i in range(3)) for j in range(3)]


def _extract_constraints(view: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, int], Dict[Tuple[int, int], int]]:
    row_sums: Dict[int, int] = {}
    col_sums: Dict[int, int] = {}
    cells: Dict[Tuple[int, int], int] = {}
    for c in view["private_data"]["constraints_machine"]:
        if c["type"] == "row_sum":
            row_sums[int(c["row"])] = int(c["value"])
        elif c["type"] == "col_sum":
            col_sums[int(c["col"])] = int(c["value"])
        elif c["type"] == "cell":
            cells[(int(c["row"]), int(c["col"]))] = int(c["value"])
    return row_sums, col_sums, cells


def _matches(mat: List[List[int]], row_sums: Dict[int, int], col_sums: Dict[int, int], cells: Dict[Tuple[int, int], int]) -> bool:
    for r, v in row_sums.items():
        if sum(mat[r]) != v:
            return False
    for c, v in col_sums.items():
        if sum(mat[i][c] for i in range(3)) != v:
            return False
    for (i, j), v in cells.items():
        if mat[i][j] != v:
            return False
    return True


def _solutions(row_sums: Dict[int, int], col_sums: Dict[int, int], cells: Dict[Tuple[int, int], int]) -> List[List[List[int]]]:
    mats = _iter_matrices()
    return [m for m in mats if _matches(m, row_sums, col_sums, cells)]


def generate_instance(rng: random.Random, difficulty: int, instance_id: str) -> Dict[str, Any]:
    mats = _iter_matrices()

    while True:
        mat = rng.choice(mats)
        row = _row_sums(mat)
        col = _col_sums(mat)

        row_sums = {i: row[i] for i in range(3)}
        col_sums = {j: col[j] for j in range(3)}

        base_solutions = _solutions(row_sums, col_sums, {})
        if len(base_solutions) <= 1:
            continue

        ambiguous_cells = [
            (i, j)
            for i in range(3)
            for j in range(3)
            if len({m[i][j] for m in base_solutions}) > 1
        ]
        if not ambiguous_cells:
            continue
        target_cell = rng.choice(ambiguous_cells)

        # Pick a hint cell (different from target) that uniquely resolves the grid.
        hint_cell = None
        hint_candidates = [c for c in ambiguous_cells if c != target_cell]
        rng.shuffle(hint_candidates)
        for candidate in hint_candidates:
            val = mat[candidate[0]][candidate[1]]
            if len(_solutions(row_sums, col_sums, {candidate: val})) == 1:
                hint_cell = candidate
                break
        if hint_cell is None:
            continue

        instance = {
            "id": instance_id,
            "family": FAMILY_NAME,
            "difficulty": difficulty,
            "n_agents": 2,
            "answer_type": "int",
            "global_solution": {"value": int(mat[target_cell[0]][target_cell[1]])},
            "global_metadata": {
                "size": [3, 3],
                "target_cell": list(target_cell),
                "row_sums": row,
                "col_sums": col,
            },
            "agent_views": [
                {
                    "agent_id": "A",
                    "private_data": {
                        "unknowns": [f"c{i}{j}" for i in range(3) for j in range(3)],
                        "externals": [],
                        "constraints_text": [
                            f"row {i} sum = {row[i]}" for i in range(3)
                        ] + [
                            f"col {j} sum = {col[j]}" for j in range(3)
                        ],
                        "constraints_machine": [
                            {"type": "row_sum", "row": i, "value": row[i]} for i in range(3)
                        ] + [
                            {"type": "col_sum", "col": j, "value": col[j]} for j in range(3)
                        ],
                    },
                },
                {
                    "agent_id": "B",
                    "private_data": {
                        "unknowns": [f"c{i}{j}" for i in range(3) for j in range(3)],
                        "externals": [],
                        "constraints_text": [
                            f"cell({hint_cell[0]},{hint_cell[1]}) = {mat[hint_cell[0]][hint_cell[1]]}"
                        ],
                        "constraints_machine": [
                            {
                                "type": "cell",
                                "row": hint_cell[0],
                                "col": hint_cell[1],
                                "value": mat[hint_cell[0]][hint_cell[1]],
                            }
                        ],
                    },
                },
            ],
            "minimal_hint_spec": {
                "k_min": 1,
                "atomic_hints": [
                    {
                        "hint_id": "h1",
                        "kind": "cell",
                        "row": hint_cell[0],
                        "col": hint_cell[1],
                        "value": mat[hint_cell[0]][hint_cell[1]],
                        "providers": ["B"],
                        "consumers": ["A"],
                    }
                ],
            },
        }
        return instance


def is_locally_illposed(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    row_sums, col_sums, cells = _extract_constraints(agent_view)
    solutions = _solutions(row_sums, col_sums, cells)
    return len(solutions) != 1


def apply_hint(agent_view: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    new_view = {
        "agent_id": agent_view["agent_id"],
        "private_data": {
            **agent_view["private_data"],
            "constraints_machine": list(agent_view["private_data"]["constraints_machine"]),
            "constraints_text": list(agent_view["private_data"]["constraints_text"]),
        },
    }
    if hint["kind"] == "cell":
        new_view["private_data"]["constraints_machine"].append(
            {"type": "cell", "row": hint["row"], "col": hint["col"], "value": hint["value"]}
        )
        new_view["private_data"]["constraints_text"].append(
            f"cell({hint['row']},{hint['col']}) = {hint['value']}"
        )
    return new_view


def is_uniquely_solvable(instance: Dict[str, Any], agent_view: Dict[str, Any]) -> bool:
    row_sums, col_sums, cells = _extract_constraints(agent_view)
    solutions = _solutions(row_sums, col_sums, cells)
    return len(solutions) == 1


def solve_global(instance: Dict[str, Any]) -> Dict[str, int]:
    row_sums: Dict[int, int] = {}
    col_sums: Dict[int, int] = {}
    cells: Dict[Tuple[int, int], int] = {}
    for v in instance["agent_views"]:
        r, c, cell = _extract_constraints(v)
        row_sums.update(r)
        col_sums.update(c)
        cells.update(cell)
    sols = _solutions(row_sums, col_sums, cells)
    if len(sols) != 1:
        raise ValueError("tomography not uniquely solvable")
    mat = sols[0]
    target_cell = tuple(instance["global_metadata"]["target_cell"])
    return {"value": int(mat[target_cell[0]][target_cell[1]])}
