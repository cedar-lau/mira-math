"""Shared display helpers for mira_math dataset runners."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from mira_math.families import FAMILY_TYPES
from mira_math.scoring import aggregate_metrics

_W = 62  # column width for dividers


def print_metric_legend() -> None:
    """Print a one-time legend explaining every logged metric field."""
    print("\n" + "=" * _W)
    print("METRIC LEGEND")
    print("=" * _W)
    rows = [
        ("family",                "problem family name"),
        ("difficulty",            "instance difficulty: 1=easy  2=medium  3=hard"),
        ("family_type",           "A = fixed hint slot  |  B = variable hint slot"),
        ("acc_final",             "1 = correct final answer, 0 = wrong or missing"),
        ("request_attempts",      "total requests A sent to B  (lower = better)"),
        ("hit_rate",              "fraction of requests B accepted (offers / total)"),
        ("hints_used",            "number of B's offers A received"),
        ("hint_overuse",          "hints beyond minimum needed: max(0, used - k_min)"),
        ("decline_count",         "times B declined A's request"),
        ("first_request_success", "1 if A received an offer on the very first request"),
        ("requests_before_offer", "requests made before B's first offer  (0 = first try)"),
        ("rounds_to_solve",       "round number when A submitted the correct answer"),
        ("rounds_to_final",       "round number when A submitted any final answer"),
        ("token_cost",            "estimated token count (word-count x 1.3)"),
    ]
    col = max(len(r[0]) for r in rows)
    for name, desc in rows:
        print(f"  {name:<{col}}  {desc}")
    print("=" * _W)


def print_dataset_distribution(instances: List[Dict[str, Any]]) -> None:
    """Print a breakdown of the dataset by family type and difficulty."""
    # Collect counts
    by_type_diff: Dict[str, Dict[int, int]] = {"A": defaultdict(int), "B": defaultdict(int), "?": defaultdict(int)}
    by_family: Dict[str, int] = defaultdict(int)

    for inst in instances:
        family = inst.get("family", "?")
        diff = inst.get("difficulty", 1)
        ftype = FAMILY_TYPES.get(family, "?")
        by_type_diff[ftype][diff] += 1
        by_family[family] += 1

    print("\n" + "=" * _W)
    print("DATASET DISTRIBUTION")
    print("=" * _W)

    all_diffs = sorted({d for td in by_type_diff.values() for d in td})

    for ftype, label in (("A", "Type A - fixed hint slot"), ("B", "Type B - variable hint slot")):
        counts = by_type_diff[ftype]
        if not counts:
            continue
        total = sum(counts.values())
        diff_str = "  ".join(f"diff {d}: {counts.get(d, 0)}" for d in all_diffs)
        print(f"  {label}: {total} instances  ({diff_str})")

    if by_type_diff["?"]:
        unknown_total = sum(by_type_diff["?"].values())
        print(f"  Unknown type: {unknown_total} instances")

    print()
    # Family counts, sorted by type then name
    type_a_families = sorted(f for f in by_family if FAMILY_TYPES.get(f) == "A")
    type_b_families = sorted(f for f in by_family if FAMILY_TYPES.get(f) == "B")
    other_families  = sorted(f for f in by_family if FAMILY_TYPES.get(f) not in ("A", "B"))

    for ftype_label, fams in (("A", type_a_families), ("B", type_b_families), ("?", other_families)):
        if not fams:
            continue
        print(f"  [Type {ftype_label}]")
        for fam in fams:
            print(f"    {fam:<45} {by_family[fam]:>4} instances")

    print("=" * _W)


def print_type_difficulty_summary(all_metrics: List[Dict[str, Any]]) -> None:
    """Print per-type and per-difficulty accuracy/attempts breakdown."""
    print()

    # --- Per type ---
    for ftype in ("A", "B"):
        tm = [m for m in all_metrics if m.get("family_type") == ftype]
        if not tm:
            continue
        ta = aggregate_metrics(tm)
        frs = ta.get("first_request_success_rate", 0)
        print(
            f"  Type {ftype}  ({ta['n_instances']} inst): "
            f"acc={ta['accuracy']:.0%}  "
            f"attempts={ta['avg_request_attempts']:.2f}  "
            f"hit={ta['avg_hit_rate']:.0%}  "
            f"1st-req-ok={frs:.0%}"
        )

    print()

    # --- Per difficulty ---
    all_diffs = sorted({m.get("difficulty", 1) for m in all_metrics if "difficulty" in m})
    for diff in all_diffs:
        dm = [m for m in all_metrics if m.get("difficulty") == diff]
        if not dm:
            continue
        da = aggregate_metrics(dm)
        print(
            f"  diff={diff}  ({da['n_instances']} inst): "
            f"acc={da['accuracy']:.0%}  "
            f"attempts={da['avg_request_attempts']:.2f}  "
            f"hit={da['avg_hit_rate']:.0%}"
        )

    print()

    # --- Per difficulty x type ---
    if all_diffs and any(m.get("family_type") for m in all_metrics):
        print("  Difficulty x Type:")
        for ftype in ("A", "B"):
            for diff in all_diffs:
                dtm = [m for m in all_metrics if m.get("family_type") == ftype and m.get("difficulty") == diff]
                if not dtm:
                    continue
                dta = aggregate_metrics(dtm)
                frs = dta.get("first_request_success_rate", 0)
                print(
                    f"    Type {ftype} diff={diff}  ({dta['n_instances']} inst): "
                    f"acc={dta['accuracy']:.0%}  "
                    f"att={dta['avg_request_attempts']:.2f}  "
                    f"1st={frs:.0%}"
                )
        print()

    # --- Per family (sorted by accuracy desc) ---
    all_families = sorted({m.get("family", "?") for m in all_metrics if "family" in m})
    family_rows = []
    for fam in all_families:
        fm = [m for m in all_metrics if m.get("family") == fam]
        if not fm:
            continue
        fa = aggregate_metrics(fm)
        ftype = FAMILY_TYPES.get(fam, "?")
        family_rows.append((fam, ftype, fa))

    family_rows.sort(key=lambda r: (-r[2]["accuracy"], r[0]))
    if family_rows:
        col = max(len(r[0]) for r in family_rows)
        for fam, ftype, fa in family_rows:
            frs = fa.get("first_request_success_rate", 0)
            print(
                f"  [{ftype}] {fam:<{col}}  "
                f"acc={fa['accuracy']:.0%}  "
                f"att={fa['avg_request_attempts']:.1f}  "
                f"1st={frs:.0%}  "
                f"({fa['n_instances']})"
            )

    # --- Per difficulty x family ---
    if all_diffs and family_rows:
        print()
        print("  Difficulty x Family:")
        col = max(len(r[0]) for r in family_rows)
        for fam, ftype, _ in family_rows:
            for diff in all_diffs:
                dfm = [m for m in all_metrics if m.get("family") == fam and m.get("difficulty") == diff]
                if not dfm:
                    continue
                dfa = aggregate_metrics(dfm)
                frs = dfa.get("first_request_success_rate", 0)
                print(
                    f"    [{ftype}] {fam:<{col}} d={diff}  "
                    f"acc={dfa['accuracy']:.0%}  "
                    f"att={dfa['avg_request_attempts']:.1f}  "
                    f"1st={frs:.0%}  "
                    f"({dfa['n_instances']})"
                )
