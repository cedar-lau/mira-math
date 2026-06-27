"""Prompt templates for mira_math v2.

Key features:
- Agents know all problem types in the benchmark
- Agent A makes natural language requests
- Agent B only provides info if request semantically matches what B has

FIXES:
- Added _format_target_description(): reads global_metadata to tell Agent A
  WHICH specific variable/cell/value to report as the final answer.
  Without this, A solved the system correctly but didn't know which of the
  solved variables was the target (e.g. returned u_3_1=24 instead of u_2_1=-3).
- Removed dead available_hint reference from user_prompt_agent_a history block.
- Removed "hint at what TYPE of info you have" from Agent B prompts.
- Added NOTATION_GUIDE for semantic matching across all 13 families.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Problem family descriptions
# ---------------------------------------------------------------------------
PROBLEM_FAMILIES = """
## Problem Families in This Benchmark

1. **Linear System (linear_system_separator)**
   - Solve a system of linear equations in variables (e.g., a, b, c)
   - Agent A has 2 equations in 3 unknowns; Agent B has the value of exactly one variable (the separator)
   - You do NOT know which variable B holds — try each by name until B responds with an offer
   - Request format: "the value of variable X" (try a, b, c one at a time)
   - Example: "the value of variable c" matches constraint "c = 6"

2. **CRT Reconstruction (crt_reconstruction)**
   - Find x given modular congruences (x ≡ r mod m)
   - Agent A has some congruences, Agent B has exactly one remaining congruence
   - You do NOT know B's modulus — request generically without specifying it
   - Request format: "your remaining congruence" or "the missing congruence"
   - Do NOT say "x mod 13" unless you already know the modulus from your own constraints
   - Example: "your remaining congruence" matches B's single congruence regardless of modulus

3. **Polynomial Interpolation (poly_interpolation)**
   - Determine p(target_x) given interpolation points
   - Agent A has degree points; Agent B has exactly one more point needed to uniquely define the polynomial
   - You do NOT know B's x-value — request generically without specifying it
   - Request format: "your interpolation point" or "the missing interpolation point"
   - Do NOT say "the point p(5)" unless you already know x=5 from your own constraints
   - Example: "your interpolation point" matches B's single point regardless of its x-value

4. **Recurrence Sequence (recurrence_missing_init)**
   - Compute a(n) for a linear recurrence with missing initial condition
   - One agent has a(0), the other has a(1), both know the recurrence
   - Request format: "the initial value a(0)" or "the initial value a(1)"
   - Example: "the initial value a(1)" matches constraint "a(1) = 7"

5. **Rank-Deficient Linear (rankdef_linear_shared)**
   - Similar to linear systems but both agents have partial equation sets
   - Need to share equations to get full rank
   - Request format: "an equation involving variable X"
   - Example: "an equation involving variable a" matches any constraint that contains
     the variable a, such as "2a + 3b = 7" or "a - c = 4"

6. **Laplace Grid (laplace_grid)**
   - Solve discrete Laplace equation on a grid
   - One agent missing a boundary value
   - Variable notation: u_i_j means the grid value at row i, column j
   - Request format: PREFER "the value of u_i_j" (e.g., "the value of u_3_1")
     You may also say "the boundary value at position (i,j)" — both mean the same
   - Example: "the value of u_3_1" and "the boundary value at position (3,1)"
     both match constraint "u_3_1 = 24"

7. **Deconvolution (deconvolution)**
   - Recover signal x from convolution measurements y = x * h
   - One agent missing one measurement
   - Request format: "the measurement y[i]" or "the convolution value at index i"
   - Example: "the measurement y[3]" and "the convolution value at index 3"
     both match constraint "y[3] = 12"

8. **Phase Retrieval (phase_retrieval)**
   - Recover 4-element sequence from FFT magnitudes
   - Magnitudes have sign ambiguity; need sign information
   - Request format: "the sign of x0 (+1 or -1)" or "sign(x0)"
   - Note: FFT(x) refers to frequency-domain coefficients X_k, not x_n.
   - Note: |x_n|^2 refers to |x_n|^2, not |X_k|^2.
   - Note: You must use the inverse DFT relationship between X and x.
   - IMPORTANT: "the sign of x0" ≠ "the value of x0" — these are different quantities.
     If you need the sign, request the sign explicitly.

9. **Matrix Completion (matrix_completion)**
   - Complete a rank-1 matrix from partial entries
   - Variable notation: m[i][j] means the matrix entry at row i, column j
   - Request format: PREFER "the matrix entry m[i][j]" (e.g., "the matrix entry m[2][3]")
     You may also say "the value at position (i,j)" — both mean the same
   - Example: "the matrix entry m[2][3]" and "the value at position (2,3)"
     both match constraint "m[2][3] = 5"

10. **Discrete Tomography (discrete_tomography)**
    - Reconstruct binary grid from row/column sums
    - Need one cell value to disambiguate
    - Variable notation: x_i_j or cell(i,j) means the cell value at row i, column j
    - Request format: "the cell value at (i,j)"
    - Example: "the cell value at (1,2)" matches constraint "x_1_2 = 1" or "cell(1,2) = 1"

11. **Graph Path Sums (graph_path_sums)**
    - Recover edge weights from path sums on a triangle (edges e01, e12, e20)
    - Agent A has 2 path sums; Agent B has both of those PLUS one additional path sum A lacks
    - Identify which edge pair is missing from your sums, then request it specifically by edge pair
    - Request format: "the path sum for edges e_XX and e_YY" (name the specific edge pair you are missing)
    - Example: if you have e01+e12 and e12+e20, request "the path sum for e01 and e20"
    - IMPORTANT: B also holds a path sum you already have — be specific so B gives the one you lack

12. **Moment Problem (moment_problem)**
    - Recover discrete distribution from moments
    - Variable notation: moment_k means the k-th statistical moment
    - Request format: "the k-th moment" or "the value of moment_k"
    - Example: "the 2nd moment" and "the value of moment_2" both match "moment_2 = 15"
    - Note: ordinals map directly to indices: 1st→1, 2nd→2, 3rd→3, etc.

13. **Geometry Coordinates (geometry_coordinates)**
    - Find intersection point and compute distance
    - Request format: "the line equation" or "the coordinates of point Q"
    - Example: "the line equation" matches any constraint of the form "y = mx + b"
    - Example: "the coordinates of point Q" matches "Q = (4, 5)"

14. **Bayesian Inference (bayes_missing_prior)**
    - Compute P(H|E) via Bayes' theorem given likelihoods and a prior
    - Agent A holds P(E|H) (sensitivity) and P(E|¬H) (false-positive rate)
    - Agent B holds P(H) — the prior / base rate
    - Request format: "the prior P(H)" or "the base rate" or "the prior probability"
    - Example: "the prior P(H)" and "the base rate" both match constraint "P(H) = 1/3"
    - Answer: P(H|E) as a REDUCED fraction {"numerator": n, "denominator": d}

15. **Piecewise Function (piecewise_missing_threshold)**
    - Evaluate f(x) for a 2-branch piecewise linear function with unknown threshold
    - Agent A holds x (query point) and BOTH branch formulas, but NOT the threshold t
    - Agent B holds x and the threshold t (determines which branch is active)
    - Request format: "the threshold t" or "the value of t" or "the branch threshold"
    - Example: "the threshold t" and "the branch threshold" both match "threshold: t = -3"
    - Answer: the integer value f(x)

16. **Markov Chain Transition (markov_missing_transition)**
    - Compute the steady-state probability pi(k) of an n-state Markov chain
    - Agent A holds most transition probabilities as NATURAL LANGUAGE descriptions
      (e.g., "From state 0, there is a 1/3 chance of transitioning to state 1.")
    - One specific entry P(i -> j) is missing from your view; Agent B holds it
    - You must figure out which entry is missing by checking which row is incomplete
    - Request format: "the transition probability P(i -> j)" with the exact state indices
    - Example: "the transition probability P(2 -> 1)" or just "P(2 -> 1)"
    - Agent B will respond with that value in P(i -> j) = x/y notation
    - Answer: pi(k) as a FULLY REDUCED fraction {"numerator": n, "denominator": d}

17. **Linear System Missing Coefficient (linear_system_missing_coeff)**
    - Solve a linear system Ax = b; one matrix coefficient A[i][j] is absent
    - Agent A holds all coefficients and RHS values EXCEPT A[missing_row][missing_col]
    - Each coefficient is given as a natural-language sentence:
      e.g. "In equation 2, the coefficient of x1 is 3."
    - You must scan all (row, col) pairs to find which coefficient sentence is missing
    - Request format: "the coefficient of x{j} in equation {i}" or "A[{i}][{j}]"
    - Example: "the coefficient of x1 in equation 2" matches "A[2][1] = 3"
    - After hint: solve the full n x n system; report x[target_k] as a fraction
    - Answer: x[target_k] as {"numerator": n, "denominator": d}

18. **Circuit Missing Resistance (circuit_missing_resistance)**
    - Compute the equivalent resistance R_eq of a series/parallel resistor network
    - Agent A holds the circuit topology (which resistors are in series/parallel) and
      all individual resistance values EXCEPT one resistor R_k
    - You must check all named resistors R1, R2, ... to find which value is missing
    - Request format: "the resistance of R{k}" or "R{k}"
    - Example: "the resistance of R3" matches "R3 = 5 ohms"
    - After hint: evaluate the series/parallel tree to compute R_eq
    - Answer: R_eq as {"numerator": n, "denominator": d}

19. **Portfolio Variance Missing Correlation (portfolio_variance_missing_corr)**
    - Compute portfolio variance sigma_p^2 = sum_i sum_j w_i*w_j*sigma_i*sigma_j*rho(i,j)
    - Agent A holds weights w_i, standard deviations sigma_i, and all pairwise correlations
      rho(i,j) EXCEPT one pair (a,b)
    - You must scan all expected rho(i,j) pairs to find which is missing
    - Note: rho(i,j) = rho(j,i), so always request with i < j
    - Request format: "the correlation rho({i},{j})" or "rho({i},{j})"
    - Example: "the correlation rho(0,2)" matches "rho(0,2) = 1/3"
    - After hint: complete the double sum and report as a fraction
    - Answer: sigma_p^2 as {"numerator": n, "denominator": d}

20. **Birth-Death Chain Missing Rate (birth_death_missing_rate)**
    - Compute the steady-state probability pi(k) of a birth-death Markov chain
    - States {0, 1, ..., n-1}; birth rates lambda_i (state i -> i+1),
      death rates mu_i (state i -> i-1)
    - Agent A holds all rates as NATURAL LANGUAGE sentences EXCEPT one rate
      (either lambda_i or mu_i, randomised)
    - You must scan all expected birth and death rate sentences to find the gap
    - Request format: "the birth rate lambda_{i}" or "the death rate mu_{i}"
    - Example: "the birth rate lambda_2" matches "lambda_2 = 4"
    - Example: "the death rate mu_1" matches "mu_1 = 3"
    - After hint: apply detailed-balance to get pi; report pi(target) as a fraction
    - Answer: pi(k) as {"numerator": n, "denominator": d}

21. **Eigenvector Missing Entry (eigenvector_missing_entry)**
    - Compute (M^2)[missing_row][target_col] for a matrix M with one entry M[i][j] missing
    - Agent A holds all entries of M EXCEPT M[missing_row][missing_col], plus the
      full eigenvector v (given component by component)
    - Crucially, v[missing_col] = 0, so the eigenvector equation CANNOT recover M[i][j]
    - You must scan all (row, col) matrix entry sentences to find the gap
    - Request format: "the entry M[{i}][{j}]" or "M[{i}][{j}]"
    - Example: "the entry M[1][2]" matches "M[1][2] = 5"
    - After hint: compute M^2 = M*M and report the specific entry
    - Answer: (M^2)[missing_row][target_col] as {"value": integer}

22. **Steady-State Emission (steady_state_missing_emission)**
    - Compute P(observe symbol o) = sum_s pi[s] * e(s, o) for an HMM
    - Agent A holds the full transition matrix P (all entries), all emission probabilities
      e(state, symbol) EXCEPT one entry e(missing_state, target_symbol)
    - A must first compute steady-state pi from P, then discover which emission is missing
    - The query symbol IS the symbol whose emission is missing (this is by design)
    - Request format: "the emission probability e({s}, {o})" or "e({s},{o})"
    - Example: "the emission probability e(2, 1)" matches "e(2, 1) = 1/3"
    - After hint: complete the emission-weighted sum and report as a fraction
    - Answer: P(observe o) as {"numerator": n, "denominator": d}
"""

# ---------------------------------------------------------------------------
# Notation guide for Agent B semantic matching
# ---------------------------------------------------------------------------
NOTATION_GUIDE = """
## Notation Guide — How to Recognize a Semantic Match

A request is a MATCH if it refers to the same mathematical quantity as one of
your constraints, even if the wording differs. Use the following rules:

### Grid / Positional Notation
- u_i_j (laplace_grid):
    "the boundary value at position (i,j)" = "the value of u_i_j" = u_i_j
    Example: A asks "boundary value at (3,1)" → matches your "u_3_1 = 24" ✓
- m[i][j] (matrix_completion):
    "the value at position (i,j)" = "the matrix entry m[i][j]" = m[i][j]
    Example: A asks "value at position (2,3)" → matches your "m[2][3] = 5" ✓
- x_i_j or cell(i,j) (discrete_tomography):
    "the cell value at (i,j)" = x_i_j = cell(i,j)
    Example: A asks "cell value at (1,2)" → matches your "x_1_2 = 1" ✓

### Measurement / Signal Notation
- y[k] (deconvolution):
    "the measurement y[k]" = "the convolution value at index k" = y[k]
    Example: A asks "convolution value at index 3" → matches your "y[3] = 12" ✓

### Modular Arithmetic
- x ≡ r mod M (crt_reconstruction):
    "your remaining congruence" = "the missing congruence" = x ≡ r mod M (regardless of M)
    "x mod M" = "the congruence for modulus M" = x ≡ r mod M (when M is specified)
    Example: A asks "your remaining congruence" → matches your single congruence ✓
    Example: A asks "x mod 13" → matches your "x ≡ 7 mod 13" ✓

### Polynomial Points
- p(x_b) (poly_interpolation):
    "your interpolation point" = "the missing interpolation point" = p(x_b) = y_b
    "the point p(X)" = p(X) (when X is specified)
    Example: A asks "your interpolation point" → matches your single point ✓
    Example: A asks "the point p(5)" → matches your "p(5) = -88" ✓

### Moment Notation
- moment_k (moment_problem):
    "the k-th moment" = "the value of moment_k" = moment_k
    Ordinals map to indices: 1st→1, 2nd→2, 3rd→3, 4th→4, etc.
    Example: A asks "the 2nd moment" → matches your "moment_2 = 15" ✓

### Predicate Requests (Rank-Deficient Linear)
- "an equation involving variable X":
    Matches ANY of your constraints that contains variable X.
    Example: A asks "an equation involving variable a" → matches your "2a + 3b = 7" ✓
    because the variable a appears in that constraint.

### Probability / Bayes (bayes_missing_prior)
- P(H) (prior):
    "the prior P(H)" = "the base rate" = "the prior probability" = P(H)
    Example: A asks "the prior P(H)" → matches your "P(H) = 1/3" ✓
    Example: A asks "the base rate"  → matches your "P(H) = 1/3" ✓

### Piecewise Threshold (piecewise_missing_threshold)
- threshold t:
    "the threshold t" = "the branch threshold" = "the value of t" = threshold
    Example: A asks "the threshold t"     → matches your "threshold: t = -3" ✓
    Example: A asks "the branch threshold" → matches your "threshold: t = -3" ✓

### Markov Chain Transition (markov_missing_transition)
- P(i -> j):
    "the transition probability P(i -> j)" = "P(i -> j)" = transition from state i to state j
    Example: A asks "the transition probability P(2 -> 1)" -> matches your "P(2 -> 1) = 1/4" ✓
    Example: A asks "P(2 -> 1)" -> matches your "P(2 -> 1) = 1/4" ✓
- You hold exactly ONE entry P(i -> j); only a request for that exact (i, j) pair is a match
- NOT a match: a P(i -> j) where i or j differs from your constraint

### Linear System Coefficient (linear_system_missing_coeff)
- A[i][j] coefficient:
    "the coefficient of x{j} in equation {i}" = "A[{i}][{j}]" = the matrix entry at row i, col j
    Example: A asks "the coefficient of x1 in equation 2" -> matches your "A[2][1] = 3" ✓
    Example: A asks "A[2][1]" -> matches your "A[2][1] = 3" ✓
- You hold exactly ONE coefficient A[i][j]; only that exact (i, j) is a match

### Circuit Resistance (circuit_missing_resistance)
- R_k resistance value:
    "the resistance of R{k}" = "R{k}" = the ohm value of resistor number k
    Example: A asks "the resistance of R3" -> matches your "R3 = 5 ohms" ✓
    Example: A asks "R3" -> matches your "R3 = 5 ohms" ✓
- You hold exactly ONE resistor value; only that exact resistor index is a match

### Portfolio Correlation (portfolio_variance_missing_corr)
- rho(i,j) correlation:
    "the correlation rho({i},{j})" = "rho({i},{j})" = pairwise correlation of assets i and j
    Note: rho(i,j) = rho(j,i); both orderings match the same constraint
    Example: A asks "the correlation rho(0,2)" -> matches your "rho(0,2) = 1/3" ✓
    Example: A asks "rho(2,0)" -> also matches "rho(0,2) = 1/3" (symmetric) ✓
- You hold exactly ONE correlation; only that pair is a match

### Birth-Death Rate (birth_death_missing_rate)
- lambda_i (birth rate) or mu_i (death rate):
    "the birth rate lambda_{i}" = "lambda_{i}" = birth rate from state i to state i+1
    "the death rate mu_{i}" = "mu_{i}" = death rate from state i to state i-1
    Example: A asks "the birth rate lambda_2" -> matches your "lambda_2 = 4" ✓
    Example: A asks "the death rate mu_1" -> matches your "mu_1 = 3" ✓
- You hold exactly ONE rate; only that exact rate index and type (birth/death) is a match

### Eigenvector Missing Entry (eigenvector_missing_entry)
- M[i][j] matrix entry:
    "the entry M[{i}][{j}]" = "M[{i}][{j}]" = matrix entry at row i, column j
    Example: A asks "the entry M[1][2]" -> matches your "M[1][2] = 5" ✓
    Example: A asks "M[1][2]" -> matches your "M[1][2] = 5" ✓
- You hold exactly ONE entry M[i][j]; only that exact (i, j) is a match

### Steady-State Emission (steady_state_missing_emission)
- e(s, o) emission probability:
    "the emission probability e({s},{o})" = "e({s},{o})" = probability state s emits symbol o
    Example: A asks "the emission probability e(2, 1)" -> matches your "e(2, 1) = 1/3" ✓
    Example: A asks "e(2,1)" -> matches your "e(2, 1) = 1/3" ✓
- You hold exactly ONE emission; only that exact (state, symbol) pair is a match

### What is NOT a match (decline these)
- sign(x0) ≠ value of x0 (different quantities — sign is +-1, value is the number itself)
- p(3) ≠ p(4) (different evaluation point)
- u_3_1 ≠ u_3_2 (different grid positions)
- m[1][2] ≠ m[2][1] (different matrix entries)
- P(H) ≠ P(E|H) (prior ≠ likelihood — these are different quantities)
- P(0 -> 1) ≠ P(1 -> 0) (different source states — direction matters)
"""


# ---------------------------------------------------------------------------
# Target description helper
# ---------------------------------------------------------------------------
def _format_target_description(instance: Dict[str, Any]) -> str:
    """Return a precise statement of WHAT Agent A must compute as its final answer.

    This reads global_metadata (coordinates and indices only — never values or
    secrets held by B) and formats a family-specific instruction so A knows
    exactly which quantity to report.

    Without this, A solves the full system correctly but has no way to know
    which of the solved variables is the target. For example in laplace_grid,
    A solves for u_1_1, u_1_2, u_2_1, u_2_2 but must report only u_2_1.

    Safety: only target_cell / target_x / target_n / etc. coordinates are
    read — never boundary_values or any field that might contain B's secret data.
    """
    family = instance.get("family", "")
    gm = instance.get("global_metadata", {})
    answer_type = instance.get("answer_type", "")

    # ── family-specific target instructions ──────────────────────────────────

    if family == "linear_system_separator":
        vars_ = gm.get("variables", ["a", "b", "c"])
        report_fmt = ", ".join(f'"{v}": <integer>' for v in vars_)
        return (
            f"Solve the full linear system for all variables: {', '.join(vars_)}.\n"
            f"Agent B holds the value of exactly one variable (the separator). "
            f"Try requesting each variable by name until B provides it.\n"
            f'Report as: {{{report_fmt}}}'
        )

    elif family == "laplace_grid":
        tc = gm.get("target_cell")
        if tc is not None:
            i, j = tc[0], tc[1]
            return (
                f"Compute u_{i}_{j} — the grid value at row {i}, column {j}.\n"
                f'Report as: {{"value": <integer>}}'
            )

    elif family == "poly_interpolation":
        tx = gm.get("target_x")
        if tx is not None:
            return (
                f"Compute p({tx}) — the value of the polynomial at x = {tx}.\n"
                f'Report as: {{"value": <integer>}}'
            )

    elif family == "recurrence_missing_init":
        tn = gm.get("target_n")
        if tn is not None:
            return (
                f"Compute a({tn}) — the {tn}-th term of the recurrence sequence.\n"
                f'Report as: {{"value": <integer>}}'
            )

    elif family == "linear_system_separator":
        tv = gm.get("target_var")
        if tv is not None:
            return (
                f"Compute the value of variable {tv}.\n"
                f'Report as: {{"value": <integer>}}'
            )
        # Some variants ask for the full solution
        tvs = gm.get("target_vars")
        if tvs is not None:
            vars_str = ", ".join(str(v) for v in tvs)
            return (
                f"Compute the values of variables: {vars_str}.\n"
                f'Report as: {{"var1": <integer>, "var2": <integer>, ...}}'
            )

    elif family == "rankdef_linear_shared":
        tv = gm.get("target_var")
        if tv is not None:
            return (
                f"Compute the value of variable {tv}.\n"
                f'Report as: {{"value": <integer>}}'
            )
        tvs = gm.get("target_vars")
        if tvs is not None:
            vars_str = ", ".join(str(v) for v in tvs)
            return (
                f"Compute the values of variables: {vars_str}.\n"
                f'Report as: {{"var1": <integer>, "var2": <integer>, ...}}'
            )

    elif family == "crt_reconstruction":
        return (
            "Compute x — the unique integer satisfying all congruences (by CRT).\n"
            f'Report as: {{"value": <integer>}}'
        )

    elif family == "deconvolution":
        ti = gm.get("target_index")
        if ti is not None:
            return (
                f"Compute x[{ti}] — the recovered signal value at index {ti}.\n"
                f'Report as: {{"value": <integer>}}'
            )

    elif family == "phase_retrieval":
        te = gm.get("target_element")
        if te is not None:
            return (
                f"Compute x[{te}] — element {te} of the recovered sequence.\n"
                f'Report as: {{"value": <integer>}}'
            )
        # Some variants ask for the full sequence
        if answer_type == "dict_int":
            return (
                "Compute the full recovered sequence x[0], x[1], x[2], x[3].\n"
                f'Report as: {{"x0": <int>, "x1": <int>, "x2": <int>, "x3": <int>}}'
            )

    elif family == "matrix_completion":
        tc = gm.get("target_cell") or gm.get("target_entry")
        if tc is not None:
            i, j = tc[0], tc[1]
            return (
                f"Compute m[{i}][{j}] — the matrix entry at row {i}, column {j}.\n"
                f'Report as: {{"value": <integer>}}'
            )

    elif family == "discrete_tomography":
        tc = gm.get("target_cell")
        if tc is not None:
            i, j = tc[0], tc[1]
            return (
                f"Compute the cell value at row {i}, column {j} (x_{i}_{j}).\n"
                f'Report as: {{"value": <integer>}}'
            )
        ts = gm.get("target_sum")
        if ts is not None:
            return (
                f"Compute the requested sum: {ts}.\n"
                f'Report as: {{"value": <integer>}}'
            )

    elif family == "graph_path_sums":
        te = gm.get("target_edge")
        if te is not None:
            if isinstance(te, (list, tuple)) and len(te) == 2:
                return (
                    f"Compute the edge weight e_{te[0]}{te[1]}.\n"
                    f'Report as: {{"value": <integer>}}'
                )
            return (
                f"Compute the edge weight: {te}.\n"
                f'Report as: {{"value": <integer>}}'
            )
        tp = gm.get("target_path")
        if tp is not None:
            return (
                f"Compute the path sum: {tp}.\n"
                f'Report as: {{"value": <integer>}}'
            )

    elif family == "moment_problem":
        tm = gm.get("target_moment")
        if tm is not None:
            return (
                f"Compute moment_{tm} — the {tm}-th moment of the distribution.\n"
                f'Report as: {{"value": <integer>}}'
            )
        if answer_type == "dict_int":
            return (
                "Compute the full probability distribution.\n"
                f'Report as: {{"p0": <int>, "p1": <int>, ...}}'
            )

    elif family == "geometry_coordinates":
        pq = gm.get("point_q")
        if pq is not None:
            return (
                f"Find the intersection point (x, y) of the two lines, then compute\n"
                f"its integer distance to Q = ({pq[0]}, {pq[1]}).\n"
                'Report as: {"value": <integer>}'
            )
        # Legacy keys kept for backwards compatibility
        td = gm.get("target_distance")
        if td is not None:
            return (
                "Compute the distance between the intersection point and Q.\n"
                f'Report as: {{"value": <integer>}}'
            )
        tp = gm.get("target_point")
        if tp is not None:
            return (
                f"Compute the coordinates of point {tp}.\n"
                f'Report as: {{"x": <integer>, "y": <integer>}}'
            )

    elif family == "bayes_missing_prior":
        return (
            "Compute P(H|E) using Bayes' theorem:\n"
            "  P(H|E) = P(E|H) * P(H) / [P(E|H) * P(H) + P(E|not_H) * (1 - P(H))]\n"
            "Simplify the result to a FULLY REDUCED fraction.\n"
            'Report as: {"numerator": <integer>, "denominator": <integer>}'
        )

    elif family == "markov_missing_transition":
        ts = gm.get("target_state", 0)
        n_states = gm.get("n_states", 2)
        return (
            f"Compute pi({ts}) — the steady-state probability of being in state {ts}.\n"
            f"The chain has {n_states} states (0 to {n_states - 1}). Once you have all transition\n"
            f"probabilities, solve the balance equations pi * P = pi with sum(pi) = 1.\n"
            "Simplify the result to a FULLY REDUCED fraction.\n"
            'Report as: {"numerator": <integer>, "denominator": <integer>}'
        )

    elif family == "piecewise_missing_threshold":
        x_val = gm.get("x")
        if x_val is not None:
            return (
                f"Compute f({x_val}): once you know the threshold t, determine the active branch\n"
                f"  (branch 0: x < t, branch 1: x >= t) and evaluate that formula at x = {x_val}.\n"
                'Report as: {"value": <integer>}'
            )
        return (
            "Compute f(x): once you know the threshold t, determine the active branch\n"
            "  (branch 0: x < t, branch 1: x >= t) and evaluate that formula at x.\n"
            'Report as: {"value": <integer>}'
        )

    elif family == "linear_system_missing_coeff":
        n = gm.get("n", 2)
        tk = gm.get("target_k", 0)
        vars_str = ", ".join(f"x{i}" for i in range(n))
        return (
            f"Scan all expected coefficient sentences for rows 0..{n-1} and columns 0..{n-1}.\n"
            f"Find which (row, col) pair has no sentence — that is the missing A[i][j].\n"
            f"Ask B for it, then solve the complete {n}x{n} system for {vars_str}.\n"
            f"Report x{tk} as a fully reduced fraction.\n"
            'Report as: {"numerator": <integer>, "denominator": <integer>}'
        )

    elif family == "circuit_missing_resistance":
        n_r = gm.get("n_resistors", 3)
        resistors = ", ".join(f"R{i+1}" for i in range(n_r))
        return (
            f"Check all {n_r} resistors ({resistors}) — find which one has no value sentence.\n"
            f"Ask B for that resistor's resistance, then evaluate the series/parallel tree.\n"
            "Simplify R_eq to a fully reduced fraction.\n"
            'Report as: {"numerator": <integer>, "denominator": <integer>}'
        )

    elif family == "portfolio_variance_missing_corr":
        n = gm.get("n_assets", 2)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        pairs_str = ", ".join(f"rho({i},{j})" for i, j in pairs)
        return (
            f"Check all {len(pairs)} correlation pairs: {pairs_str}.\n"
            f"Find which rho(i,j) has no sentence — ask B for it.\n"
            "Compute sigma_p^2 = sum_i sum_j w_i*w_j*sigma_i*sigma_j*rho(i,j) (rho(i,i)=1).\n"
            "Simplify to a fully reduced fraction.\n"
            'Report as: {"numerator": <integer>, "denominator": <integer>}'
        )

    elif family == "birth_death_missing_rate":
        n = gm.get("n_states", 3)
        ts = gm.get("target_state", 0)
        return (
            f"Scan all expected birth rates lambda_0..lambda_{n-2} and death rates mu_1..mu_{n-1}.\n"
            f"Find which rate sentence is absent — ask B for it (specify lambda or mu and index).\n"
            f"Apply detailed balance: pi_i = pi_0 * (lambda_0*...*lambda_{{i-1}}) / (mu_1*...*mu_i).\n"
            f"Normalise so sum(pi) = 1, then report pi({ts}) as a fully reduced fraction.\n"
            'Report as: {"numerator": <integer>, "denominator": <integer>}'
        )

    elif family == "eigenvector_missing_entry":
        n = gm.get("n", 2)
        mr = gm.get("missing_row", 0)
        tc = gm.get("target_col", 0)
        return (
            f"Scan all expected M[r][c] entries for r,c in 0..{n-1}.\n"
            f"Find which (r,c) pair has no sentence — that is M[missing_row][missing_col].\n"
            f"Note: v[missing_col] = 0, so the eigenvector equation cannot recover that entry.\n"
            f"Ask B for M[missing_row][missing_col], then compute M^2 = M*M.\n"
            f"Report (M^2)[{mr}][{tc}] as an integer.\n"
            'Report as: {"value": <integer>}'
        )

    elif family == "steady_state_missing_emission":
        n = gm.get("n_states", 2)
        m = gm.get("n_symbols", 2)
        ts = gm.get("target_symbol", 0)
        ms = gm.get("missing_state", 0)
        return (
            f"Step 1: Solve the steady-state balance equations pi*P = pi (sum=1) for all {n} states.\n"
            f"Step 2: Try to compute P(observe symbol {ts}) = sum_s pi[s]*e(s,{ts}).\n"
            f"Step 3: You will find e({ms},{ts}) is missing — ask B for it.\n"
            f"Step 4: Complete the sum and report as a fully reduced fraction.\n"
            'Report as: {"numerator": <integer>, "denominator": <integer>}'
        )

    # ── generic fallback — answer_type based ─────────────────────────────────
    if answer_type == "int":
        return 'Compute the single requested value.\nReport as: {"value": <integer>}'
    elif answer_type == "dict_int":
        return 'Compute the requested values.\nReport as: {"var1": <integer>, "var2": <integer>, ...}'
    return "Compute the final answer as a JSON object appropriate to the problem."


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

def system_prompt_agent_a() -> str:
    """System prompt for Agent A (the requester)."""
    return f"""You are Agent A in a collaborative math problem-solving system.

## Your Situation
- You have PRIVATE constraints that only you can see
- Your view alone is UNDERDETERMINED by exactly one piece of information
- Agent B holds that ONE missing piece (a boundary value, variable value, congruence, etc.)
- You must REQUEST that missing piece from Agent B, then compute the final answer yourself

## CRITICAL DISTINCTION: Target vs. Request

The problem tells you WHAT to compute as your final answer (the target).
That is NOT what you should request from Agent B.

Agent B holds the MISSING PIECE — one specific value not in your constraints —
that makes the system solvable. You compute the target yourself after getting it.

## Mechanical pre-check BEFORE making any request

Run this exact check on your constraints first:

  1. Write down the TARGET variable (shown in Step 2 of your prompt).
  2. Search your constraints for any line where the target is the SUBJECT:
       - "4 [target] = ..." (system equation with target on LHS)
       - "[target] = <number>" (direct assignment)
     If EITHER exists → the target is COMPUTABLE, NOT what B holds.
     DO NOT request it. Continue to step 3.
  3. Collect all variables that appear on the RIGHT-hand side of your equations
     but have NO "4 X = ..." line AND NO "X = <number>" line in your constraints.
     That variable is what B holds — REQUEST IT.

Example (laplace_grid, target = u_2_1):
  Check: is there a line "4 u_2_1 = ..." in your constraints? YES → u_2_1 is computable.
  Now scan RHS variables: u_0_1, u_2_1, u_1_0, u_1_2, u_0_2, u_2_2, u_1_1, u_1_3,
                          u_3_1, u_2_0, u_3_2, u_2_3
  Which of these has NO "4 X = ..." AND NO "X = number"? → u_3_1
  → Request: "the value of u_3_1"

## Your Task
1. Run the mechanical pre-check above — identify the ONE variable B holds
2. Request it using its exact variable name (e.g., "the value of u_3_1")
3. Agent B will provide it if your request matches what they hold
4. If B declines, you identified the wrong variable — try the next candidate from step 3
5. Once you receive it, solve the complete system and report the target

## Important Rules
- NEVER request the target — it is computable once you have B's missing piece
- Use exact variable names in requests, not spatial descriptions
- Do NOT submit a final answer until you have received B's information

{PROBLEM_FAMILIES}
"""


def system_prompt_agent_b() -> str:
    """System prompt for Agent B (the information holder)."""
    return f"""You are Agent B in a collaborative math problem-solving system.

## Your Situation
- You have PRIVATE constraints/information that only you can see
- Agent A is trying to solve a problem and needs specific information from you
- You must provide information if Agent A's request SEMANTICALLY MATCHES what you have
- You must DECLINE if Agent A is asking for something you do not have

## What "Semantic Match" means
A request is a match when it refers to the SAME MATHEMATICAL QUANTITY as one of
your constraints — even if the wording is different. It is NOT a literal string
comparison. Use the Notation Guide below to identify matches.

## Your Task
1. Read Agent A's request carefully
2. Apply the Notation Guide below to translate the request into variable notation
3. Look at YOUR private constraints — does ANY of them give the value of that variable?
4. If YES (semantic match found):
   - Set type="offer", has_exact_match=true
   - In constraint_quoted, copy the EXACT constraint text from your private data
   - In hint, state the specific value/info extracted from that constraint
5. If NO (no match — you genuinely do not have it):
   - Set type="decline"
   - In decline_message, simply state: "I do not have that specific information."
   - Do NOT provide hints or clues about what you do have

## CRITICAL RULES
- NEVER make up information — only provide what is EXPLICITLY in your private constraints
- You must QUOTE the exact constraint text before providing info
- Do NOT hallucinate or infer values not explicitly stated in your constraints
- Do NOT hint at what other information you hold when declining
- Quantity mismatch examples (these must be DECLINED):
    * A asks for "the value of x0" but you only have "sign(x0) = +1" → DECLINE
    * A asks for "p(3)" but you only have "p(4) = 10" → DECLINE
    * A asks for "u_3_1" but you only have "u_3_2 = 7" → DECLINE

{NOTATION_GUIDE}

{PROBLEM_FAMILIES}
"""


# ---------------------------------------------------------------------------
# User prompts
# ---------------------------------------------------------------------------

def _format_few_shot_section(examples: List[Dict[str, Any]]) -> str:
    """Format few-shot examples into a prompt section for Agent A.

    Each example shows: constraints, the correct request, B's response, and the
    final answer - so Agent A learns the pattern for this family/difficulty.
    """
    if not examples:
        return ""

    lines = [
        "## Worked Examples (same family and difficulty)",
        "Study these examples to understand the pattern: what to request, "
        "what Agent B provides, and the correct answer format.\n",
    ]

    for i, ex in enumerate(examples, 1):
        constraints_str = "\n".join(f"    - {c}" for c in ex["constraints"])
        answer_str = json.dumps(ex["answer"], sort_keys=True)

        lines.append(f"### Example {i}")
        lines.append(f"  Your constraints:")
        lines.append(constraints_str)
        lines.append(f"  Your request: \"{ex['request']}\"")
        lines.append(f"  Agent B responds: \"{ex['hint']}\"")
        lines.append(f"  Final answer: {answer_str}")
        lines.append("")

    return "\n".join(lines) + "\n"


def user_prompt_agent_a(
    instance: Dict[str, Any],
    shared_history: list,
    attempt_num: int,
    max_attempts: int,
) -> str:
    """User prompt for Agent A with problem context and history."""
    view = None
    for v in instance["agent_views"]:
        if v["agent_id"] == "A":
            view = v
            break
    if view is None:
        raise ValueError("No view found for Agent A")

    constraints = "\n".join(f"  - {c}" for c in view["private_data"]["constraints_text"])
    unknowns = view["private_data"].get("unknowns", [])

    # Build history section
    history_text = ""
    if shared_history:
        history_text = "\n## Previous Exchange\n"
        for msg in shared_history:
            content = msg.get("message", {})
            msg_type = content.get("type", "?")
            if msg_type == "request":
                history_text += f"You requested: \"{content.get('request', '')}\"\n"
            elif msg_type == "offer":
                history_text += f"Agent B provided: \"{content.get('hint', '')}\"\n"
            elif msg_type == "decline":
                history_text += f"Agent B declined: \"{content.get('message', '')}\"\n"

    target_description = _format_target_description(instance)

    # Few-shot examples (only on first attempt to avoid bloating retries)
    few_shot_text = ""
    if attempt_num == 1:
        few_shot_examples = instance.get("few_shot_examples", [])
        few_shot_text = _format_few_shot_section(few_shot_examples)

    return f"""## Problem Instance
Family: {instance['family']}
Unknowns to solve: {', '.join(unknowns) if unknowns else 'see constraints'}

## Your Private Constraints
{constraints}

{few_shot_text}{history_text}
## Step 1 — Find What to Request from Agent B (run this before anything else)

  CHECK 1: Note the target variable from Step 2. Search your constraints above
           for any line where it is the SUBJECT — either:
             "4 [target] = ..."   or   "[target] = <number>"
           If such a line exists → the target is COMPUTABLE. Do NOT request it. Go to CHECK 2.

  CHECK 2: Go through every variable on the RIGHT-hand side of your system equations.
           For each one, check your constraints:
             - Has "4 X = ..." line?    → computable, skip it
             - Has "X = <number>" line? → already known, skip it
             - Neither?                 → THIS is what B holds. Request it now.

  Request that variable by exact name: e.g., "the value of u_3_1"

  ⚠ Do NOT request the target. Do NOT guess. Only request the variable from CHECK 2.

## Step 2 — What You Must Compute (final answer target)
{target_description}

⚠ Before submitting your final answer, verify your solution satisfies ALL
  equations simultaneously. If any equation is violated, your arithmetic is wrong.

## Current Status
Request attempt: {attempt_num} of {max_attempts}

## Instructions
{"Follow the Step 1 algorithm above to identify the missing piece, then request it from Agent B." if attempt_num == 1 else "Based on the exchange above, either refine your request or — if you now have the missing piece — solve the COMPLETE system of equations simultaneously and submit the target value."}
"""


def user_prompt_agent_b(
    instance: Dict[str, Any],
    request_msg: Dict[str, Any],
) -> str:
    """User prompt for Agent B with their private info and A's request."""
    view = None
    for v in instance["agent_views"]:
        if v["agent_id"] == "B":
            view = v
            break
    if view is None:
        raise ValueError("No view found for Agent B")

    constraints = view["private_data"]["constraints_text"]
    constraints_formatted = "\n".join(f"  {i+1}. \"{c}\"" for i, c in enumerate(constraints))

    request_text = request_msg.get("request", "")
    reasoning_text = request_msg.get("reasoning", "")
    family = instance.get("family", "")

    return f"""## YOUR PRIVATE CONSTRAINTS (this is ALL you have)
{constraints_formatted}

## Agent A's Request
Family: {family}
A's reasoning: "{reasoning_text}"
A's request: "{request_text}"

## Your Task
Step 1 — Translate A's request into variable notation using the Notation Guide
         (from your system instructions). For example:
           "boundary value at (3,1)" → u_3_1
           "value at position (2,3)" → m[2][3]
           "x mod 13"               → x ≡ ? mod 13
           "2nd moment"             → moment_2
           "equation involving a"   → any constraint containing variable a
           "convolution value at 3" → y[3]
           "the prior P(H)"              → P(H) = num/denom
           "the base rate"               → P(H) = num/denom
           "the threshold t"             → threshold: t = <value>
           "the branch threshold"        → threshold: t = <value>
           "your remaining congruence"   → x ≡ r mod M (your single congruence)
           "your interpolation point"    → p(x_b) = y_b (your single point)
           "coefficient of x1 in eq 2"  → A[2][1] = <value>
           "A[2][1]"                     → A[2][1] = <value>
           "resistance of R3"           → R3 = <value> ohms
           "R3"                          → R3 = <value> ohms
           "correlation rho(0,2)"        → rho(0,2) = num/denom
           "rho(2,0)"                    → same as rho(0,2) (symmetric)
           "birth rate lambda_2"         → lambda_2 = <value>
           "death rate mu_1"             → mu_1 = <value>
           "entry M[1][2]"               → M[1][2] = <value>
           "M[1][2]"                     → M[1][2] = <value>
           "emission probability e(2,1)" → e(2, 1) = num/denom
           "e(2,1)"                      → e(2, 1) = num/denom

Step 2 — Check your constraints for a match. Use this ORDER:

         FIRST check for a direct value assignment that matches A's request
         (e.g. "u_3_1 = 24", "a(1) = 7", "x ≡ 5 mod 13").
         If found → offer THAT constraint. Stop here.

         ONLY IF no value assignment matches, check for a system equation match
         (e.g. "4 u_1_1 = ...").
         WARNING: System equations are often shared between agents — A may already
         have it. Only offer a system equation if it is the only possible match.

         This is a SEMANTIC check, not a word-for-word comparison.

Step 3a — If you HAVE it (semantic match):
   - Set type="offer", has_exact_match=true
   - constraint_quoted: copy the EXACT constraint text from the list above
   - hint: state the specific value (e.g., "u_3_1 = 24" or "sign(x0) = +1")

Step 3b — If you DO NOT HAVE it (no match):
   - Set type="decline"
   - decline_message: "I do not have that specific information."
   - Do NOT reveal what other information you hold

CRITICAL: Do NOT make up information. Only provide what is EXPLICITLY listed above.
Do NOT provide hints about what other constraints you have when declining.
"""