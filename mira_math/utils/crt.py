"""Chinese Remainder Theorem helpers."""
from __future__ import annotations

from typing import List, Tuple


def egcd(a: int, b: int) -> Tuple[int, int, int]:
    if b == 0:
        return a, 1, 0
    g, x, y = egcd(b, a % b)
    return g, y, x - (a // b) * y


def inv_mod(a: int, m: int) -> int:
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("no modular inverse")
    return x % m


def crt_solve(congruences: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Solve x ≡ r_i (mod m_i) for pairwise coprime moduli.

    Returns (x, M) where x is the smallest non-negative solution modulo M.
    """
    x = 0
    M = 1
    for m, r in congruences:
        if m <= 0:
            raise ValueError("modulus must be positive")
        inv = inv_mod(M, m)
        t = ((r - x) % m) * inv % m
        x = x + M * t
        M *= m
        x %= M
    return x, M


def product_moduli(congruences: List[Tuple[int, int]]) -> int:
    M = 1
    for m, _ in congruences:
        M *= m
    return M

