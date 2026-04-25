"""Max-plus algebra helpers used by the Python CAL notebooks."""

from __future__ import annotations

import numpy as np

Matrix = np.ndarray
Vector = np.ndarray

EPS = float("-inf")
INF = float("inf")


def _ensure_matrix(value: Matrix, name: str) -> None:
    if not isinstance(value, np.ndarray) or value.ndim != 2:
        raise TypeError(f"{name} must be a 2d numpy array")


def oplus(A: Matrix, B: Matrix) -> Matrix:
    """Max-plus addition: element-wise max."""
    _ensure_matrix(A, "A")
    _ensure_matrix(B, "B")
    if A.shape != B.shape:
        raise TypeError(f"A and B must have the same shape: A.shape={A.shape}, B.shape={B.shape}")
    return np.maximum(A, B)


def odot(a: Vector, b: Vector) -> float:
    """Max-plus dot product of two vectors: max_i(a[i] + b[i])."""
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError("Vector a and b must be numpy arrays")
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Vector a and b must be 1D arrays")
    if len(a) != len(b):
        raise ValueError("Vector a and b must have the same shape")
    return float(np.max(a + b))


def otimes(A: Matrix, B: Matrix) -> Matrix:
    """Max-plus multiplication: (A otimes B)[i,j] = max_k(A[i,k] + B[k,j])."""
    _ensure_matrix(A, "A")
    _ensure_matrix(B, "B")
    if A.shape[1] != B.shape[0]:
        raise TypeError(
            "A's 2nd dimension does not match B's 1st dimension: "
            f"A.shape={A.shape}, B.shape={B.shape}"
        )

    if A.shape == (1, 1) or B.shape == (1, 1):
        return A + B

    result = np.zeros((A.shape[0], B.shape[1]))
    B_T = B.T
    for i, row_a in enumerate(A):
        for j, row_b in enumerate(B_T):
            result[i][j] = odot(row_a, row_b)
    return result


def pow_otimes(A: Matrix, n: int) -> Matrix:
    """Max-plus matrix power A^n."""
    _ensure_matrix(A, "A")
    if n < 1:
        raise ValueError("n must be at least 1")

    result = A
    for _ in range(n - 1):
        result = otimes(result, A)
    return result


def trace(A: Matrix) -> float:
    """Max-plus trace: max of the diagonal."""
    _ensure_matrix(A, "A")
    return float(np.max(A.diagonal()))


def identity(size: int) -> Matrix:
    """Create a max-plus identity matrix."""
    result = np.full((size, size), EPS)
    np.fill_diagonal(result, 0)
    return result


def epsilon_matrix(rows: int, cols: int) -> Matrix:
    """Create a matrix filled with EPS."""
    return np.full((rows, cols), EPS)


def reset(a: Matrix) -> Matrix:
    """Return an EPS-filled row vector with the same width as the input vector."""
    _ensure_matrix(a, "a")
    return np.array([[EPS] * a.shape[1]])


def star(A: Matrix) -> Matrix:
    """Kleene star A* = I oplus A oplus ... oplus A^(n-1)."""
    _ensure_matrix(A, "A")

    result = identity(len(A))
    for i in range(1, len(A)):
        result = oplus(result, pow_otimes(A, i))
    return result


def plus(A: Matrix) -> Matrix:
    """Max-plus plus A+ = A oplus A^2 oplus ... oplus A^n."""
    _ensure_matrix(A, "A")

    result = A
    for i in range(1, len(A) + 1):
        result = oplus(result, pow_otimes(A, i))
    return result
