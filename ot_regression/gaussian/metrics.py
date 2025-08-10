"""
gaussian.metrics
================
Numerical utilities: SPD checks, error norms, convergence measures.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import eigh

Array = np.ndarray


def is_spd(M: Array, tol: float = 1e-10) -> bool:
    """Check if a matrix is symmetric positive definite."""
    if not np.allclose(M, M.T, atol=tol):
        return False
    eigvals = eigh(M, eigvals_only=True)
    return np.all(eigvals > tol)


def frobenius_error(A: Array, B: Array) -> float:
    """Frobenius norm ‖A - B‖_F."""
    return np.linalg.norm(A - B, "fro")
