"""
gaussian.metrics
================
Numerical utilities: SPD checks, error norms, convergence measures.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, sqrtm

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


def w2_gaussian(S1: np.ndarray, S2: np.ndarray) -> float:
    A = sqrtm(S1)
    return float(np.trace(S1 + S2 - 2.0 * sqrtm(A @ S2 @ A)))


def rho_empirical(T1: np.ndarray, T2: np.ndarray, Ms: list[np.ndarray]) -> float:
    vals = [w2_gaussian(T1 @ M @ T1, T2 @ M @ T2) for M in Ms]
    return float(np.mean(vals))


def empirical_loss(T: np.ndarray, Ms: list[np.ndarray], Ns: list[np.ndarray]) -> float:
    """
    Average Gaussian W2^2 loss: (1/(2N)) * Σ W2^2(T M_i T, N_i).
    """
    return 0.5 * np.mean([w2_gaussian(T @ M @ T, N) for M, N in zip(Ms, Ns)])
