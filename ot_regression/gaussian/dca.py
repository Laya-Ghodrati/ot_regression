"""
gaussian.dca
============
Difference-of-Convex Algorithm for estimating the transport matrix
in Gaussian OT regression.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import inv, sqrtm
from scipy.optimize import minimize

from .metrics import (empirical_loss, frobenius_error, rho_empirical,
                      w2_gaussian)

Array = np.ndarray


def compute_G_matrices(Tk: Array, Ms: List[Array], Ns: List[Array]) -> List[Array]:
    """
    Compute the G_i matrices used in DCA updates.

    G_i = (T_k M_i T_k)^{-1/2}  [ (T_k M_i T_k)^{1/2} N_i (T_k M_i T_k)^{1/2} ]^{1/2} (T_k M_i T_k)^{-1/2}
    """
    Gs: List[Array] = []
    for M, N in zip(Ms, Ns):
        S = Tk @ M @ Tk
        S_half = sqrtm(S)
        S_half_inv = inv(S_half)
        SN = S_half @ N @ S_half
        G = S_half_inv @ sqrtm(SN) @ S_half_inv
        Gs.append(G)
    return Gs


def _objective(L_flat: Array, Ms: List[Array], Gs: List[Array], Tk: Array) -> float:
    """
    Objective for a single DCA subproblem with T = L L^T (L lower-triangular).
    """
    d = Ms[0].shape[0]
    L = np.tril(L_flat.reshape(d, d))
    T = L @ L.T
    cost = 0.0
    for M, G in zip(Ms, Gs):
        cost += np.trace(T @ M @ T) - 2 * np.trace(T @ M @ G @ Tk)
    return cost


def optimize_T(Ms: List[Array], Gs: List[Array], Tk: Array) -> Array:
    """Solve the convex subproblem in the DCA step."""
    d = Ms[0].shape[0]
    L_init = np.linalg.cholesky(np.eye(d))
    res = minimize(_objective, L_init.flatten(), args=(Ms, Gs, Tk), method="L-BFGS-B")
    L_opt = np.tril(res.x.reshape(d, d))
    return L_opt @ L_opt.T


def fit_gaussian_dca(
    Ms: List[Array],
    Ns: List[Array],
    *,
    T_true: Optional[Array] = None,
    max_iter: int = 10,
    verbose: bool = False,
) -> Tuple[Array, Dict[str, Any]]:
    """
    Fit Gaussian OT regression via DCA.

    Runs for exactly `max_iter` iterations.
    If T_true is provided, it is used only for tracking ρ(T, T_true) each iteration.

    Returns
    -------
    T_hat : np.ndarray
        Estimated transport matrix.
    history : dict
        {
          "error_true": [ρ(T_k, T_true) per iter] or None,
          "loss": [empirical W2^2 loss per iter],
          "num_iter": int
        }
    """

    d = Ms[0].shape[0]
    Tk = np.eye(d)

    # record iteration 0
    error_true: Optional[list[float]] = None
    if T_true is not None:
        error_true = [rho_empirical(Tk, T_true, Ms)]

    loss_history: list[float] = [empirical_loss(Tk, Ms, Ns)]

    for it in range(1, max_iter + 1):
        Gs = compute_G_matrices(Tk, Ms, Ns)
        Tk = optimize_T(Ms, Gs, Tk)

        if error_true is not None:
            error_true.append(rho_empirical(Tk, T_true, Ms))

        loss = empirical_loss(Tk, Ms, Ns)
        loss_history.append(loss)

        if verbose:
            print(f"[iter {it}] loss={loss:.3e}")

    history: Dict[str, Any] = {
        "error_true": error_true,
        "loss": loss_history,
        "num_iter": max_iter,
    }
    return Tk, history
