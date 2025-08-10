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

from .metrics import frobenius_error

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
    tol: float = 1e-8,
    verbose: bool = False,
) -> Tuple[Array, Dict[str, Any]]:
    """
    Fit the Gaussian OT regression map using DCA.

    Convergence is based on parameter change ‖T_{k+1} - T_k‖_F, independent of ground truth.
    If T_true is provided, we record error-to-true each iteration for diagnostics.

    Returns
    -------
    T_hat : ndarray
        Estimated transport matrix.
    history : dict
        {
          "delta_T": [‖T_{k+1}-T_k‖_F per iter],
          "error_true": [‖T_k - T_true‖_F per iter] or None,
          "num_iter": int
        }
    """
    d = Ms[0].shape[0]
    Tk = np.eye(d)

    delta_T: list[float] = []
    error_true: Optional[list[float]] = [] if T_true is not None else None

    for it in range(1, max_iter + 1):
        Gs = compute_G_matrices(Tk, Ms, Ns)
        Tk_new = optimize_T(Ms, Gs, Tk)

        # parameter-change stopping
        delta = frobenius_error(Tk_new, Tk)
        delta_T.append(delta)

        if error_true is not None:
            error_true.append(frobenius_error(Tk_new, T_true))  # type: ignore[arg-type]

        if verbose:
            msg = f"[iter {it}] ΔT={delta:.3e}"
            if error_true is not None:
                msg += f", ‖T−T_true‖={error_true[-1]:.3e}"
            print(msg)

        Tk = Tk_new
        if delta < tol:
            break

    history: Dict[str, Any] = {
        "delta_T": delta_T,
        "error_true": error_true,
        "num_iter": len(delta_T),
    }
    return Tk, history
