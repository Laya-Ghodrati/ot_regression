"""
gaussian.ot_gaussian
====================
Closed-form optimal transport maps between Gaussian measures.
"""

from __future__ import annotations
import numpy as np
import scipy.linalg
from typing import Tuple

Array = np.ndarray


def gaussian_ot_map(Sigma_src: Array, Sigma_tgt: Array) -> Array:
    """
    Compute the optimal transport matrix A between two centered Gaussians:
        N(0, Σ_src) → N(0, Σ_tgt)
    in the 2-Wasserstein sense.

    Formula:
        A = Σ_src^{-1/2} ( Σ_src^{1/2} Σ_tgt Σ_src^{1/2} )^{1/2} Σ_src^{-1/2}

    Parameters
    ----------
    Sigma_src : Array
        Source covariance (SPD).
    Sigma_tgt : Array
        Target covariance (SPD).

    Returns
    -------
    Array
        Transport matrix A such that A Σ_src A^T = Σ_tgt.
    """
    S_half = scipy.linalg.sqrtm(Sigma_src)
    S_half_inv = scipy.linalg.inv(S_half)
    inner = S_half @ Sigma_tgt @ S_half
    return S_half_inv @ scipy.linalg.sqrtm(inner) @ S_half_inv
