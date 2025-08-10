"""
gaussian.generate
=================
Utilities for generating synthetic Gaussian regression data:
SPD matrices, noise, true transport maps, and dataset construction.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.stats import ortho_group

Array = np.ndarray


def generate_spd_matrix(
    d: int, eig_min: float = 0.1, eig_max: float = 10.0, seed: int | None = None
) -> Array:
    """
    Generate a random symmetric positive definite (SPD) matrix of size (d, d).

    Parameters
    ----------
    d : int
        Dimension.
    eig_min, eig_max : float
        Minimum and maximum eigenvalues for conditioning.
    seed : int | None
        Optional random seed.

    Returns
    -------
    Array
        SPD matrix.
    """
    if seed is not None:
        np.random.seed(seed)
    eigvals = np.random.uniform(eig_min, eig_max, size=d)
    Q = np.diag(eigvals)
    U = ortho_group.rvs(dim=d)
    return U @ Q @ U.T


def generate_true_transport(
    d: int, eig_min: float = 0.1, eig_max: float = 3.0, seed: int | None = None
) -> Array:
    """
    Generate a true SPD transport matrix T0 for Gaussian OT regression.
    """
    return generate_spd_matrix(d, eig_min=eig_min, eig_max=eig_max, seed=seed)


def generate_noise_matrices(
    N: int, d: int, eig_min: float = 0.5, eig_max: float = 1.5
) -> List[Array]:
    """
    Generate N SPD noise matrices Q_i.

    In the regression model:  ν_i = Q_i # (T0 # μ_i)
    """
    return [generate_spd_matrix(d, eig_min, eig_max) for _ in range(N)]


def generate_input_matrices(N: int, d: int) -> List[Array]:
    """
    Generate N SPD predictor covariance matrices Σ_X,i.
    """
    return [generate_spd_matrix(d) for _ in range(N)]


def generate_output_matrices(
    T0: Array, Qs: List[Array], Ms: List[Array]
) -> List[Array]:
    """
    Generate N SPD response covariance matrices Σ_Y,i
    according to the regression model ν_i = Q_i # (T0 # μ_i).

    Parameters
    ----------
    T0 : Array
        True transport matrix.
    Qs : List[Array]
        Noise matrices Q_i.
    Ms : List[Array]
        Predictor covariances Σ_X,i.

    Returns
    -------
    List[Array]
        Response covariances Σ_Y,i.
    """
    Ns = []
    for Q, M in zip(Qs, Ms):
        Ns.append(Q @ T0 @ M @ T0 @ Q)
    return Ns


def generate_dataset(
    N: int, d: int
) -> Tuple[Array, List[Array], List[Array], List[Array]]:
    """
    Generate a complete Gaussian OT regression dataset.

    Returns
    -------
    T0 : Array
        True transport matrix.
    Qs : list of Array
        Noise matrices.
    Ms : list of Array
        Predictor covariances.
    Ns : list of Array
        Response covariances.
    """
    T0 = generate_true_transport(d)
    Qs = generate_noise_matrices(N, d)
    Ms = generate_input_matrices(N, d)
    Ns = generate_output_matrices(T0, Qs, Ms)
    return T0, Qs, Ms, Ns
