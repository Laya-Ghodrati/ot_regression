# ot_regression/one_d/simulate.py

import math
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta


def map_T(x: NDArray[np.float64], k: float = -2.0) -> NDArray[np.float64]:
    """Systematic transport map T0."""
    return x - np.sin(math.pi * k * x) / (abs(k) * math.pi)


def map_noise(
    X: NDArray[np.float64],
    K: List[int],
    L: int,
    intervals: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Apply piecewise sinusoidal maps with random slope parameter k over L intervals.
    """
    # ensure inputs lie in [intervals[0], intervals[-1]] (usually [0,1])
    X = np.clip(X, intervals[0], intervals[-1])

    Y = np.empty_like(X)
    for i in range(L):
        a, b = intervals[i], intervals[i + 1]
        # left-closed, right-open except last interval is right-closed
        if i < L - 1:
            mask = (X >= a) & (X < b)
        else:
            mask = (X >= a) & (X <= b)

        x_local = X[mask]
        if x_local.size == 0:
            continue

        scale = 2.0 / (b - a)
        shift = (a + b) / (b - a)

        x_scaled = x_local * scale - shift
        k = K[i]
        y_scaled = x_scaled - np.sin(np.pi * k * x_scaled) / (abs(k) * np.pi)
        y_rescaled = (y_scaled + shift) / scale
        Y[mask] = y_rescaled

    return Y


def rand_mixture_beta(
    n: int, alpha_params: NDArray[np.float64], beta_params: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Samples from a 3-component Beta mixture."""
    out = beta.rvs(alpha_params[0], beta_params[0], size=n // 2)
    out = np.concatenate(
        [
            out,
            beta.rvs(alpha_params[1], beta_params[1], size=n // 4),
            beta.rvs(alpha_params[2], beta_params[2], size=n // 4),
        ]
    )
    return out


def generate_pair(
    sample_per_dist: int, grid_size: int, rng: np.random.Generator | None = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generates one predictor sample, one response sample, and the true map on the m-point grid."""
    rng = rng or np.random.default_rng()

    alpha_params = rng.uniform(5, 10, 3)
    beta_params = rng.uniform(0.4, 0.6, 3)
    shift = rng.uniform(0, 1)

    rx = np.mod(
        rand_mixture_beta(sample_per_dist, alpha_params, beta_params) + shift, 1.0
    )
    rx2 = np.mod(
        rand_mixture_beta(sample_per_dist, alpha_params, beta_params) + shift, 1.0
    )

    rx = rx[(rx >= 0.0) & (rx <= 1.0)]
    rx2 = rx2[(rx2 >= 0.0) & (rx2 <= 1.0)]

    L = 5
    intervals = np.concatenate(([0.0], rng.uniform(0.0, 1.0, L - 1), [1.0]))
    intervals.sort()
    k_choices = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
    k_list = rng.choice(k_choices, size=L, replace=True)

    ry = map_noise(map_T(rx2), k_list, L, intervals)
    grid = np.linspace(0, 1, grid_size)
    pair_map = map_noise(map_T(grid), k_list, L, intervals)

    return rx, ry, pair_map


def generate_dataset(
    num_pairs: int, samples_per_dist: int, grid_size: int, seed: int | None = None
) -> Tuple[
    List[NDArray[np.float64]], List[NDArray[np.float64]], List[NDArray[np.float64]]
]:
    """Generates N synthetic (rx, ry, T_true) triples."""
    rng = np.random.default_rng(seed)
    rxs, rys, pair_maps = [], [], []
    for _ in range(num_pairs):
        rx, ry, pair_map = generate_pair(samples_per_dist, grid_size, rng)
        rxs.append(rx)
        rys.append(ry)
        pair_maps.append(pair_map)
    return rxs, rys, pair_maps
