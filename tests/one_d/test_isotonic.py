import numpy as np

from ot_regression.one_d.isotonic import fit_isotonic_transport
from ot_regression.one_d.kde import sample_to_cdf
from ot_regression.one_d.simulate import generate_dataset, map_T
from ot_regression.one_d.transport import optimal_map


def test_T0_estimation_accuracy(grid):
    # Stronger settings for stability
    num_pairs = 100
    n = 10000
    bw = n ** (-1 / 5)

    Xs, Ys, _ = generate_dataset(num_pairs, n, len(grid), seed=42)

    Fs, Ts = [], []
    for x, y in zip(Xs, Ys):
        F = sample_to_cdf(x, grid, bw)
        G = sample_to_cdf(y, grid, bw)
        T = np.clip(optimal_map(F, G, grid), grid[0], grid[-1])
        Fs.append(F)
        Ts.append(T)

    T_hat = fit_isotonic_transport(Fs, Ts, grid)
    # robustify
    T_hat = np.maximum.accumulate(T_hat)
    T_hat = np.clip(T_hat, grid[0], grid[-1])

    mse = float(np.mean((T_hat - map_T(grid)) ** 2))
    assert mse < 2e-2, f"T0 MSE too high: {mse:.4f}"
