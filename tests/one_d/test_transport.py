import numpy as np
from ot_regression.one_d.simulate import generate_dataset
from ot_regression.one_d.kde import sample_to_cdf, sample_to_pdf
from ot_regression.one_d.transport import optimal_map
import matplotlib.pyplot as plt


def test_optimal_map_identity():
    m = 101
    grid = np.linspace(0.0, 1.0, m)
    F = np.linspace(0.0, 1.0, m)
    G = np.linspace(0.0, 1.0, m)
    T = optimal_map(F, G, grid)
    assert np.allclose(T, grid, atol=1e-12)



def test_per_pair_consistency(grid):
    """
    For each pair, OT(F,G) should be close to the simulator's pair_map.
    We check median MSE across pairs to allow a few noisy cases.
    """
    num_pairs = 12
    n = 15000
    bw = n ** (-1/5)

    Xs, Ys, pair_maps = generate_dataset(num_pairs, n, len(grid), seed=2024)

    mses = []
    # plot_indices = [0, 1, 2]  # which pairs to visualize
    plot_indices = []

    for idx, (x, y, pair_map) in enumerate(zip(Xs, Ys, pair_maps)):
        F = sample_to_cdf(x, grid, bw)
        G = sample_to_cdf(y, grid, bw)
        T_hat = np.clip(optimal_map(F, G, grid), grid[0], grid[-1])
        mse = float(np.mean((T_hat - pair_map) ** 2))
        mses.append(mse)

        if idx in plot_indices:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # Left: PDFs
           # Left: PDFs
            pdf_x = sample_to_pdf(x, grid, bw)  # pass grid here
            pdf_y = sample_to_pdf(y, grid, bw)
            axes[0].plot(grid, pdf_x, label="PDF of X", color="red")
            axes[0].plot(grid, pdf_y, label="PDF of Y", color="blue")

            axes[0].set_title(f"Pair {idx} — PDFs")
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("Density")
            axes[0].legend()
            axes[0].grid(True)

            # Right: Transport maps
            axes[1].plot(grid, pair_map, label="Simulator's pair_map", lw=2)
            axes[1].plot(grid, T_hat, "--", label="OT(F, G)", lw=1.5)
            axes[1].set_title(f"Transport maps — MSE={mse:.2e}")
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("T(x)")
            axes[1].legend()
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()

    med = float(np.median(mses))
    assert med < 3e-3, f"Median per-pair OT MSE too high: {med:.3f}"
