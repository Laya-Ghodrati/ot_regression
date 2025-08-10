import matplotlib.pyplot as plt
import numpy as np

from ot_regression.one_d.isotonic import (compute_residual_maps,
                                          fit_isotonic_transport)
from ot_regression.one_d.kde import sample_to_cdf, sample_to_pdf
from ot_regression.one_d.simulate import generate_dataset, map_T
from ot_regression.one_d.transport import optimal_map

# Parameters
num_pairs = 100  # number of (predictor, response) distribution pairs
samples_per_dist = 1000  # number of samples per distribution
grid_size = 100  # number of evaluation points
bandwidth = samples_per_dist ** (-1 / 5)  # KDE bandwidth
grid = np.linspace(0, 1, grid_size)

# Generate synthetic dataset
predictors, responses, pair_maps = generate_dataset(
    num_pairs, samples_per_dist, grid_size, seed=42
)

# Convert samples to CDFs and compute OT maps
predictor_cdfs, response_cdfs, ot_maps = [], [], []
for x_samples, y_samples in zip(predictors, responses):
    F = sample_to_cdf(x_samples, grid, bandwidth)
    G = sample_to_cdf(y_samples, grid, bandwidth)
    T = optimal_map(F, G, grid)
    predictor_cdfs.append(F)
    response_cdfs.append(G)
    ot_maps.append(T)

# Fit regression map
T_hat = fit_isotonic_transport(predictor_cdfs, ot_maps, grid)

# Compute residual maps
residual_maps = compute_residual_maps(T_hat, predictor_cdfs, response_cdfs, grid)

# ====== Plots ======

# For a few pairs, show PDFs and maps side-by-side
pairs_to_plot = min(3, num_pairs)  # adjust as needed
for i in range(pairs_to_plot):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: PDFs of predictor and response
    pred_pdf = sample_to_pdf(predictors[i], grid, bandwidth)
    resp_pdf = sample_to_pdf(responses[i], grid, bandwidth)
    axes[0].plot(grid, pred_pdf, color="red", lw=2, label="Predictor PDF")
    axes[0].plot(grid, resp_pdf, color="blue", lw=2, label="Response PDF")
    axes[0].set_title(f"Pair {i+1} — PDFs")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True)

    # Right: Pair-level optimal map vs. true pair_map
    axes[1].plot(grid, pair_maps[i], color="green", lw=2, label="True pair_map")
    axes[1].plot(
        grid, ot_maps[i], "--", color="orange", lw=1.5, label="Estimated optimal map"
    )
    axes[1].set_title(f"Pair {i+1} — Maps")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("T(x)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

# Estimated regression map vs true map
plt.figure(figsize=(6, 4))
plt.plot(grid, T_hat, label=r"Estimated $T_0$", lw=2)
plt.plot(grid, map_T(grid), label=r"True $T_0$", lw=1, color="black")
plt.xlabel("x")
plt.ylabel("T(x)")
plt.title("Estimated Map vs True Map")
plt.legend()
plt.grid(alpha=0.3)

# Residual maps
plt.figure(figsize=(6, 4))
for T_eps in residual_maps:
    plt.plot(grid, T_eps, color="C0", lw=0.3)
plt.title("Residual Maps")
plt.xlabel("x")
plt.ylabel(r"$T_{\varepsilon}(x)$")
plt.grid(alpha=0.3)

plt.show()
