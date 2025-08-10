from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ot_regression.one_d.isotonic import compute_residual_maps, fit_isotonic_transport
from ot_regression.one_d.kde import sample_to_cdf, sample_to_pdf
from ot_regression.one_d.simulate import generate_dataset, map_T
from ot_regression.one_d.transport import optimal_map


def main(output_dir: Path, seed: int):
    # Parameters
    num_pairs = 100
    samples_per_dist = 1000
    grid_size = 100
    bandwidth = samples_per_dist ** (-1 / 5)
    grid = np.linspace(0, 1, grid_size)

    # Generate synthetic dataset
    predictors, responses, pair_maps = generate_dataset(
        num_pairs, samples_per_dist, grid_size, seed=seed
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
    pairs_to_plot = min(3, num_pairs)

    for i in range(pairs_to_plot):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Left: PDFs
        pred_pdf = sample_to_pdf(predictors[i], grid, bandwidth)
        resp_pdf = sample_to_pdf(responses[i], grid, bandwidth)
        axes[0].plot(grid, pred_pdf, color="red", lw=2, label="Predictor PDF")
        axes[0].plot(grid, resp_pdf, color="blue", lw=2, label="Response PDF")
        axes[0].set_title(f"Pair {i+1} — PDFs")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("Density")
        axes[0].legend()
        axes[0].grid(True)

        # Right: maps
        axes[1].plot(grid, pair_maps[i], color="green", lw=2, label="True pair_map")
        axes[1].plot(
            grid,
            ot_maps[i],
            "--",
            color="orange",
            lw=1.5,
            label="Estimated optimal map",
        )
        axes[1].set_title(f"Pair {i+1} — Maps")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("T(x)")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        fig.savefig(output_dir / f"pair_{i+1}_pdfs_and_maps.png", dpi=300)
        plt.close(fig)

    # Estimated regression map vs true map
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, T_hat, label=r"Estimated $T_0$", lw=2)
    ax.plot(grid, map_T(grid), label=r"True $T_0$", lw=1, color="black")
    ax.set_xlabel("x")
    ax.set_ylabel("T(x)")
    ax.set_title("Estimated Map vs True Map")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(output_dir / "estimated_vs_true_map.png", dpi=300)
    plt.close(fig)

    # Residual maps
    fig, ax = plt.subplots(figsize=(6, 4))
    for T_eps in residual_maps:
        ax.plot(grid, T_eps, color="C0", lw=0.3)
    ax.plot(grid, grid, color="red", lw=2, label="Identity map")
    ax.set_title("Residual Maps")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$T_{\varepsilon}(x)$")
    ax.grid(alpha=0.3)
    fig.savefig(output_dir / "residual_maps.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(f"outputs/one_d_simulation_seed{seed}_{timestamp}")
    output_path.mkdir(parents=True, exist_ok=True)

    main(output_path, seed)
