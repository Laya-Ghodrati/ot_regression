"""
Run Gaussian OT regression simulation with DCA and produce plots:
- Convergence curve(s)
- Vector fields
- 3D Gaussian surfaces for a few sample pairs
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.stats import multivariate_normal

from ot_regression.gaussian.generate import generate_dataset
from ot_regression.gaussian.dca import fit_gaussian_dca


def plot_vector_field(M: np.ndarray, title: str, subtract_identity: bool = True):
    """Plot a 2D vector field for a linear map M."""
    if M.shape != (2, 2):
        raise ValueError("Vector field plot is only implemented for d=2.")

    x, y = np.meshgrid(np.linspace(-10, 10, 15), np.linspace(-10, 10, 15))
    u = M[0, 0] * x + M[0, 1] * y
    v = M[1, 0] * x + M[1, 1] * y

    if subtract_identity:
        u -= x
        v -= y

    plt.figure()
    plt.quiver(x, y, u, v, color="black", angles="xy", scale_units="xy", scale=5)
    plt.title(title)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)


def gaussian_surface_grid(M: np.ndarray, grid_lim: float = 10, step: float = 0.1):
    """Return meshgrid and PDF values for N(0, M)."""
    mean = np.array([0.0, 0.0])
    x, y = np.mgrid[-grid_lim:grid_lim:step, -grid_lim:grid_lim:step]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean, M)
    z = rv.pdf(pos)
    return x, y, z


def plot_3d_gaussians(Ms, Ns, samples_num: int, out_path: Path):
    """Plot 3D Gaussian surfaces for first `samples_num` pairs."""
    fig, axs = plt.subplots(
        samples_num, 2, figsize=(10, samples_num * 4), subplot_kw={"projection": "3d"}
    )
    if samples_num == 1:
        axs = np.array([axs])  # ensure indexing consistency

    for i in range(samples_num):
        x, y, z1 = gaussian_surface_grid(Ms[i])
        _, _, z2 = gaussian_surface_grid(Ns[i])

        axs[i, 0].plot_surface(x, y, z1, cmap="cividis")
        axs[i, 0].set_title(f"$\\mu_{{{i+1}}}$", color="b", fontsize=14)
        axs[i, 0].set_xlim(-10, 10)
        axs[i, 0].set_ylim(-10, 10)

        axs[i, 1].plot_surface(x, y, z2, cmap="cividis")
        axs[i, 1].set_title(f"$\\nu_{{{i+1}}}$", color="r", fontsize=14)
        axs[i, 1].set_xlim(-10, 10)
        axs[i, 1].set_ylim(-10, 10)

    fig.tight_layout()
    fig.savefig(out_path / "gaussian_surfaces.png", dpi=300)


def main(output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    # (seed is set in __main__, so nothing here)

    # Experiment params
    N = 100
    d = 2
    max_iter = 10

    # Data
    T0, Qs, Ms, Ns = generate_dataset(N, d)

    # Fit (new signature returns history dict)
    T_hat, hist = fit_gaussian_dca(Ms, Ns, T_true=T0, max_iter=max_iter, tol=1e-8, verbose=False)

    # --- Convergence plots ---
    # 1) Parameter change ΔT
    plt.figure()
    plt.plot(range(1, len(hist["delta_T"]) + 1), hist["delta_T"], marker="o")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\Delta T = \|T_{k+1} - T_k\|_F$")
    plt.title("DCA Convergence (parameter change)")
    plt.grid(True)
    plt.savefig(output_path / "convergence_deltaT.png", dpi=300)

    # 2) Error to ground truth (if tracked)
    if hist["error_true"] is not None:
        plt.figure()
        plt.plot(range(1, len(hist["error_true"]) + 1), hist["error_true"], marker="o")
        plt.xlabel("Iteration")
        plt.ylabel(r"$\|\hat{T} - T_0\|_F$")
        plt.title("Error to Ground Truth")
        plt.grid(True)
        plt.savefig(output_path / "convergence_error_true.png", dpi=300)

    # --- Vector fields ---
    plot_vector_field(T0, r"$T_0 - \mathrm{id}$", subtract_identity=True)
    plt.savefig(output_path / "vector_field_T0.png", dpi=300)

    plot_vector_field(T_hat, r"$\hat{T} - \mathrm{id}$", subtract_identity=True)
    plt.savefig(output_path / "vector_field_That.png", dpi=300)

    plot_vector_field(T0 - T_hat, r"$T_0 - \hat{T}$", subtract_identity=False)
    plt.savefig(output_path / "vector_field_diff.png", dpi=300)

    # --- 3D Gaussian surfaces ---
    plot_3d_gaussians(Ms, Ns, samples_num=2, out_path=output_path)

    # Save numeric results
    np.savez(
        output_path / "results.npz",
        T0=T0,
        T_hat=T_hat,
        delta_T=np.array(hist["delta_T"]),
        error_true=(np.array(hist["error_true"]) if hist["error_true"] is not None else None),
        num_iter=np.array(hist["num_iter"]),
    )
    print(f"Results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    seed = 20
    np.random.seed(seed)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(f"outputs/gaussian_simulation_seed{seed}_{timestamp}")
    main(output_path)
