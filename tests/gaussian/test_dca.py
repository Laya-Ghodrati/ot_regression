import numpy as np

from ot_regression.gaussian.dca import fit_gaussian_dca
from ot_regression.gaussian.generate import (
    generate_input_matrices,
    generate_noise_matrices,
    generate_output_matrices,
)
from ot_regression.gaussian.metrics import frobenius_error, is_spd


def test_fit_gaussian_dca_recovers_identity_when_T_true_is_I_and_Q_is_I():
    # Easiest case: T_true = I, Q_i = I. We should recover ~I.
    rng = np.random.default_rng(0)
    d, N = 3, 8
    T_true = np.eye(d)
    Qs = [np.eye(d) for _ in range(N)]
    Ms = generate_input_matrices(N, d)
    Ns = generate_output_matrices(T_true, Qs, Ms)

    T_hat, hist = fit_gaussian_dca(
        Ms, Ns, T_true=T_true, max_iter=12, tol=1e-10, verbose=False
    )

    # Convergence diagnostics present
    assert "delta_T" in hist and "error_true" in hist
    assert len(hist["delta_T"]) >= 1
    # Final error should be tiny
    assert frobenius_error(T_hat, T_true) < 1e-6
    # Still SPD
    assert is_spd(T_hat)


def test_fit_gaussian_dca_runs_without_T_true_and_stops_by_parameter_change():
    d, N = 2, 5
    T_true = np.eye(d)  # only for generating Ns below
    # Add realistic Q noise to avoid trivial case
    Qs = generate_noise_matrices(N, d, eig_min=0.8, eig_max=1.2)
    Ms = generate_input_matrices(N, d)
    Ns = generate_output_matrices(T_true, Qs, Ms)

    T_hat, hist = fit_gaussian_dca(Ms, Ns, max_iter=10, tol=1e-8, verbose=False)

    # Error-to-true is not tracked
    assert hist["error_true"] is None
    # We computed at least one delta and it's finite
    assert len(hist["delta_T"]) >= 1
    assert np.isfinite(hist["delta_T"][-1])
    assert is_spd(T_hat)
