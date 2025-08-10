import numpy as np
from ot_regression.gaussian.ot_gaussian import gaussian_ot_map
from ot_regression.gaussian.generate import generate_spd_matrix

def test_gaussian_ot_map_pushes_src_to_tgt():
    rng = np.random.default_rng(1)
    d = 3
    Sigma_src = generate_spd_matrix(d, eig_min=0.5, eig_max=2.0, seed=123)
    Sigma_tgt = generate_spd_matrix(d, eig_min=0.7, eig_max=3.0, seed=456)

    A = gaussian_ot_map(Sigma_src, Sigma_tgt)
    pushed = A @ Sigma_src @ A.T

    # The defining property: A Σ_src A^T ≈ Σ_tgt
    assert np.allclose(pushed, Sigma_tgt, atol=1e-6, rtol=1e-6)

def test_gaussian_ot_map_identity_when_equal():
    d = 4
    Sigma = generate_spd_matrix(d, eig_min=0.8, eig_max=1.2, seed=7)
    A = gaussian_ot_map(Sigma, Sigma)
    assert np.allclose(A, np.eye(d), atol=1e-8)
