import numpy as np

from ot_regression.gaussian.generate import (generate_dataset,
                                             generate_input_matrices,
                                             generate_noise_matrices,
                                             generate_output_matrices,
                                             generate_spd_matrix,
                                             generate_true_transport)
from ot_regression.gaussian.metrics import is_spd


def test_generate_spd_matrix_properties_and_seed():
    d = 5
    A1 = generate_spd_matrix(d, eig_min=0.3, eig_max=1.5, seed=123)
    A2 = generate_spd_matrix(d, eig_min=0.3, eig_max=1.5, seed=123)
    A3 = generate_spd_matrix(d, eig_min=0.3, eig_max=1.5, seed=124)

    assert is_spd(A1)
    assert is_spd(A2)
    assert not np.allclose(A1, A3)  # different seed → different draw
    assert np.allclose(A1, A2)  # same seed  → same draw


def test_generate_true_transport_spd():
    T0 = generate_true_transport(3, eig_min=0.4, eig_max=2.0, seed=999)
    assert is_spd(T0)


def test_generate_noise_and_inputs_lengths():
    N, d = 7, 3
    Qs = generate_noise_matrices(N, d, eig_min=0.8, eig_max=1.2)
    Ms = generate_input_matrices(N, d)
    assert len(Qs) == N and len(Ms) == N
    assert all(is_spd(Q) for Q in Qs)
    assert all(is_spd(M) for M in Ms)


def test_generate_output_matrices_relation():
    # With Q = I, Ns should equal T0 M T0
    d, N = 3, 4
    T0 = generate_true_transport(d, eig_min=0.5, eig_max=2.0, seed=1)
    Qs = [np.eye(d) for _ in range(N)]
    Ms = generate_input_matrices(N, d)
    Ns = generate_output_matrices(T0, Qs, Ms)
    for M, Nmat in zip(Ms, Ns):
        assert np.allclose(Nmat, T0 @ M @ T0, atol=1e-10)


def test_generate_dataset_shapes_and_spd():
    N, d = 6, 2
    T0, Qs, Ms, Ns = generate_dataset(N, d)
    assert T0.shape == (d, d)
    assert len(Qs) == len(Ms) == len(Ns) == N
    assert all(Q.shape == (d, d) for Q in Qs)
    assert all(M.shape == (d, d) for M in Ms)
    assert all(Nm.shape == (d, d) for Nm in Ns)
    assert all(is_spd(Q) for Q in Qs)
    assert all(is_spd(M) for M in Ms)
    assert all(is_spd(Nm) for Nm in Ns)
