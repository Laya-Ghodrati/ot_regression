import numpy as np
from ot_regression.gaussian.metrics import is_spd, frobenius_error

def test_is_spd_true_and_false():
    # SPD (orthogonal * diag+ * orthogonal^T)
    rng = np.random.default_rng(0)
    Q = np.diag(rng.uniform(0.5, 2.0, size=3))
    U, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    A = U @ Q @ U.T
    assert is_spd(A)

    # Not symmetric
    B = np.array([[1.0, 2.0], [0.0, 1.0]])
    assert not is_spd(B)

    # Symmetric but not PD
    C = np.array([[1.0, 0.0], [0.0, -1.0]])
    assert not is_spd(C)

def test_frobenius_error_basic():
    A = np.array([[1.0, 2.0],[3.0, 4.0]])
    B = np.array([[1.0, 0.0],[0.0, 0.0]])
    # ||A-B||_F = sqrt(0^2 + 2^2 + 3^2 + 4^2) = sqrt(29)
    assert np.isclose(frobenius_error(A, B), np.sqrt(29.0))
