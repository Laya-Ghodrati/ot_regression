# ot_regression/one_d/isotonic.py

from typing import List, Optional
import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import interp1d
from ot_regression.one_d.transport import optimal_map

from sklearn.isotonic import IsotonicRegression
from typing import List, Optional
import numpy as np
from numpy.typing import NDArray

from typing import List, Optional
import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression

def fit_isotonic_transport(
    Fs: List[NDArray[np.float64]],
    Ts: List[NDArray[np.float64]],
    grid: NDArray[np.float64],
    weights: Optional[List[NDArray[np.float64]]] = None,
) -> NDArray[np.float64]:
    N = len(Fs)
    m = grid.size

    if weights is None:
        weights = [np.diff(F, prepend=0.0) for F in Fs]

    # Make arrays shape (N, m): rows = pairs, cols = grid
    T_mat = np.vstack(Ts)                 # (N, m)
    W_mat = np.vstack(weights)            # (N, m)

    # Flatten column-wise so x1’s N values come first, then x2’s N, etc.
    y = np.ravel(T_mat, order="F")        # length N*m
    w = np.ravel(W_mat, order="F")        # length N*m
    X = np.repeat(grid, N)                # length N*m, aligned with y,w

    reg = IsotonicRegression(out_of_bounds="clip").fit(X, y, sample_weight=w)
    T_hat = reg.predict(grid)

    # Project to a clean monotone map in [grid[0], grid[-1]]
    T_hat = np.maximum.accumulate(T_hat)
    T_hat = np.clip(T_hat, grid[0], grid[-1])
    return T_hat



def compute_residual_maps(
    T_hat: NDArray[np.float64],
    Fs: List[NDArray[np.float64]],
    Gs: List[NDArray[np.float64]],
    grid: NDArray[np.float64],
) -> List[NDArray[np.float64]]:
    """Computes residual OT maps for each sample."""
    T_inv = interp1d(T_hat, grid, bounds_error=False, fill_value=(grid[0], grid[-1]))
    residuals = []
    for F, G in zip(Fs, Gs):
        Gtilde = interp1d(grid, F, bounds_error=False, fill_value=(0.0, 1.0))(T_inv(grid))
        T_eps = optimal_map(Gtilde, G, grid)
        residuals.append(T_eps)
    return residuals
