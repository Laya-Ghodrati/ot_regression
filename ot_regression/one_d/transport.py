import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d


def optimal_map(
    F: NDArray[np.float64], G: NDArray[np.float64], grid: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Computes OT map from CDF F to CDF G using inverse interpolation."""
    G_inv = interp1d(G, grid, bounds_error=False, fill_value=(grid[0], grid[-1]))
    return G_inv(F)
