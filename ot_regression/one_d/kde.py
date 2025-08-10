from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import gaussian_kde


def _get_kde(samples: NDArray[np.float64], bw: float | str) -> gaussian_kde:
    """Creates a Gaussian KDE object with given bandwidth."""
    return gaussian_kde(samples, bw_method=bw)


def sample_to_pdf(
    samples: NDArray[np.float64],
    grid: NDArray[np.float64],
    bw: float | Literal["scott", "silverman"],
) -> NDArray[np.float64]:
    """Estimates PDF on a grid from samples."""
    kde = _get_kde(samples, bw)
    return kde(grid)


def sample_to_cdf(
    samples: NDArray[np.float64],
    grid: NDArray[np.float64],
    bw: float | Literal["scott", "silverman"],
) -> NDArray[np.float64]:
    """Estimates CDF on a grid from samples."""
    pdf = sample_to_pdf(samples, grid, bw)
    cdf = np.cumsum(pdf)
    return cdf / cdf[-1]
