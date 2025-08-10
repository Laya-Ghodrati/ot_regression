import numpy as np
from ot_regression.one_d.kde import sample_to_pdf, sample_to_cdf

def test_pdf_and_cdf_shapes(grid, bandwidth):
    rng = np.random.default_rng(0)
    samples = rng.uniform(0, 1, 500)
    pdf = sample_to_pdf(samples, grid, bandwidth)
    cdf = sample_to_cdf(samples, grid, bandwidth)
    assert pdf.shape == grid.shape
    assert cdf.shape == grid.shape

def test_cdf_monotone_and_normalized(grid, bandwidth):
    rng = np.random.default_rng(1)
    samples = rng.beta(5, 2, 400)
    cdf = sample_to_cdf(samples, grid, bandwidth)
    assert np.all(np.diff(cdf) >= -1e-12), "CDF must be non-decreasing."
    assert np.isclose(cdf[0], cdf.min(), atol=1e-8)
    assert np.isclose(cdf[-1], 1.0, atol=1e-6), "CDF must end at ~1."
