import numpy as np

from ot_regression.one_d.simulate import generate_dataset, generate_pair, map_T


def test_map_T_monotone(grid):
    diffs = np.diff(map_T(grid))
    assert np.all(diffs >= -1e-8), "map_T must be non-decreasing."


def test_generate_pair_shapes(grid_size):
    rx, ry, pair_map = generate_pair(sample_per_dist=300, grid_size=grid_size)
    assert rx.ndim == 1 and ry.ndim == 1
    assert pair_map.shape == (grid_size,), "pair_map must align with the grid size."


def test_generate_dataset_shapes(grid_size):
    Xs, Ys, pair_maps = generate_dataset(
        num_pairs=3, samples_per_dist=300, grid_size=grid_size, seed=0
    )
    assert len(Xs) == len(Ys) == len(pair_maps) == 3
    assert pair_maps[0].shape == (grid_size,), "pair_map shape mismatch."
