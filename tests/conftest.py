# tests/conftest.py
import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure repo root is importable for pytest
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

@pytest.fixture(scope="session")
def grid_size() -> int:
    return 200

@pytest.fixture(scope="session")
def grid(grid_size: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, grid_size)

@pytest.fixture(scope="session")
def bandwidth() -> float:
    # numeric bandwidth like original paper code: n^(-1/5)
    n = 1000
    return n ** (-1/5)
