# OT Regression: Simulations & Experiments (1D & Gaussian)

This repository contains code for the simulation studies from the papers:
- **[Distribution-on-distribution regression with optimal transport (1D case)](https://academic.oup.com/biomet/article-abstract/109/4/957/6515608)** by Laya Ghodrati and Victor M. Panaretos
- **[Transportation of measure regression in higher dimensions (Gaussian case)](https://arxiv.org/abs/2305.17503)** by Laya Ghodrati and Victor M. Panaretos


The main package code is in the `ot_regression` subfolder, with:
- `one_d/` -- 1D simulations and utilities
- `gaussian/`-- Gaussian simulations, metrics, and DCA solver

All experiment scripts from the papers are in the `experiments/` folder.

---

## Installation

```bash
# (optional) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install runtime dependencies
pip install -r requirements.txt

# install in editable mode
pip install -e .
```

For development (tests, linters):
```bash
pip install -r requirements-dev.txt
```

## Running experiments

Example: Gaussian simulation
```bash
python -m experiments.gaussian_simulation
```

Results will be saved in an output folder named:
```bash
outputs/gaussian_simulation_seed{seed}_{timestamp}
```

## Running tests

```bash
pytest -v
```

Tests are provided for the main modules in ot_regression.

## Reproducibility

- Experiments accept a `--seed` argument or set seed in the script.
- Output folders include both the seed and a timestamp.
- Parallel runs randomize task order deterministically from the seed.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ghodrati2022distribution,
  title={Distribution-on-distribution regression via optimal transport maps},
  author={Ghodrati, Laya and Panaretos, Victor M},
  journal={Biometrika},
  volume={109},
  number={4},
  pages={957--974},
  year={2022},
  publisher={Oxford University Press}
}
```

```bibtex
@article{ghodrati2023transportation,
  title={Transportation of measure regression in higher dimensions},
  author={Ghodrati, Laya and Panaretos, Victor M},
  journal={arXiv preprint arXiv:2305.17503},
  year={2023}
}
```