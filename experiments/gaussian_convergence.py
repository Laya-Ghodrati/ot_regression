"""
Gaussian OT regression convergence (configure at bottom).
- SPD Q, M, T; N_i = Q T M T Q
- Vary d, N; repeat reps times
- Save per-d boxplots, a JSON of inputs, and error logs
- Optional parallel execution (set with params['parallel'] = True/False)
"""

from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ot_regression.gaussian.dca import fit_gaussian_dca
from ot_regression.gaussian.generate import (
    generate_input_matrices,
    generate_noise_matrices,
    generate_output_matrices,
    generate_true_transport,
)
from ot_regression.gaussian.metrics import frobenius_error


def run_single(
    d: int,
    N: int,
    seed: int,
    *,
    transport_kind: str,
    T_eig_min: float,
    T_eig_max: float,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> Tuple[float, str | None]:
    np.random.seed(seed)
    try:
        if transport_kind.lower() == "id":
            T_true = np.eye(d)
        elif transport_kind.lower() == "uniform":
            T_true = generate_true_transport(d, eig_min=T_eig_min, eig_max=T_eig_max)
        elif transport_kind.lower() == "folded_normal":
            eigs = 2.0 * np.abs(np.random.normal(0.0, 1.0, size=d)) + 0.3
            T_true = generate_true_transport(
                d, eig_min=float(np.min(eigs)), eig_max=float(np.max(eigs))
            )
        else:
            return float("nan"), f"Invalid transport_kind: {transport_kind}"

        Qs = generate_noise_matrices(N, d, eig_min=0.5, eig_max=1.5)
        Ms = generate_input_matrices(N, d)
        Ns = generate_output_matrices(T_true, Qs, Ms)

        T_hat, hist = fit_gaussian_dca(
            Ms, Ns, T_true=T_true, max_iter=max_iter, verbose=verbose
        )
        if not hist.get("error_true"):
            return float(frobenius_error(T_hat, T_true)), None
        return float(hist["error_true"][-1]), None
    except Exception as e:
        return float("nan"), f"[error] d={d}, N={N}, seed={seed}: {e}"


def save_boxplots(
    results: Dict[Tuple[int, int], List[float]],
    d_values: List[int],
    N_values: List[int],
    out_dir: Path,
    tag: str,
    dpi: int = 300,
) -> None:
    for d in d_values:
        vals = {N: results.get((d, N)) for N in N_values}
        vals = {k: v for k, v in vals.items() if v is not None}
        if not vals:
            continue
        labels = list(vals.keys())
        data = [np.log2(vals[N]) for N in labels]
        plt.figure(figsize=(9, 5))
        plt.boxplot(data, labels=labels)
        plt.title(f"log2 error vs N (d={d}, T={tag})")
        plt.xlabel("N")
        plt.ylabel("log2 ||T_hat - T_true||_F")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"boxplot_d{d}_{tag}.png", dpi=dpi)
        plt.close()


def validate_params(p: Dict) -> None:
    if not p["d_values"] or not p["N_values"]:
        raise ValueError("d_values and N_values must be non-empty")
    if p["reps"] <= 0:
        raise ValueError("reps must be > 0")
    if p["transport_kind"].lower() not in {"id", "uniform", "folded_normal"}:
        raise ValueError("transport_kind must be in {'id','uniform','folded_normal'}")


if __name__ == "__main__":
    cpu_count = os.cpu_count() or 4
    default_workers = min(max(1, cpu_count // 2), 4)

    params: Dict = {
        "d_values": [2, 3, 10, 20],
        "N_values": [4, 8, 16, 32, 64, 128],
        "reps": 50,
        "skip_d_gt": 3,
        "skip_N_gt": 128,
        "transport_kind": "uniform",
        "T_eig_min": 0.3,
        "T_eig_max": 3.0,
        "max_iter": 10,
        "tol": 1e-8,
        "verbose": False,
        "base_seed": 42,
        "output_root": "outputs",
        "tag": None,
        "dpi": 300,
        "parallel": True,  # Set this to False to run without parallelization
        "workers": default_workers,
        "blas_threads": 1,
    }

    validate_params(params)

    if params["blas_threads"]:
        os.environ.setdefault("OMP_NUM_THREADS", str(params["blas_threads"]))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(params["blas_threads"]))
        os.environ.setdefault("MKL_NUM_THREADS", str(params["blas_threads"]))

    tag = params["tag"] or params["transport_kind"]
    out_dir = (
        Path(params["output_root"])
        / f"gaussian_convergence_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "inputs_used.json").write_text(json.dumps(params, indent=2))

    tasks: List[Tuple[int, int, int]] = []
    seed_counter = 0
    for d in params["d_values"]:
        for N in params["N_values"]:
            if d > params["skip_d_gt"] and N > params["skip_N_gt"]:
                continue
            for _ in range(params["reps"]):
                tasks.append((d, N, params["base_seed"] + seed_counter))
                seed_counter += 1

    random.seed(params["base_seed"])
    random.shuffle(tasks)

    total_tasks = len(tasks)
    print(
        f"[start] total tasks: {total_tasks}; parallel={params['parallel']}; workers={params['workers']}"
    )

    collected: List[Tuple[int, int, float, str | None]] = []
    error_log: List[str] = []

    start_time = time.time()

    if params["parallel"] and params["workers"] > 0:
        with ProcessPoolExecutor(max_workers=params["workers"]) as ex:
            fut2meta = {
                ex.submit(
                    run_single,
                    d,
                    N,
                    seed,
                    transport_kind=params["transport_kind"],
                    T_eig_min=params["T_eig_min"],
                    T_eig_max=params["T_eig_max"],
                    max_iter=params["max_iter"],
                    tol=params["tol"],
                    verbose=params["verbose"],
                ): (d, N)
                for (d, N, seed) in tasks
            }

            for fut in tqdm(
                as_completed(fut2meta), total=total_tasks, desc="Simulations"
            ):
                d, N = fut2meta[fut]
                val, err = fut.result()
                collected.append((d, N, val, err))
                if err:
                    error_log.append(err)
    else:
        for d, N, seed in tqdm(tasks, total=total_tasks, desc="Simulations"):
            val, err = run_single(
                d,
                N,
                seed,
                transport_kind=params["transport_kind"],
                T_eig_min=params["T_eig_min"],
                T_eig_max=params["T_eig_max"],
                max_iter=params["max_iter"],
                tol=params["tol"],
                verbose=params["verbose"],
            )
            collected.append((d, N, val, err))
            if err:
                error_log.append(err)

    elapsed_time = time.time() - start_time

    results: Dict[Tuple[int, int], List[float]] = {
        (d, N): []
        for d in params["d_values"]
        for N in params["N_values"]
        if not (d > params["skip_d_gt"] and N > params["skip_N_gt"])
    }
    for d, N, val, _ in collected:
        results[(d, N)].append(val)

    np.savez(
        out_dir / "results_arrays.npz",
        d_values=np.array(params["d_values"]),
        N_values=np.array(params["N_values"]),
        reps=np.array(params["reps"]),
        results=results,
    )

    if error_log:
        (out_dir / "error_log.txt").write_text("\n".join(error_log))

    save_boxplots(
        results, params["d_values"], params["N_values"], out_dir, tag, dpi=params["dpi"]
    )

    print(f"Saved results and figures to: {out_dir.resolve()}")
    print(f"Total runtime: {elapsed_time:.2f} seconds")
