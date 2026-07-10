from __future__ import annotations

import torch

from src.benchmarks import ComplexPhi4Benchmark, ComplexPhi4Config


def test_exact_grid_runs_for_dim2():
    bench = ComplexPhi4Benchmark(ComplexPhi4Config(dim=2, grid_points=31, grid_limit=4.0))
    exact = bench.exact_grid(dtype=torch.float64)
    assert "Z" in exact
    assert abs(exact["Z"]) > 0
    assert "z2_mean" in exact
