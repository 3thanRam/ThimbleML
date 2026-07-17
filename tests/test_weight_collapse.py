from __future__ import annotations

import torch

from src.estimator import contour_loss, importance_weight_diagnostics
from src.train import checkpoint_score, collapse_reasons


def test_constant_weights_have_full_effective_sample_size() -> None:
    logw = torch.zeros(8, dtype=torch.complex128)

    diagnostics = importance_weight_diagnostics(logw)

    assert torch.allclose(diagnostics["ess"], torch.tensor(8.0, dtype=torch.float64))
    assert torch.allclose(diagnostics["ess_fraction"], torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(diagnostics["max_weight_fraction"], torch.tensor(0.125, dtype=torch.float64))
    assert torch.allclose(diagnostics["weight_perplexity_fraction"], torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(diagnostics["real_logw_range"], torch.tensor(0.0, dtype=torch.float64))


def test_single_dominant_weight_is_detected() -> None:
    logw = torch.tensor([0.0] + [-100.0] * 7, dtype=torch.complex128)

    diagnostics = importance_weight_diagnostics(logw)
    reasons = collapse_reasons(
        {name: float(value.item()) for name, value in diagnostics.items()},
        min_ess_fraction=0.05,
        max_weight_fraction=0.50,
    )

    assert torch.allclose(diagnostics["ess"], torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(diagnostics["ess_fraction"], torch.tensor(0.125, dtype=torch.float64))
    assert diagnostics["max_weight_fraction"] > 0.999
    assert reasons == ["maximum weight fraction 1 exceeds 0.5"]


def test_ess_penalty_increases_collapsed_loss() -> None:
    logw = torch.tensor([0.0] + [-100.0] * 7, dtype=torch.complex128)
    z = torch.zeros(8, 2, dtype=torch.complex128)

    baseline, _ = contour_loss(logw, z, ess_penalty=0.0)
    penalized, _ = contour_loss(logw, z, ess_penalty=0.1)

    assert penalized > baseline


def test_checkpoint_score_rejects_high_phase_weight_collapse() -> None:
    collapsed = checkpoint_score({"avg_phase": 1.0, "ess_fraction": 0.001})
    healthy = checkpoint_score({"avg_phase": 0.05, "ess_fraction": 0.6})

    assert healthy > collapsed
