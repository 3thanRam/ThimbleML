from __future__ import annotations

import torch

from src.estimator import average_phase_from_logw, effective_sample_size_from_logw
from src.triangular_flow import HolomorphicTriangularFlow


def test_zero_layer_flow_is_exact_identity() -> None:
    model = HolomorphicTriangularFlow(dim=3, num_layers=0, dtype=torch.float64)
    x = torch.tensor([[0.2, -0.5, 1.1], [-0.3, 0.4, 0.7]], dtype=torch.float64)

    out = model(x=x)
    expected = torch.complex(x, torch.zeros_like(x))

    assert torch.equal(out.z, expected)
    assert torch.equal(out.logdet, torch.zeros(2, dtype=torch.complex128))
    assert out.layer_logdets == []


def test_constant_logweights_have_unit_phase_and_maximum_ess() -> None:
    logw = torch.zeros(8, dtype=torch.complex128)

    phase = average_phase_from_logw(logw)
    ess = effective_sample_size_from_logw(logw)

    assert torch.allclose(phase, torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(ess, torch.tensor(8.0, dtype=torch.float64))
