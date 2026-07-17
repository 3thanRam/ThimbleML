from __future__ import annotations

import pytest
import torch

from src.train import require_finite, tensor_is_finite
from src.triangular_flow import HolomorphicAffineCoupling


def test_complex_finiteness_checks_real_and_imaginary_parts() -> None:
    finite = torch.tensor([1.0 + 2.0j], dtype=torch.complex128)
    nonfinite_real = torch.tensor([complex(float("nan"), 0.0)], dtype=torch.complex128)
    nonfinite_imag = torch.tensor([complex(0.0, float("inf"))], dtype=torch.complex128)

    assert tensor_is_finite(finite)
    assert not tensor_is_finite(nonfinite_real)
    assert not tensor_is_finite(nonfinite_imag)


def test_require_finite_fails_with_step_context() -> None:
    with pytest.raises(FloatingPointError, match="training step 12"):
        require_finite("loss", torch.tensor(float("nan")), step=12)


def test_scale_head_avoids_complex_tanh_pole() -> None:
    layer = HolomorphicAffineCoupling(
        dim=2,
        mask=torch.tensor([True, False]),
        hidden_dim=4,
        depth=1,
        dtype=torch.float64,
        scale_factor=0.01,
        translation_scale=0.02,
    )
    z = torch.tensor([[0.0 + 1.5707j, 0.5 + 0.0j]], dtype=torch.complex128)

    out, logdet, layer_logdet = layer(z)

    assert tensor_is_finite(out)
    assert tensor_is_finite(logdet)
    assert tensor_is_finite(layer_logdet)
