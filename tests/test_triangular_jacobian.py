from __future__ import annotations

import torch

from src.triangular_flow import HolomorphicTriangularFlow


def finite_difference_complex_jacobian(model, x, eps=1e-6):
    x = x.clone().detach()
    batch, dim = x.shape
    assert batch == 1
    base = model(x=x).z[0]
    cols = []
    for j in range(dim):
        xp = x.clone()
        xp[0, j] += eps
        yp = model(x=xp).z[0]
        cols.append((yp - base) / eps)
    return torch.stack(cols, dim=1)


def test_triangular_logdet_matches_finite_difference_det():
    torch.manual_seed(3)
    dim = 4
    model = HolomorphicTriangularFlow(
        dim=dim,
        num_layers=4,
        hidden_dim=12,
        depth=2,
        dtype=torch.float64,
        scale_clip=0.10,
        translation_scale=0.10,
    )
    x = torch.randn(1, dim, dtype=torch.float64) * 0.3
    out = model(x=x)
    jac = finite_difference_complex_jacobian(model, x, eps=1e-6)
    det_fd = torch.linalg.det(jac)
    det_tr = torch.exp(out.logdet[0])
    assert torch.allclose(det_fd, det_tr, rtol=5e-4, atol=5e-4), (det_fd, det_tr)


def test_inverse_roundtrip():
    torch.manual_seed(4)
    model = HolomorphicTriangularFlow(dim=6, num_layers=6, hidden_dim=16, dtype=torch.float64)
    x = torch.randn(5, 6, dtype=torch.float64)
    out = model(x=x)
    z_inv, inv_logdet = model.inverse(out.z)
    target = torch.complex(x, torch.zeros_like(x))
    assert torch.allclose(z_inv, target, rtol=1e-8, atol=1e-8)
    assert torch.allclose(inv_logdet, -out.logdet, rtol=1e-8, atol=1e-8)
