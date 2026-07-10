from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ComplexPhi4Config:
    dim: int = 2
    m2: float = 1.0
    lam: float = 0.10
    coupling: float = 0.20
    mu: float = 2.2
    source: float = 0.15
    grid_limit: float = 5.0
    grid_points: int = 401


class ComplexPhi4Benchmark:
    """Small holomorphic sign-problem benchmark.

    S(z) = sum_i [0.5 m^2 z_i^2 + lambda z_i^4 + i mu z_i]
           + 0.5 kappa sum_i (z_i - z_{i+1})^2
           + source * prod_i z_i / dim

    The polynomial action is holomorphic. For dim <= 3 the exact integral over
    the original real contour is computed by grid quadrature and used only for
    validation, not supervised training.
    """

    def __init__(self, config: ComplexPhi4Config):
        self.config = config

    @property
    def dim(self) -> int:
        return self.config.dim

    def action(self, z: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        onsite = 0.5 * cfg.m2 * z.pow(2) + cfg.lam * z.pow(4) + (1j * cfg.mu) * z
        s = onsite.sum(dim=-1)
        if cfg.coupling != 0.0 and z.shape[-1] > 1:
            diffs = z - torch.roll(z, shifts=-1, dims=-1)
            s = s + 0.5 * cfg.coupling * diffs.pow(2).sum(dim=-1)
        if cfg.source != 0.0 and z.shape[-1] > 1:
            s = s + cfg.source * z.prod(dim=-1) / float(z.shape[-1])
        return s

    def observables(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "z_mean": z.mean(dim=-1),
            "z2_mean": z.pow(2).mean(dim=-1),
            "radius2": z.pow(2).sum(dim=-1),
        }

    @torch.no_grad()
    def exact_grid(self, grid_points: int | None = None, grid_limit: float | None = None, dtype: torch.dtype = torch.float64) -> dict[str, complex]:
        cfg = self.config
        dim = cfg.dim
        if dim > 3:
            raise ValueError("exact_grid is intended for dim <= 3; use MC baselines for larger dim")
        n = int(grid_points or cfg.grid_points)
        limit = float(grid_limit or cfg.grid_limit)
        device = torch.device("cpu")
        xs_1d = torch.linspace(-limit, limit, n, dtype=dtype, device=device)
        dx = (2.0 * limit) / float(n - 1)
        meshes = torch.meshgrid(*([xs_1d] * dim), indexing="ij")
        x = torch.stack([m.reshape(-1) for m in meshes], dim=-1)
        z = torch.complex(x, torch.zeros_like(x))
        weights = torch.exp(-self.action(z))
        cell = dx ** dim
        Z = weights.sum() * cell
        obs = self.observables(z)
        out: dict[str, complex] = {"Z": complex(Z.item())}
        for name, value in obs.items():
            out[name] = complex(((value * weights).sum() * cell / Z).item())
        phase = weights / weights.abs().clamp_min(torch.finfo(dtype).tiny)
        avg_phase = weights.sum().abs() / weights.abs().sum().clamp_min(torch.finfo(dtype).tiny)
        out["average_phase_original_grid"] = complex(avg_phase.item())
        out["mean_unit_phase_original_grid"] = complex(phase.mean().item())
        return out
