from __future__ import annotations

import math

import torch

from .benchmarks import ComplexPhi4Benchmark
from .triangular_flow import HolomorphicTriangularFlow


def log_q_normal(x: torch.Tensor, scale: float) -> torch.Tensor:
    dim = x.shape[-1]
    return -0.5 * (x / scale).pow(2).sum(dim=-1) - dim * math.log(scale) - 0.5 * dim * math.log(2.0 * math.pi)


def complex_log_weights(
    model: HolomorphicTriangularFlow,
    benchmark: ComplexPhi4Benchmark,
    x: torch.Tensor,
    proposal_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = model(x=x)
    logq = log_q_normal(x.to(out.z.real.dtype), proposal_scale).to(out.z.dtype)
    logw = -benchmark.action(out.z) + out.logdet - logq
    return logw, out.z, out.logdet


def safe_centered_weights(logw: torch.Tensor) -> torch.Tensor:
    shift = logw.real.max().detach()
    return torch.exp(logw - shift)


def average_phase_from_logw(logw: torch.Tensor) -> torch.Tensor:
    w = safe_centered_weights(logw)
    return w.mean().abs() / w.abs().mean().clamp_min(torch.finfo(w.real.dtype).tiny)


def effective_sample_size_from_logw(logw: torch.Tensor) -> torch.Tensor:
    w_abs = safe_centered_weights(logw).abs()
    return w_abs.sum().pow(2) / w_abs.pow(2).sum().clamp_min(torch.finfo(w_abs.dtype).tiny)


def contour_loss(logw: torch.Tensor, z: torch.Tensor, imag_penalty: float = 1e-4) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    w = safe_centered_weights(logw)
    avg_phase = w.mean().abs() / w.abs().mean().clamp_min(torch.finfo(w.real.dtype).tiny)
    real_var = logw.real.var(unbiased=False)
    # Unit phase concentration is a direct phase-alignment term. It is not the
    # only possible objective, but it is simple and stable for this benchmark.
    phase_unit = torch.exp(1j * logw.imag)
    phase_concentration = phase_unit.mean().abs().clamp_min(1e-12)
    loss = -torch.log(avg_phase.clamp_min(1e-12)) + 0.02 * real_var - 0.10 * torch.log(phase_concentration)
    if imag_penalty > 0:
        loss = loss + imag_penalty * z.imag.pow(2).mean()
    metrics = {
        "loss": loss.detach(),
        "avg_phase": avg_phase.detach(),
        "real_logw_var": real_var.detach(),
        "unit_phase_concentration": phase_concentration.detach(),
        "mean_abs_imag_z": z.imag.abs().mean().detach(),
    }
    return loss.real, metrics


@torch.no_grad()
def estimate_observables(logw: torch.Tensor, z: torch.Tensor, benchmark: ComplexPhi4Benchmark) -> dict[str, complex | float]:
    w = safe_centered_weights(logw)
    Z_shifted = w.mean()
    out: dict[str, complex | float] = {
        "Z_shifted_real": float(Z_shifted.real.item()),
        "Z_shifted_imag": float(Z_shifted.imag.item()),
        "avg_phase": float((w.mean().abs() / w.abs().mean().clamp_min(torch.finfo(w.real.dtype).tiny)).item()),
        "ess": float((w.abs().sum().pow(2) / w.abs().pow(2).sum().clamp_min(torch.finfo(w.real.dtype).tiny)).item()),
    }
    obs = benchmark.observables(z)
    denom = w.sum()
    for name, value in obs.items():
        est = (w * value).sum() / denom
        out[f"{name}_real"] = float(est.real.item())
        out[f"{name}_imag"] = float(est.imag.item())
    return out
