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


def importance_weight_diagnostics(logw: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return scale-invariant diagnostics for absolute importance weights.

    The diagnostics use ``abs(exp(logw - max(real(logw))))``. Centering prevents
    overflow and does not change normalized weights, ESS, entropy, or perplexity.
    """
    if logw.numel() == 0:
        raise ValueError("logw must contain at least one value")

    log_abs = logw.real.reshape(-1)
    w_abs = safe_centered_weights(logw.reshape(-1)).abs()
    tiny = torch.finfo(w_abs.dtype).tiny
    total = w_abs.sum().clamp_min(tiny)
    probabilities = w_abs / total
    ess = probabilities.square().sum().clamp_min(tiny).reciprocal()
    sample_count = probabilities.new_tensor(float(probabilities.numel()))
    entropy = -(probabilities * probabilities.clamp_min(tiny).log()).sum()
    perplexity = entropy.exp()
    if probabilities.numel() > 1:
        normalized_entropy = entropy / math.log(probabilities.numel())
    else:
        normalized_entropy = probabilities.new_tensor(1.0)

    quantiles = torch.quantile(
        log_abs.detach(),
        log_abs.new_tensor([0.01, 0.50, 0.99]),
    )
    return {
        "ess": ess,
        "ess_fraction": ess / sample_count,
        "max_weight_fraction": probabilities.max(),
        "weight_entropy": entropy,
        "normalized_weight_entropy": normalized_entropy,
        "weight_perplexity": perplexity,
        "weight_perplexity_fraction": perplexity / sample_count,
        "real_logw_min": log_abs.min(),
        "real_logw_q01": quantiles[0],
        "real_logw_median": quantiles[1],
        "real_logw_q99": quantiles[2],
        "real_logw_max": log_abs.max(),
        "real_logw_range": log_abs.max() - log_abs.min(),
    }


def average_phase_from_logw(logw: torch.Tensor) -> torch.Tensor:
    w = safe_centered_weights(logw)
    return w.mean().abs() / w.abs().mean().clamp_min(torch.finfo(w.real.dtype).tiny)


def effective_sample_size_from_logw(logw: torch.Tensor) -> torch.Tensor:
    return importance_weight_diagnostics(logw)["ess"]


def contour_loss(
    logw: torch.Tensor,
    z: torch.Tensor,
    imag_penalty: float = 1e-4,
    ess_penalty: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if ess_penalty < 0:
        raise ValueError("ess_penalty must be non-negative")

    w = safe_centered_weights(logw)
    avg_phase = w.mean().abs() / w.abs().mean().clamp_min(torch.finfo(w.real.dtype).tiny)
    real_var = logw.real.var(unbiased=False)
    weight_diagnostics = importance_weight_diagnostics(logw)
    # Unit phase concentration is a direct phase-alignment term. It is not the
    # only possible objective, but it is simple and stable for this benchmark.
    phase_unit = torch.exp(1j * logw.imag)
    phase_concentration = phase_unit.mean().abs().clamp_min(1e-12)
    loss = -torch.log(avg_phase.clamp_min(1e-12)) + 0.02 * real_var - 0.10 * torch.log(phase_concentration)
    if ess_penalty > 0:
        loss = loss - ess_penalty * torch.log(weight_diagnostics["ess_fraction"].clamp_min(1e-12))
    if imag_penalty > 0:
        loss = loss + imag_penalty * z.imag.pow(2).mean()
    metrics = {
        "loss": loss.detach(),
        "avg_phase": avg_phase.detach(),
        "real_logw_var": real_var.detach(),
        "unit_phase_concentration": phase_concentration.detach(),
        "mean_abs_imag_z": z.imag.abs().mean().detach(),
        "ess": weight_diagnostics["ess"].detach(),
        "ess_fraction": weight_diagnostics["ess_fraction"].detach(),
        "max_weight_fraction": weight_diagnostics["max_weight_fraction"].detach(),
        "weight_perplexity_fraction": weight_diagnostics["weight_perplexity_fraction"].detach(),
        "real_logw_range": weight_diagnostics["real_logw_range"].detach(),
    }
    return loss.real, metrics


@torch.no_grad()
def estimate_observables(logw: torch.Tensor, z: torch.Tensor, benchmark: ComplexPhi4Benchmark) -> dict[str, complex | float]:
    w = safe_centered_weights(logw)
    Z_shifted = w.mean()
    weight_diagnostics = importance_weight_diagnostics(logw)
    out: dict[str, complex | float] = {
        "sample_count": float(logw.numel()),
        "Z_shifted_real": float(Z_shifted.real.item()),
        "Z_shifted_imag": float(Z_shifted.imag.item()),
        "avg_phase": float((w.mean().abs() / w.abs().mean().clamp_min(torch.finfo(w.real.dtype).tiny)).item()),
    }
    out.update({name: float(value.item()) for name, value in weight_diagnostics.items()})
    obs = benchmark.observables(z)
    denom = w.sum()
    for name, value in obs.items():
        est = (w * value).sum() / denom
        out[f"{name}_real"] = float(est.real.item())
        out[f"{name}_imag"] = float(est.imag.item())
    return out
