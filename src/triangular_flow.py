from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from .complex_layers import HolomorphicMLP, complex_dtype_from_float


@dataclass
class FlowOutput:
    z: torch.Tensor
    logdet: torch.Tensor
    layer_logdets: list[torch.Tensor]


def alternating_binary_mask(dim: int, parity: int, device: torch.device | None = None) -> torch.Tensor:
    idx = torch.arange(dim, device=device)
    return ((idx + parity) % 2 == 0)


class HolomorphicAffineCoupling(nn.Module):
    """Triangular holomorphic affine coupling layer.

    z_a' = z_a
    z_b' = z_b * exp(s(z_a)) + t(z_a)

    The conditioner receives z_a embedded back into the full dimension with zeros
    on inactive entries. Because s and t depend only on pass-through variables,
    the Jacobian is block triangular and

        log det J = sum_{b entries} s_b(z_a)

    exactly. This includes the complex determinant phase.
    """

    def __init__(
        self,
        dim: int,
        mask: torch.Tensor,
        hidden_dim: int = 64,
        depth: int = 2,
        activation: Literal["poly", "tanh", "sin"] = "poly",
        dtype: torch.dtype = torch.float32,
        scale_clip: float = 0.35,
        translation_scale: float = 0.25,
    ):
        super().__init__()
        if mask.numel() != dim:
            raise ValueError("mask must have shape [dim]")
        self.dim = dim
        self.register_buffer("mask_bool", mask.bool(), persistent=False)
        mask_f = mask.to(dtype=dtype).view(1, dim)
        self.register_buffer("mask", mask_f, persistent=False)
        self.register_buffer("inv_mask", 1.0 - mask_f, persistent=False)
        self.scale_clip = float(scale_clip)
        self.translation_scale = float(translation_scale)
        self.conditioner = HolomorphicMLP(
            dim_in=dim,
            dim_out=2 * dim,
            hidden_dim=hidden_dim,
            depth=depth,
            activation=activation,
            dtype=dtype,
            final_scale=1e-2,
        )

    def _condition(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask.to(dtype=z.real.dtype, device=z.device).to(z.dtype)
        inv_mask = self.inv_mask.to(dtype=z.real.dtype, device=z.device).to(z.dtype)
        h = self.conditioner(z * mask)
        raw_s, raw_t = h.chunk(2, dim=-1)
        # tanh is holomorphic; multiplying by a real scalar preserves holomorphicity.
        s = self.scale_clip * torch.tanh(raw_s) * inv_mask
        t = self.translation_scale * raw_t * inv_mask
        return s, t

    def forward(self, z: torch.Tensor, logdet: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z.ndim != 2 or z.shape[-1] != self.dim:
            raise ValueError(f"Expected z with shape [batch, {self.dim}], got {tuple(z.shape)}")
        if logdet is None:
            logdet = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        mask = self.mask.to(dtype=z.real.dtype, device=z.device).to(z.dtype)
        inv_mask = self.inv_mask.to(dtype=z.real.dtype, device=z.device).to(z.dtype)
        s, t = self._condition(z)
        z_next = z * mask + inv_mask * (z * torch.exp(s) + t)
        layer_logdet = s.sum(dim=-1)
        return z_next, logdet + layer_logdet, layer_logdet

    def inverse(self, z: torch.Tensor, logdet: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if logdet is None:
            logdet = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        mask = self.mask.to(dtype=z.real.dtype, device=z.device).to(z.dtype)
        inv_mask = self.inv_mask.to(dtype=z.real.dtype, device=z.device).to(z.dtype)
        s, t = self._condition(z)
        z_prev = z * mask + inv_mask * ((z - t) * torch.exp(-s))
        return z_prev, logdet - s.sum(dim=-1)



class ComplexTriangularLinear(nn.Module):
    """Learned complex triangular linear map with exact determinant.

    y = W z + b, where W is lower or upper triangular. The complex log
    determinant is exactly sum(log_diag). The layer is initialized as identity.
    This is the literal "series of triangular matrices" component; coupling
    layers add nonlinear holomorphic expressivity.
    """

    def __init__(self, dim: int, orientation: Literal["lower", "upper"] = "lower", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dim = dim
        self.orientation = orientation
        ctype = complex_dtype_from_float(dtype)
        self.strict = nn.Parameter(torch.zeros(dim, dim, dtype=ctype))
        self.log_diag = nn.Parameter(torch.zeros(dim, dtype=ctype))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=ctype))
        if orientation not in {"lower", "upper"}:
            raise ValueError("orientation must be 'lower' or 'upper'")
        if orientation == "lower":
            mask = torch.tril(torch.ones(dim, dim, dtype=torch.bool), diagonal=-1)
        else:
            mask = torch.triu(torch.ones(dim, dim, dtype=torch.bool), diagonal=1)
        self.register_buffer("strict_mask", mask, persistent=False)

    def matrix(self) -> torch.Tensor:
        strict = self.strict * self.strict_mask.to(self.strict.device)
        return strict + torch.diag(torch.exp(self.log_diag))

    def forward(self, z: torch.Tensor, logdet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        W = self.matrix()
        y = z @ W.transpose(-1, -2) + self.bias
        return y, logdet + self.log_diag.sum() * torch.ones_like(logdet)

    def inverse(self, z: torch.Tensor, logdet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        W = self.matrix()
        rhs = (z - self.bias).transpose(0, 1)
        x = torch.linalg.solve_triangular(W, rhs, upper=(self.orientation == "upper")).transpose(0, 1)
        return x, logdet - self.log_diag.sum() * torch.ones_like(logdet)

class ComplexPermutation(nn.Module):
    """Holomorphic coordinate permutation with determinant sign tracked exactly."""

    def __init__(self, perm: torch.Tensor):
        super().__init__()
        if perm.ndim != 1:
            raise ValueError("perm must be a vector")
        self.register_buffer("perm", perm.long(), persistent=False)
        inv = torch.empty_like(perm.long())
        inv[perm.long()] = torch.arange(perm.numel(), device=perm.device)
        self.register_buffer("inv_perm", inv, persistent=False)
        self.sign = self._permutation_sign(perm.long().cpu()).item()

    @staticmethod
    def _permutation_sign(perm: torch.Tensor) -> torch.Tensor:
        inv_count = 0
        p = perm.tolist()
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                inv_count += int(p[i] > p[j])
        return torch.tensor(1.0 if inv_count % 2 == 0 else -1.0)

    def forward(self, z: torch.Tensor, logdet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_next = z[:, self.perm.to(z.device)]
        if self.sign > 0:
            return z_next, logdet
        # log(-1) = i*pi; determinant phase matters.
        return z_next, logdet + (1j * torch.pi) * torch.ones_like(logdet)

    def inverse(self, z: torch.Tensor, logdet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_prev = z[:, self.inv_perm.to(z.device)]
        if self.sign > 0:
            return z_prev, logdet
        return z_prev, logdet - (1j * torch.pi) * torch.ones_like(logdet)


class HolomorphicTriangularFlow(nn.Module):
    """Stack of triangular holomorphic coupling layers.

    The flow maps the real proposal point x to z = f(x + 0i). It returns the
    complex log determinant det(df/dz) evaluated along x + 0i.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 8,
        hidden_dim: int = 64,
        depth: int = 2,
        activation: Literal["poly", "tanh", "sin"] = "poly",
        dtype: torch.dtype = torch.float32,
        scale_clip: float = 0.35,
        translation_scale: float = 0.25,
        use_permutations: bool = False,
        use_triangular_linear: bool = True,
    ):
        super().__init__()
        if dim < 2:
            raise ValueError("dim must be >= 2 for coupling layers")
        self.dim = dim
        self.dtype = dtype
        self.ctype = complex_dtype_from_float(dtype)
        modules: list[nn.Module] = []
        for layer in range(num_layers):
            mask = alternating_binary_mask(dim, parity=layer % 2)
            modules.append(
                HolomorphicAffineCoupling(
                    dim=dim,
                    mask=mask,
                    hidden_dim=hidden_dim,
                    depth=depth,
                    activation=activation,
                    dtype=dtype,
                    scale_clip=scale_clip,
                    translation_scale=translation_scale,
                )
            )
            if use_triangular_linear:
                modules.append(ComplexTriangularLinear(dim, orientation="lower" if layer % 2 == 0 else "upper", dtype=dtype))
            if use_permutations and layer != num_layers - 1:
                modules.append(ComplexPermutation(torch.arange(dim - 1, -1, -1)))
        self.layers = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor | None = None, z: torch.Tensor | None = None) -> FlowOutput:
        if z is None:
            if x is None:
                raise ValueError("Provide x or z")
            if x.ndim != 2 or x.shape[-1] != self.dim:
                raise ValueError(f"Expected x with shape [batch, {self.dim}], got {tuple(x.shape)}")
            z = torch.complex(x.to(self.dtype), torch.zeros_like(x.to(self.dtype))).to(self.ctype)
        else:
            z = z.to(self.ctype)
        logdet = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        layer_logdets: list[torch.Tensor] = []
        for layer in self.layers:
            if isinstance(layer, HolomorphicAffineCoupling):
                z, logdet, ld = layer(z, logdet)
                layer_logdets.append(ld)
            elif isinstance(layer, ComplexTriangularLinear):
                z, logdet = layer(z, logdet)
            elif isinstance(layer, ComplexPermutation):
                z, logdet = layer(z, logdet)
            else:
                raise TypeError(f"Unsupported layer type {type(layer)}")
        return FlowOutput(z=z, logdet=logdet, layer_logdets=layer_logdets)

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = z.to(self.ctype)
        logdet = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for layer in reversed(self.layers):
            if isinstance(layer, HolomorphicAffineCoupling):
                z, logdet = layer.inverse(z, logdet)
            elif isinstance(layer, ComplexTriangularLinear):
                z, logdet = layer.inverse(z, logdet)
            elif isinstance(layer, ComplexPermutation):
                z, logdet = layer.inverse(z, logdet)
            else:
                raise TypeError(f"Unsupported layer type {type(layer)}")
        return z, logdet
