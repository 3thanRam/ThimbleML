from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

ComplexDType = torch.dtype


def complex_dtype_from_float(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float64:
        return torch.complex128
    if dtype == torch.float32:
        return torch.complex64
    raise TypeError(f"Unsupported real dtype {dtype}; use float32 or float64.")


class ComplexLinear(nn.Module):
    """Holomorphic affine layer y = x W^T + b with complex parameters.

    The layer never uses conjugates, magnitudes, real/imag splitting, normalization,
    or non-analytic gates. It is therefore complex analytic in its input.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        init_scale: float | None = None,
    ):
        super().__init__()
        ctype = complex_dtype_from_float(dtype)
        if init_scale is None:
            init_scale = 1.0 / math.sqrt(max(1, in_features))
        w_re = torch.randn(out_features, in_features, dtype=dtype) * init_scale
        w_im = torch.randn(out_features, in_features, dtype=dtype) * init_scale
        self.weight = nn.Parameter(torch.complex(w_re, w_im).to(ctype))
        if bias:
            b_re = torch.zeros(out_features, dtype=dtype)
            b_im = torch.zeros(out_features, dtype=dtype)
            self.bias = nn.Parameter(torch.complex(b_re, b_im).to(ctype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight.transpose(-1, -2)
        if self.bias is not None:
            y = y + self.bias
        return y


class HolomorphicActivation(nn.Module):
    """Small catalogue of elementwise holomorphic activations.

    Notes:
    - tanh is meromorphic globally, but holomorphic away from its poles. In practice
      it is common and stable for bounded arguments. The polynomial option is entire.
    - Do not replace this with abs, ReLU on real/imag, sigmoid(|z|), LayerNorm,
      softmax, RMSNorm, etc.; those break holomorphicity.
    """

    def __init__(self, kind: Literal["poly", "tanh", "sin"] = "poly"):
        super().__init__()
        self.kind = kind

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.kind == "poly":
            return z + 0.15 * z.pow(3)
        if self.kind == "tanh":
            return torch.tanh(z)
        if self.kind == "sin":
            return torch.sin(z)
        raise ValueError(f"Unknown holomorphic activation {self.kind}")


class HolomorphicMLP(nn.Module):
    """Complex MLP used inside triangular conditioners.

    It is holomorphic because it is a composition of complex affine maps and an
    elementwise holomorphic activation.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_dim: int,
        depth: int = 2,
        activation: Literal["poly", "tanh", "sin"] = "poly",
        dtype: torch.dtype = torch.float32,
        final_scale: float = 1e-3,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = []
        if depth == 1:
            layers.append(ComplexLinear(dim_in, dim_out, dtype=dtype, init_scale=final_scale))
        else:
            layers.append(ComplexLinear(dim_in, hidden_dim, dtype=dtype))
            layers.append(HolomorphicActivation(activation))
            for _ in range(depth - 2):
                layers.append(ComplexLinear(hidden_dim, hidden_dim, dtype=dtype))
                layers.append(HolomorphicActivation(activation))
            out = ComplexLinear(hidden_dim, dim_out, dtype=dtype, init_scale=final_scale)
            # Make the initial map close to identity when used in coupling layers.
            with torch.no_grad():
                out.weight.mul_(final_scale)
                if out.bias is not None:
                    out.bias.zero_()
            layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
