from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from .benchmarks import ComplexPhi4Benchmark, ComplexPhi4Config
from .data_loading import ProposalConfig, make_loaders, make_proposal_npz
from .estimator import complex_log_weights, contour_loss, estimate_observables
from .triangular_flow import HolomorphicTriangularFlow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a holomorphic triangular neural thimble flow.")
    p.add_argument("--data", type=str, default="data/proposal_samples.npz")
    p.add_argument("--run-dir", type=str, default="runs/triangular_phi4")
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--valid-batches", type=int, default=10)
    p.add_argument("--proposal-scale", type=float, default=1.6)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--activation", type=str, choices=["poly", "tanh", "sin"], default="poly")
    p.add_argument(
        "--scale-factor",
        type=float,
        default=0.01,
        help="Multiplicative factor applied to the holomorphic log-scale head.",
    )
    p.add_argument(
        "--scale-clip",
        dest="scale_factor",
        type=float,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--translation-scale", type=float, default=0.02)
    p.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float64")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--grid-points", type=int, default=401)
    p.add_argument("--grid-limit", type=float, default=5.0)
    p.add_argument("--make-data", action="store_true", help="Regenerate proposal data before training.")
    p.add_argument("--train-samples", type=int, default=80000)
    p.add_argument("--valid-samples", type=int, default=20000)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-triangular-linear", action="store_true", help="Disable learned triangular linear mixing layers.")
    p.add_argument("--plot-contour", action="store_true", help="Save a 2D contour projection plot at the end.")
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def tensor_is_finite(value: torch.Tensor) -> bool:
    if value.is_complex():
        return bool(torch.isfinite(value.real).all() and torch.isfinite(value.imag).all())
    return bool(torch.isfinite(value).all())


def require_finite(name: str, value: torch.Tensor, step: int) -> None:
    if not tensor_is_finite(value):
        raise FloatingPointError(f"Non-finite {name} at training step {step}.")


def require_finite_parameters(model: torch.nn.Module, step: int) -> None:
    for name, parameter in model.named_parameters():
        require_finite(f"parameter {name}", parameter.detach(), step)


@torch.no_grad()
def validate(model, benchmark, valid_loader, proposal_scale, device, max_batches: int) -> dict[str, float]:
    model.eval()
    logws = []
    zs = []
    for i, (x,) in enumerate(valid_loader):
        if i >= max_batches:
            break
        x = x.to(device)
        logw, z, _ = complex_log_weights(model, benchmark, x, proposal_scale)
        if not tensor_is_finite(logw) or not tensor_is_finite(z):
            raise FloatingPointError("Non-finite values encountered during validation.")
        logws.append(logw.detach().cpu())
        zs.append(z.detach().cpu())
    if not logws:
        raise ValueError("Validation produced no batches; increase valid samples or valid-batches.")
    logw_all = torch.cat(logws, dim=0)
    z_all = torch.cat(zs, dim=0)
    result = estimate_observables(logw_all, z_all, benchmark)
    for name, value in result.items():
        if isinstance(value, complex):
            finite = math.isfinite(value.real) and math.isfinite(value.imag)
        else:
            finite = math.isfinite(float(value))
        if not finite:
            raise FloatingPointError(f"Non-finite validation metric {name}.")
    return {k: float(v) if isinstance(v, (int, float)) else v for k, v in result.items()}


@torch.no_grad()
def maybe_plot_contour(model, run_dir: Path, device: torch.device, dtype: torch.dtype, dim: int) -> None:
    if dim != 2:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    model.eval()
    xs = torch.linspace(-3.0, 3.0, 41, dtype=dtype)
    lines = []
    for fixed_axis in [0, 1]:
        for val in torch.linspace(-3.0, 3.0, 13, dtype=dtype):
            pts = torch.zeros(41, 2, dtype=dtype)
            pts[:, fixed_axis] = val
            pts[:, 1 - fixed_axis] = xs
            z = model(x=pts.to(device)).z.cpu()
            lines.append(z)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    for z in lines:
        ax.plot(z[:, 0].real.numpy(), z[:, 0].imag.numpy(), alpha=0.45)
    ax.set_xlabel("Re z0")
    ax.set_ylabel("Im z0")
    ax.set_title("Learned contour projection for coordinate z0")
    fig.tight_layout()
    fig.savefig(run_dir / "contour_projection_z0.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    data_path = Path(args.data)
    if args.make_data or not data_path.exists():
        make_proposal_npz(
            data_path,
            ProposalConfig(
                dim=args.dim,
                proposal_scale=args.proposal_scale,
                train_samples=args.train_samples,
                valid_samples=args.valid_samples,
                seed=args.seed,
            ),
        )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_loader, valid_loader, proposal_scale = make_loaders(data_path, args.batch_size)

    cfg = ComplexPhi4Config(dim=args.dim, grid_points=args.grid_points, grid_limit=args.grid_limit)
    benchmark = ComplexPhi4Benchmark(cfg)
    exact = benchmark.exact_grid(grid_points=args.grid_points, grid_limit=args.grid_limit, dtype=torch.float64) if args.dim <= 3 else {}

    model = HolomorphicTriangularFlow(
        dim=args.dim,
        num_layers=args.layers,
        hidden_dim=args.hidden,
        depth=args.depth,
        activation=args.activation,
        dtype=dtype,
        scale_factor=args.scale_factor,
        translation_scale=args.translation_scale,
        use_triangular_linear=not args.no_triangular_linear,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    config_json = vars(args).copy()
    config_json["device_resolved"] = str(device)
    config_json["exact_grid"] = {k: [v.real, v.imag] for k, v in exact.items()}
    (run_dir / "config.json").write_text(json.dumps(config_json, indent=2))

    metrics_path = run_dir / "metrics.csv"
    train_iter = iter(train_loader)
    for step in range(1, args.steps + 1):
        model.train()
        try:
            (x,) = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            (x,) = next(train_iter)
        x = x.to(device=device, dtype=dtype)
        opt.zero_grad(set_to_none=True)
        logw, z, logdet = complex_log_weights(model, benchmark, x, proposal_scale)
        require_finite("log weights", logw, step)
        require_finite("contour values", z, step)
        require_finite("log determinant", logdet, step)

        loss, train_metrics = contour_loss(logw, z)
        require_finite("loss", loss, step)
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=10.0, error_if_nonfinite=True)
        if not math.isfinite(float(grad_norm)):
            raise FloatingPointError(f"Non-finite gradient norm at training step {step}.")
        opt.step()
        require_finite_parameters(model, step)

        if step == 1 or step % max(1, min(100, args.steps // 10)) == 0 or step == args.steps:
            val = validate(model, benchmark, valid_loader, proposal_scale, device, args.valid_batches)
            row: dict[str, Any] = {"step": step, "grad_norm": float(grad_norm)}
            row.update({k: float(v.item()) for k, v in train_metrics.items()})
            row.update({f"valid_{k}": v for k, v in val.items()})
            if exact:
                for name in ["z_mean", "z2_mean", "radius2"]:
                    er = exact[name]
                    vr = complex(val[f"{name}_real"], val[f"{name}_imag"])
                    row[f"exact_{name}_real"] = er.real
                    row[f"exact_{name}_imag"] = er.imag
                    row[f"abs_err_{name}"] = abs(vr - er)
                row["exact_avg_phase_original_grid"] = exact["average_phase_original_grid"].real
            if not all(math.isfinite(float(v)) for v in row.values()):
                raise FloatingPointError(f"Non-finite logged metric at training step {step}.")
            append_csv(metrics_path, row)
            print(json.dumps(row, indent=2))

    torch.save({"model": model.state_dict(), "args": vars(args)}, run_dir / "model.pt")
    if args.plot_contour:
        maybe_plot_contour(model, run_dir, device, dtype, args.dim)
    print(f"Saved run outputs to {run_dir}")


if __name__ == "__main__":
    main()
