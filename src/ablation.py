from __future__ import annotations

import csv
import json
import math
import shlex
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


LOCKED_SEEDS: tuple[int, ...] = (7, 17, 27)
LOCKED_SETTINGS: dict[str, str] = {
    "dim": "2",
    "steps": "800",
    "batch_size": "512",
    "valid_batches": "10",
    "proposal_scale": "1.6",
    "learning_rate": "2e-4",
    "layers": "4",
    "hidden": "32",
    "depth": "2",
    "activation": "poly",
    "scale_factor": "0.01",
    "translation_scale": "0.02",
    "dtype": "float64",
    "grid_points": "401",
    "grid_limit": "5.0",
    "min_valid_ess_fraction": "0.01",
    "max_valid_weight_fraction": "0.50",
    "collapse_patience": "1",
    "ess_penalty": "0",
}

REPORT_METRICS: tuple[str, ...] = (
    "valid_avg_phase",
    "valid_ess",
    "valid_ess_fraction",
    "valid_max_weight_fraction",
    "valid_weight_perplexity_fraction",
    "valid_real_logw_range",
    "abs_err_z_mean",
    "abs_err_z2_mean",
    "abs_err_radius2",
    "exact_avg_phase_original_grid",
)


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    scale_factor: float
    use_triangular_linear: bool
    description: str


ARCHITECTURES: dict[str, ArchitectureSpec] = {
    "additive": ArchitectureSpec(
        name="additive",
        scale_factor=0.0,
        use_triangular_linear=False,
        description="additive coupling without triangular-linear mixing",
    ),
    "affine": ArchitectureSpec(
        name="affine",
        scale_factor=0.01,
        use_triangular_linear=False,
        description="affine coupling without triangular-linear mixing",
    ),
    "full": ArchitectureSpec(
        name="full",
        scale_factor=0.01,
        use_triangular_linear=True,
        description="affine coupling with triangular-linear mixing",
    ),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def architecture_spec(name: str) -> ArchitectureSpec:
    try:
        return ARCHITECTURES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown architecture {name!r}; choose from {tuple(ARCHITECTURES)}") from exc


def build_data_command(python: str, repo_root: Path, data_path: Path) -> list[str]:
    return [
        python,
        str(repo_root / "scripts" / "make_data.py"),
        "--out",
        str(data_path),
        "--dim",
        LOCKED_SETTINGS["dim"],
        "--proposal-scale",
        LOCKED_SETTINGS["proposal_scale"],
        "--train-samples",
        "80000",
        "--valid-samples",
        "20000",
        "--seed",
        "1234",
    ]


def build_training_command(
    python: str,
    repo_root: Path,
    data_path: Path,
    run_dir: Path,
    architecture: str,
    seed: int,
    device: str = "auto",
) -> list[str]:
    spec = architecture_spec(architecture)
    command = [
        python,
        str(repo_root / "main.py"),
        "--data",
        str(data_path),
        "--run-dir",
        str(run_dir),
        "--dim",
        LOCKED_SETTINGS["dim"],
        "--steps",
        LOCKED_SETTINGS["steps"],
        "--batch-size",
        LOCKED_SETTINGS["batch_size"],
        "--valid-batches",
        LOCKED_SETTINGS["valid_batches"],
        "--proposal-scale",
        LOCKED_SETTINGS["proposal_scale"],
        "--lr",
        LOCKED_SETTINGS["learning_rate"],
        "--layers",
        LOCKED_SETTINGS["layers"],
        "--hidden",
        LOCKED_SETTINGS["hidden"],
        "--depth",
        LOCKED_SETTINGS["depth"],
        "--activation",
        LOCKED_SETTINGS["activation"],
        "--scale-factor",
        str(spec.scale_factor),
        "--translation-scale",
        LOCKED_SETTINGS["translation_scale"],
        "--dtype",
        LOCKED_SETTINGS["dtype"],
        "--seed",
        str(seed),
        "--grid-points",
        LOCKED_SETTINGS["grid_points"],
        "--grid-limit",
        LOCKED_SETTINGS["grid_limit"],
        "--min-valid-ess-fraction",
        LOCKED_SETTINGS["min_valid_ess_fraction"],
        "--max-valid-weight-fraction",
        LOCKED_SETTINGS["max_valid_weight_fraction"],
        "--collapse-patience",
        LOCKED_SETTINGS["collapse_patience"],
        "--ess-penalty",
        LOCKED_SETTINGS["ess_penalty"],
        "--device",
        device,
    ]
    if not spec.use_triangular_linear:
        command.append("--no-triangular-linear")
    return command


def command_string(command: Sequence[str]) -> str:
    return shlex.join(list(command))


def run_directory(run_root: Path, architecture: str, seed: int) -> Path:
    architecture_spec(architecture)
    return run_root / f"{architecture}-seed{seed}"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_metrics(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def _float_value(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_value(value: Any) -> int | None:
    parsed = _float_value(value)
    return int(parsed) if parsed is not None else None


def _row_collapsed(row: dict[str, str]) -> bool:
    return bool(_int_value(row.get("validation_collapsed")) or 0)


def select_best_row(rows: Iterable[dict[str, str]]) -> dict[str, str] | None:
    candidates: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        score = _float_value(row.get("checkpoint_score"))
        if score is not None and not _row_collapsed(row):
            candidates.append((score, row))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def classify_failure(status: dict[str, Any], run_exists: bool) -> str:
    state = str(status.get("status", "")).lower()
    if state == "completed":
        return "none"
    if not run_exists:
        return "missing"
    if not status:
        return "incomplete"
    text = f"{status.get('error_type', '')} {status.get('error', '')}".lower()
    if "importance-weight collapse" in text:
        return "collapse"
    if "non-finite" in text or "floatingpoint" in text:
        return "nonfinite"
    return "other"


def collect_run_record(run_dir: Path, architecture: str, seed: int) -> dict[str, Any]:
    architecture_spec(architecture)
    status = _load_json(run_dir / "status.json")
    manifest = _load_json(run_dir / "run_manifest.json")
    rows = _read_metrics(run_dir / "metrics.csv")
    selected = select_best_row(rows)
    completed = str(status.get("status", "")).lower() == "completed"
    final = rows[-1] if completed and rows else None
    last_row = rows[-1] if rows else None

    record: dict[str, Any] = {
        "architecture": architecture,
        "seed": seed,
        "status": status.get("status", "missing" if not run_dir.exists() else "incomplete"),
        "failure_kind": classify_failure(status, run_dir.exists()),
        "error_type": status.get("error_type", ""),
        "error": status.get("error", ""),
        "last_step": status.get("last_step", _int_value(last_row.get("step")) if last_row else None),
        "selected_step": _int_value(selected.get("step")) if selected else None,
        "selected_checkpoint_score": _float_value(selected.get("checkpoint_score")) if selected else None,
        "final_step": _int_value(final.get("step")) if final else None,
        "final_checkpoint_score": _float_value(final.get("checkpoint_score")) if final else None,
        "parameter_count": manifest.get("parameter_count"),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "return_code": manifest.get("return_code"),
        "git_commit": manifest.get("git_commit", ""),
        "command": manifest.get("command_shell", ""),
        "run_dir": str(run_dir),
    }
    for metric in REPORT_METRICS:
        record[f"selected_{metric}"] = _float_value(selected.get(metric)) if selected else None
        record[f"final_{metric}"] = _float_value(final.get(metric)) if final else None
    return record


def _finite_values(records: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _float_value(record.get(key))
        if value is not None:
            values.append(value)
    return values


def _mean_and_std(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else None
    return mean, std


def aggregate_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        subset = [record for record in records if record.get("architecture") == architecture]
        if not subset:
            continue
        completed = [record for record in subset if record.get("status") == "completed"]
        row: dict[str, Any] = {
            "architecture": architecture,
            "planned_runs": len(subset),
            "completed_runs": len(completed),
            "failed_runs": sum(record.get("status") == "failed" for record in subset),
            "collapsed_runs": sum(record.get("failure_kind") == "collapse" for record in subset),
            "nonfinite_runs": sum(record.get("failure_kind") == "nonfinite" for record in subset),
            "missing_or_incomplete_runs": sum(
                record.get("failure_kind") in {"missing", "incomplete"} for record in subset
            ),
            "parameter_count": next(
                (record.get("parameter_count") for record in subset if record.get("parameter_count") is not None),
                None,
            ),
        }
        for scope in ("selected", "final"):
            for metric in REPORT_METRICS:
                values = _finite_values(completed, f"{scope}_{metric}")
                mean, std = _mean_and_std(values)
                row[f"{scope}_{metric}_mean"] = mean
                row[f"{scope}_{metric}_std"] = std
        elapsed = _finite_values(completed, "elapsed_seconds")
        elapsed_mean, elapsed_std = _mean_and_std(elapsed)
        row["elapsed_seconds_mean"] = elapsed_mean
        row["elapsed_seconds_std"] = elapsed_std
        summary.append(row)
    return summary


def _run_fieldnames() -> list[str]:
    base = [
        "architecture",
        "seed",
        "status",
        "failure_kind",
        "error_type",
        "error",
        "last_step",
        "selected_step",
        "selected_checkpoint_score",
        "final_step",
        "final_checkpoint_score",
        "parameter_count",
        "elapsed_seconds",
        "return_code",
        "git_commit",
        "command",
        "run_dir",
    ]
    for scope in ("selected", "final"):
        base.extend(f"{scope}_{metric}" for metric in REPORT_METRICS)
    return base


def _summary_fieldnames() -> list[str]:
    base = [
        "architecture",
        "planned_runs",
        "completed_runs",
        "failed_runs",
        "collapsed_runs",
        "nonfinite_runs",
        "missing_or_incomplete_runs",
        "parameter_count",
        "elapsed_seconds_mean",
        "elapsed_seconds_std",
    ]
    for scope in ("selected", "final"):
        for metric in REPORT_METRICS:
            base.extend((f"{scope}_{metric}_mean", f"{scope}_{metric}_std"))
    return base


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 4) -> str:
    parsed = _float_value(value)
    return "-" if parsed is None else f"{parsed:.{digits}g}"


def _fmt_mean_std(row: dict[str, Any], key: str) -> str:
    mean = _float_value(row.get(f"{key}_mean"))
    std = _float_value(row.get(f"{key}_std"))
    if mean is None:
        return "-"
    return f"{mean:.4g}" if std is None else f"{mean:.4g} +/- {std:.2g}"


def render_markdown_report(records: Sequence[dict[str, Any]], summary: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Locked ablation report",
        "",
        f"Generated: `{utc_now_iso()}`",
        "",
        "Failed runs remain in the denominator and are excluded from metric means. Selected-checkpoint metrics are aggregated only for runs that completed the full locked budget.",
        "",
        "## Architecture summary",
        "",
        "| Architecture | Completed | Collapsed | Parameters | Final avg phase | Final ESS / N | Final max weight | z error | z2 error | radius2 error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {architecture} | {completed}/{planned} | {collapsed} | {params} | {phase} | {ess} | {max_weight} | {z} | {z2} | {radius2} |".format(
                architecture=row["architecture"],
                completed=row["completed_runs"],
                planned=row["planned_runs"],
                collapsed=row["collapsed_runs"],
                params=row.get("parameter_count", "-") or "-",
                phase=_fmt_mean_std(row, "final_valid_avg_phase"),
                ess=_fmt_mean_std(row, "final_valid_ess_fraction"),
                max_weight=_fmt_mean_std(row, "final_valid_max_weight_fraction"),
                z=_fmt_mean_std(row, "final_abs_err_z_mean"),
                z2=_fmt_mean_std(row, "final_abs_err_z2_mean"),
                radius2=_fmt_mean_std(row, "final_abs_err_radius2"),
            )
        )

    lines.extend(
        [
            "",
            "## Individual runs",
            "",
            "| Architecture | Seed | Status | Failure | Selected step | Selected score | Final phase | Final ESS / N | Final max weight |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        lines.append(
            "| {architecture} | {seed} | {status} | {failure} | {selected_step} | {selected_score} | {phase} | {ess} | {max_weight} |".format(
                architecture=record["architecture"],
                seed=record["seed"],
                status=record["status"],
                failure=record["failure_kind"],
                selected_step=record.get("selected_step") or "-",
                selected_score=_fmt(record.get("selected_checkpoint_score")),
                phase=_fmt(record.get("final_valid_avg_phase")),
                ess=_fmt(record.get("final_valid_ess_fraction")),
                max_weight=_fmt(record.get("final_valid_max_weight_fraction")),
            )
        )

    failures = [record for record in records if record.get("status") == "failed"]
    if failures:
        lines.extend(["", "## Recorded failures", ""])
        for record in failures:
            error = str(record.get("error", "")).replace("\n", " ").strip()
            lines.append(f"- `{record['architecture']}-seed{record['seed']}`: {error or record['failure_kind']}")

    commits = sorted({str(record.get("git_commit")) for record in records if record.get("git_commit")})
    if commits:
        lines.extend(["", "## Provenance", "", f"Commit(s): {', '.join(f'`{commit}`' for commit in commits)}"])
    lines.append("")
    return "\n".join(lines)


def write_reports(
    run_root: Path,
    results_dir: Path,
    architectures: Sequence[str] = tuple(ARCHITECTURES),
    seeds: Sequence[int] = LOCKED_SEEDS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = [
        collect_run_record(run_directory(run_root, architecture, seed), architecture, seed)
        for architecture in architectures
        for seed in seeds
    ]
    summary = aggregate_records(records)
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(results_dir / "ablation_runs.csv", records, _run_fieldnames())
    _write_csv(results_dir / "ablation_summary.csv", summary, _summary_fieldnames())
    (results_dir / "ablation_records.json").write_text(json.dumps(records, indent=2))
    (results_dir / "ablation_report.md").write_text(render_markdown_report(records, summary))
    return records, summary
