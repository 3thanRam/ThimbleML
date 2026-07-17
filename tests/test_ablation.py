from __future__ import annotations

import csv
import json
from pathlib import Path

from src.ablation import (
    LOCKED_SEEDS,
    aggregate_records,
    build_training_command,
    collect_run_record,
    write_reports,
)


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_locked_commands_encode_the_three_architectures(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    data = tmp_path / "data.npz"
    run = tmp_path / "run"

    additive = build_training_command("python", repo, data, run, "additive", LOCKED_SEEDS[0], device="cpu")
    affine = build_training_command("python", repo, data, run, "affine", LOCKED_SEEDS[0], device="cpu")
    full = build_training_command("python", repo, data, run, "full", LOCKED_SEEDS[0], device="cpu")

    assert additive[additive.index("--scale-factor") + 1] == "0.0"
    assert "--no-triangular-linear" in additive
    assert affine[affine.index("--scale-factor") + 1] == "0.01"
    assert "--no-triangular-linear" in affine
    assert full[full.index("--scale-factor") + 1] == "0.01"
    assert "--no-triangular-linear" not in full
    assert full[full.index("--ess-penalty") + 1] == "0"


def test_failed_run_keeps_selected_checkpoint_but_has_no_final_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "full-seed7"
    run_dir.mkdir()
    rows = [
        {
            "step": 560,
            "validation_collapsed": 0,
            "checkpoint_score": 0.0265,
            "valid_avg_phase": 0.045,
            "valid_ess": 3008.0,
            "valid_ess_fraction": 0.587,
            "valid_max_weight_fraction": 0.0005,
            "valid_weight_perplexity_fraction": 0.659,
            "valid_real_logw_range": 45.5,
            "abs_err_z_mean": 1.12,
            "abs_err_z2_mean": 1.37,
            "abs_err_radius2": 2.75,
            "exact_avg_phase_original_grid": 0.026,
        },
        {
            "step": 640,
            "validation_collapsed": 1,
            "checkpoint_score": 0.0002,
            "valid_avg_phase": 0.983,
            "valid_ess": 1.04,
            "valid_ess_fraction": 0.0002,
            "valid_max_weight_fraction": 0.983,
            "valid_weight_perplexity_fraction": 0.00024,
            "valid_real_logw_range": 56.2,
            "abs_err_z_mean": 4.86,
            "abs_err_z2_mean": 12.48,
            "abs_err_radius2": 24.96,
            "exact_avg_phase_original_grid": 0.026,
        },
    ]
    _write_metrics(run_dir / "metrics.csv", rows)
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "error_type": "RuntimeError",
                "error": "Importance-weight collapse at validation step 640",
            }
        )
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"parameter_count": 123, "elapsed_seconds": 12.5, "git_commit": "abc", "return_code": 1})
    )

    record = collect_run_record(run_dir, "full", 7)

    assert record["failure_kind"] == "collapse"
    assert record["selected_step"] == 560
    assert record["selected_valid_ess_fraction"] == 0.587
    assert record["final_step"] is None
    assert record["final_valid_avg_phase"] is None


def test_aggregate_excludes_failed_runs_from_means() -> None:
    records = [
        {
            "architecture": "affine",
            "status": "completed",
            "failure_kind": "none",
            "parameter_count": 10,
            "final_valid_avg_phase": 0.2,
            "selected_valid_avg_phase": 0.25,
            "elapsed_seconds": 10.0,
        },
        {
            "architecture": "affine",
            "status": "failed",
            "failure_kind": "collapse",
            "parameter_count": 10,
            "final_valid_avg_phase": None,
            "selected_valid_avg_phase": 0.99,
            "elapsed_seconds": 8.0,
        },
    ]

    summary = aggregate_records(records)[0]

    assert summary["completed_runs"] == 1
    assert summary["collapsed_runs"] == 1
    assert summary["final_valid_avg_phase_mean"] == 0.2
    assert summary["selected_valid_avg_phase_mean"] == 0.25


def test_write_reports_materializes_all_formats(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    results = tmp_path / "results"
    records, summary = write_reports(run_root, results, architectures=["additive"], seeds=[7])

    assert len(records) == 1
    assert len(summary) == 1
    assert (results / "ablation_runs.csv").exists()
    assert (results / "ablation_summary.csv").exists()
    assert (results / "ablation_records.json").exists()
    assert (results / "ablation_report.md").exists()
