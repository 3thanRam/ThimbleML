from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ablation import ARCHITECTURES, LOCKED_SEEDS, write_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild locked-ablation CSV, JSON, and Markdown reports.")
    parser.add_argument("--run-root", type=Path, default=Path("runs/locked_ablation"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/locked_ablation"))
    parser.add_argument("--architectures", nargs="+", choices=tuple(ARCHITECTURES), default=list(ARCHITECTURES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(LOCKED_SEEDS))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_root = args.run_root if args.run_root.is_absolute() else repo_root / args.run_root
    results_dir = args.results_dir if args.results_dir.is_absolute() else repo_root / args.results_dir
    invalid_seeds = sorted(set(args.seeds) - set(LOCKED_SEEDS))
    if invalid_seeds:
        raise ValueError(f"Locked ablation seeds are {LOCKED_SEEDS}; received unsupported seeds {invalid_seeds}")
    records, summary = write_reports(run_root, results_dir, args.architectures, args.seeds)
    print(f"Wrote {len(records)} run records and {len(summary)} architecture summaries to {results_dir}")


if __name__ == "__main__":
    main()
