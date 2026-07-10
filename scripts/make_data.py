from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loading import ProposalConfig, make_proposal_npz


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/proposal_samples.npz")
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--proposal-scale", type=float, default=1.6)
    p.add_argument("--train-samples", type=int, default=80000)
    p.add_argument("--valid-samples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    path = make_proposal_npz(
        Path(args.out),
        ProposalConfig(
            dim=args.dim,
            proposal_scale=args.proposal_scale,
            train_samples=args.train_samples,
            valid_samples=args.valid_samples,
            seed=args.seed,
        ),
    )
    print(path)


if __name__ == "__main__":
    main()
