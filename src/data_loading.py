from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class ProposalConfig:
    dim: int = 2
    proposal_scale: float = 1.6
    train_samples: int = 80000
    valid_samples: int = 20000
    seed: int = 1234


def make_proposal_npz(path: str | Path, cfg: ProposalConfig) -> Path:
    rng = np.random.default_rng(cfg.seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    train_x = rng.normal(0.0, cfg.proposal_scale, size=(cfg.train_samples, cfg.dim)).astype("float32")
    valid_x = rng.normal(0.0, cfg.proposal_scale, size=(cfg.valid_samples, cfg.dim)).astype("float32")
    np.savez_compressed(
        path,
        train_x=train_x,
        valid_x=valid_x,
        dim=np.array(cfg.dim, dtype=np.int64),
        proposal_scale=np.array(cfg.proposal_scale, dtype=np.float32),
        seed=np.array(cfg.seed, dtype=np.int64),
    )
    return path


def load_proposal_npz(path: str | Path) -> tuple[torch.Tensor, torch.Tensor, float]:
    data = np.load(path)
    train = torch.from_numpy(data["train_x"].astype("float32"))
    valid = torch.from_numpy(data["valid_x"].astype("float32"))
    scale = float(data["proposal_scale"])
    return train, valid, scale


def make_loaders(path: str | Path, batch_size: int, num_workers: int = 0) -> tuple[DataLoader, DataLoader, float]:
    train, valid, scale = load_proposal_npz(path)
    train_loader = DataLoader(TensorDataset(train), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    valid_loader = DataLoader(TensorDataset(valid), batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return train_loader, valid_loader, scale
