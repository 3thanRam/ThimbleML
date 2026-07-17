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
    p = argparse