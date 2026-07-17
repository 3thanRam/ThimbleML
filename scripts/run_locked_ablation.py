from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.ablation import (
    ARCHITECTURES,
    LOCKED_SEEDS,
    LOCKED_SETTINGS,
    architecture_spec,
    build_data_command,
    build_training_command,
    command_string,
    run_directory,
    utc_now_iso,
    write_reports,
)
from src.triangular_flow import HolomorphicTriangularFlow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the preregistered 3x3 ThimbleML architecture ablation.")
