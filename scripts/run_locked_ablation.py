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
    parser = argparse.ArgumentParser(description="Run the preregistered 3x3 ThimbleML ablation.")
    parser.add_argument("--data", type=Path, default=Path("data/proposal_samples.npz"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/locked_ablation"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/locked_ablation"))
    parser.add_argument("--architectures", nargs="+", choices=tuple(ARCHITECTURES), default=list(ARCHITECTURES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(LOCKED_SEEDS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def git_commit(repo_root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def parameter_count(architecture: str) -> int:
    spec = architecture_spec(architecture)
    model = HolomorphicTriangularFlow(
        dim=int(LOCKED_SETTINGS["dim"]),
        num_layers=int(LOCKED_SETTINGS["layers"]),
        hidden_dim=int(LOCKED_SETTINGS["hidden"]),
        depth=int(LOCKED_SETTINGS["depth"]),
        activation=LOCKED_SETTINGS["activation"],
        dtype=torch.float64,
        scale_factor=spec.scale_factor,
        translation_scale=float(LOCKED_SETTINGS["translation_scale"]),
        use_triangular_linear=spec.use_triangular_linear,
    )
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def prepare_run_dir(path: Path, resume: bool, overwrite: bool) -> bool:
    if not path.exists() or not any(path.iterdir()):
        path.mkdir(parents=True, exist_ok=True)
        return True
    if overwrite:
        shutil.rmtree(path)
        path.mkdir(parents=True)
        return True
    if resume and (path / "status.json").exists():
        return False
    raise FileExistsError(f"{path} is not empty; use --resume or --overwrite")


def main() -> None:
    args = parse_args()
    if sorted(set(args.seeds) - set(LOCKED_SEEDS)):
        raise ValueError(f"Locked seeds are {LOCKED_SEEDS}")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.architectures)) != len(args.architectures):
        raise ValueError("Architectures and seeds must not contain duplicates")

    repo_root = Path(__file__).resolve().parents[1]
    data_path = args.data if args.data.is_absolute() else repo_root / args.data
    run_root = args.run_root if args.run_root.is_absolute() else repo_root / args.run_root
    results_dir = args.results_dir if args.results_dir.is_absolute() else repo_root / args.results_dir
    commit = git_commit(repo_root)
    data_command = build_data_command(sys.executable, repo_root, data_path)
    commands = []
    for architecture in args.architectures:
        for seed in args.seeds:
            target = run_directory(run_root, architecture, seed)
            command = build_training_command(sys.executable, repo_root, data_path, target, architecture, seed, args.device)
            commands.append((architecture, seed, target, command))

    if args.dry_run:
        if not data_path.exists():
            print(command_string(data_command))
        for _, _, _, command in commands:
            print(command_string(command))
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "manifest.json").write_text(json.dumps({
        "protocol": "locked-ablation-v1",
        "created_at": utc_now_iso(),
        "git_commit": commit,
        "architectures": args.architectures,
        "seeds": args.seeds,
        "complete_design": args.architectures == list(ARCHITECTURES) and args.seeds == list(LOCKED_SEEDS),
        "locked_settings": LOCKED_SETTINGS,
        "device": args.device,
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
    }, indent=2))
    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(data_command, cwd=repo_root, check=True)

    errors: list[str] = []
    for architecture, seed, target, command in commands:
        try:
            should_run = prepare_run_dir(target, args.resume, args.overwrite)
        except Exception as exc:
            errors.append(str(exc))
            continue
        if not should_run:
            write_reports(run_root, results_dir, args.architectures, args.seeds)
            continue

        manifest_path = target / "run_manifest.json"
        manifest = {
            "protocol": "locked-ablation-v1",
            "architecture": architecture,
            "seed": seed,
            "parameter_count": parameter_count(architecture),
            "git_commit": commit,
            "locked_settings": LOCKED_SETTINGS,
            "command": command,
            "command_shell": command_string(command),
            "started_at": utc_now_iso(),
            "device_requested": args.device,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        started = time.perf_counter()
        with (target / "stdout.log").open("w") as stdout, (target / "stderr.log").open("w") as stderr:
            result = subprocess.run(command, cwd=repo_root, stdout=stdout, stderr=stderr, check=False)
        manifest.update({"finished_at": utc_now_iso(), "elapsed_seconds": time.perf_counter() - started, "return_code": result.returncode})
        manifest_path.write_text(json.dumps(manifest, indent=2))
        write_reports(run_root, results_dir, args.architectures, args.seeds)

    write_reports(run_root, results_dir, args.architectures, args.seeds)
    print(f"Reports written to {results_dir}")
    if errors:
        raise RuntimeError("Ablation orchestration errors:\n- " + "\n- ".join(errors))


if __name__ == "__main__":
    main()
