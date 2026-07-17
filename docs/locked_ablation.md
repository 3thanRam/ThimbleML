# Executable locked ablation

The locked experiment compares three architectures on one fixed proposal dataset and seeds `7`, `17`, and `27`:

1. additive coupling without triangular-linear mixing;
2. affine coupling without triangular-linear mixing;
3. affine coupling with triangular-linear mixing.

All nine runs use the preregistered settings in [`experiment_protocol.md`](experiment_protocol.md): 800 steps, batch size 512, ten validation batches, double precision, learning rate `2e-4`, collapse thresholds of 1% normalized ESS and 50% maximum weight, and no ESS penalty.

## Run the complete design

```bash
python scripts/run_locked_ablation.py --device auto
```

The runner generates `data/proposal_samples.npz` only when it is absent. It then executes the full 3x3 design sequentially. A training process that exits because of importance-weight collapse is recorded as a failed run and does not stop the remaining experiments.

Use a dry run to inspect all commands without creating files:

```bash
python scripts/run_locked_ablation.py --dry-run --device cpu
```

Existing non-empty run directories are never overwritten implicitly:

```bash
python scripts/run_locked_ablation.py --resume
python scripts/run_locked_ablation.py --overwrite
```

`--resume` skips directories containing `status.json`, including declared failures. `--overwrite` deletes and reruns the selected directories. Architecture or seed subsets are supported for debugging, but only the default nine-run invocation is the complete locked design.

## Outputs

Per-run artifacts are stored under:

```text
runs/locked_ablation/additive-seed7/
runs/locked_ablation/affine-seed7/
runs/locked_ablation/full-seed7/
...
```

Each directory includes the normal training artifacts plus:

```text
run_manifest.json
stdout.log
stderr.log
```

The manifest records the exact command, commit SHA, parameter count, elapsed time, return code, software versions, and locked settings.

Aggregate outputs are rebuilt after every run and written to:

```text
results/locked_ablation/manifest.json
results/locked_ablation/ablation_runs.csv
results/locked_ablation/ablation_summary.csv
results/locked_ablation/ablation_records.json
results/locked_ablation/ablation_report.md
```

Reports can be rebuilt without rerunning training:

```bash
python scripts/summarize_ablation.py
```

Failed runs remain in the denominator and are excluded from metric means. Earlier diagnostic checkpoints from failed runs are preserved in the individual-run table, but they are not aggregated as successful results.
