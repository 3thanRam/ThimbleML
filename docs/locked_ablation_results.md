# Locked architecture ablation: completed result

## Outcome

The preregistered `3 architectures × 3 seeds` experiment completed with a uniform negative result: all nine runs remained numerically finite but failed because of held-out importance-weight collapse.

| Architecture | Planned runs | Completed runs | Failed runs | Collapsed runs | Non-finite runs | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Additive coupling | 3 | 0 | 3 | 3 | 0 | 912 |
| Affine coupling | 3 | 0 | 3 | 3 | 0 | 912 |
| Affine + triangular-linear mixing | 3 | 0 | 3 | 3 | 0 | 944 |

No final successful-run metrics are reported because no run completed without triggering the preregistered collapse criteria. Earlier `model_best.pt` files are diagnostic checkpoints from failed runs and are not counted as successful results.

## Interpretation

Collapse occurred for additive, affine, and full triangular-linear variants across seeds `7`, `17`, and `27`. This does not support the architecture hypotheses:

- **H1 unsupported:** affine coupling did not produce a successful run relative to additive coupling.
- **H2 unsupported:** triangular-linear mixing did not prevent collapse.
- **H3 negative consistency:** the failure reproduced across all three seeds for every architecture.

Because no run failed from NaNs or infinities, the numerical-stability safeguards worked as intended. The evidence instead points to a failure shared across architectures, most plausibly the common phase-focused objective, proposal distribution, or optimization trajectory. This experiment does not distinguish among those shared causes.

## Execution record

- Protocol: `locked-ablation-v1`
- Experiment code commit: `0309734d72f708afcf4a4976455fc493542fda60`
- Architectures: `additive`, `affine`, `full`
- Model seeds: `7`, `17`, `27`
- Proposal-data seed: `1234`
- Dataset: 80,000 training and 20,000 validation proposal samples
- Dimension: `2`
- Proposal scale: `1.6`
- Training steps: `800`
- Batch size: `512`
- Validation batches: `10`
- Grid points: `401`
- Precision: `float64`
- Learning rate: `2e-4`
- Coupling scale factor: `0.01` for affine and full; `0` for additive
- Translation scale: `0.02`
- Minimum validation ESS fraction: `0.01`
- Maximum single-weight fraction: `0.50`
- Collapse patience: `1` validation
- ESS penalty: `0`
- Checkpoint score: `valid_avg_phase * valid_ess_fraction`

The generated local `results/locked_ablation/manifest.json` and per-run `run_manifest.json` files remain the authoritative machine-readable records for exact commands, runtime environment, elapsed time, and execution commit.

## Reproduction

```bash
python scripts/run_locked_ablation.py --device auto
```

The experiment design and reporting rules are defined in [`experiment_protocol.md`](experiment_protocol.md), and the executable workflow is documented in [`locked_ablation.md`](locked_ablation.md).

## Scope

This is a negative empirical result, not evidence that holomorphic contour flows cannot work. It shows that this shared objective and training setup did not yield a statistically supported estimator under the locked comparison. Any ESS penalty, threshold change, proposal change, or objective modification belongs in a separately declared sensitivity study rather than a replacement run inside this experiment.
