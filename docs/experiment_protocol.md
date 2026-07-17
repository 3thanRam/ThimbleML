# Experiment protocol

This document defines the minimum experiment needed before ThimbleML should be presented as an empirical result rather than an implementation prototype.

## Research question

Can a holomorphic triangular contour flow improve held-out phase concentration and effective sample size on the complex phi4 benchmark while preserving observables measured against exact-grid integration?

## Hypotheses

- **H1:** Learned affine coupling improves held-out average phase relative to additive-only coupling under the same data and optimization budget.
- **H2:** Triangular complex-linear mixing improves either average phase or ESS relative to affine coupling alone without increasing exact-grid observable error.
- **H3:** Improvements are consistent across training seeds rather than being driven by one initialization.

A negative result is informative. The architecture should be reported as unsupported when gains reverse across seeds, disappear under a matched budget, or require materially worse observable error.

## Locked data

Generate the proposal dataset once and reuse it for every architecture and training seed:

```bash
python scripts/make_data.py \
  --out data/proposal_samples.npz \
  --dim 2 \
  --proposal-scale 1.6 \
  --train-samples 80000 \
  --valid-samples 20000 \
  --seed 1234
```

Do not regenerate the proposal samples between model seeds. This isolates model initialization and optimization variability from data variability.

## Architecture ablations

Use the same number of steps, batch size, validation batches, grid, and learning rate for all configurations.

### Additive coupling

```bash
python main.py \
  --data data/proposal_samples.npz \
  --run-dir runs/additive-seed7 \
  --seed 7 \
  --scale-clip 0 \
  --no-triangular-linear
```

### Affine coupling

```bash
python main.py \
  --data data/proposal_samples.npz \
  --run-dir runs/affine-seed7 \
  --seed 7 \
  --no-triangular-linear
```

### Affine coupling plus triangular-linear mixing

```bash
python main.py \
  --data data/proposal_samples.npz \
  --run-dir runs/full-seed7 \
  --seed 7
```

Repeat each configuration with seeds `7`, `17`, and `27`.

## Primary metrics

Read the final held-out values from each run's `metrics.csv`:

- `valid_avg_phase`
- `valid_ess`
- `abs_err_z_mean`
- `abs_err_z2_mean`
- `abs_err_radius2`
- wall-clock duration measured externally

Also report the original-contour grid reference stored as `exact_avg_phase_original_grid`. This is a deterministic reference for sign-problem severity, not a substitute for a matched held-out model baseline.

## Required reporting

For every architecture, report:

- mean and standard deviation across the three training seeds;
- each individual seed result, not only the aggregate;
- parameter count;
- training budget and hardware;
- exact command and commit SHA;
- failures, NaNs, or excluded runs.

A compact table should have one row per architecture and columns for average phase, ESS, the three observable errors, parameter count, and wall time.

## Interpretation rules

Treat an architecture as supported only when:

1. the direction of improvement is consistent across seeds;
2. the gain is not explained by materially higher observable error;
3. the comparison uses the same proposal data and training budget;
4. all determinant and inverse tests pass at the reported commit;
5. the result survives at least one modest change to the validation batch count or grid resolution.

Do not describe a one-seed increase as a robust improvement. Do not select only the best checkpoint or seed without stating that selection rule in advance.

## Further strengthening

Before publication-quality claims, add:

- bootstrap intervals over held-out proposal samples;
- a compute-matched comparison;
- a larger-dimensional benchmark with a trustworthy Monte Carlo reference;
- component-level loss ablations;
- checks for rare extreme weights and estimator instability;
- a script that aggregates all run directories into a single versioned results table.
