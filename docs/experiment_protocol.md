# Experiment protocol

This document defines the minimum experiment needed before ThimbleML should be presented as an empirical result rather than an implementation prototype.

## Research question

Can a holomorphic triangular contour flow improve held-out phase concentration and effective sample size on the complex phi4 benchmark while preserving observables measured against exact-grid integration?

## Hypotheses

- **H1:** Learned affine coupling improves held-out average phase relative to additive-only coupling under the same data and optimization budget.
- **H2:** Triangular complex-linear mixing improves either average phase or ESS relative to affine coupling alone without increasing exact-grid observable error.
- **H3:** Improvements are consistent across training seeds rather than being driven by one initialization.

A negative result is informative. The architecture is unsupported when gains reverse across seeds, disappear under a matched budget, require materially worse observable error, become non-finite, or rely on collapsed importance weights.

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

Do not regenerate proposal samples between model seeds.

## Locked optimization and failure settings

Use the same settings for every architecture unless a separate preregistered sensitivity study changes them:

```text
dtype = float64
learning rate = 2e-4
scale factor = 0.01
translation scale = 0.02
steps = 800
batch size = 512
validation batches = 10
minimum validation ESS fraction = 0.01
maximum single-weight fraction = 0.50
collapse patience = 1 validation
ess penalty = 0
checkpoint score = valid_avg_phase * valid_ess_fraction
```

A run is failed when it encounters non-finite values or when a validation reaches either collapse condition:

```text
valid_ess_fraction < 0.01
valid_max_weight_fraction > 0.50
```

Do not restart a failed seed with altered settings and silently include the replacement in the same comparison. Threshold changes and a nonzero `--ess-penalty` belong in separately declared sensitivity studies.

## Architecture ablations

Use identical proposal data, steps, batch size, validation batches, grid, learning rate, precision, thresholds, and seed set.

### Additive coupling

```bash
python main.py \
  --data data/proposal_samples.npz \
  --run-dir runs/additive-seed7 \
  --seed 7 \
  --scale-factor 0 \
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

## Checkpoint policy

Checkpoint selection is fixed before running experiments. Among non-collapsed validations, save the checkpoint that maximizes:

```text
valid_avg_phase * valid_ess_fraction
```

This score prevents a phase value near one from winning when it is supported by only one or a few samples. `model_last.pt` is written only when the full run completes successfully. A failed run may retain an earlier `model_best.pt` for diagnosis, but it remains a failed run and must not be counted as a successful final result.

Do not retrospectively select a checkpoint because its observable errors or headline metric look favourable.

## Primary metrics

Report the final successful validation and the selected checkpoint validation separately. Include:

- `valid_avg_phase`;
- `valid_ess` and `valid_ess_fraction`;
- `valid_max_weight_fraction`;
- `valid_weight_perplexity_fraction`;
- `valid_real_logw_range` and real-log-weight quantiles;
- `abs_err_z_mean`;
- `abs_err_z2_mean`;
- `abs_err_radius2`;
- wall-clock duration measured externally;
- run status and failure reason.

Also report `exact_avg_phase_original_grid` as a deterministic reference for sign-problem severity, not as a substitute for a matched held-out baseline.

## Required reporting

For every architecture, report:

- mean and standard deviation across the three seeds for successful runs;
- each individual seed result;
- successful runs out of three;
- collapsed, non-finite, or otherwise failed runs without replacement;
- parameter count;
- training budget and hardware;
- exact command and commit SHA;
- final and selected-checkpoint metrics.

A compact table should include average phase, normalized ESS, maximum weight fraction, weight perplexity fraction, the three observable errors, wall time, and run status.

## Interpretation rules

Treat an architecture as supported only when:

1. the direction of improvement is consistent across seeds;
2. the gain is not explained by worse observable error;
3. the comparison uses the same data and training budget;
4. determinant, inverse, stability, and collapse-diagnostic tests pass;
5. every included run completes without non-finite values or weight collapse;
6. improvements remain when phase, normalized ESS, and maximum weight fraction are considered together;
7. the result survives a modest change to validation batch count or grid resolution.

A near-one average phase with normalized ESS near zero is a collapse signature, not an improvement. Numerical survival alone is not evidence of statistical correctness.

## Further strengthening

Before publication-quality claims, add:

- bootstrap intervals over held-out proposal samples;
- a compute-matched comparison;
- a larger-dimensional benchmark with a trustworthy Monte Carlo reference;
- separately declared loss-component and ESS-penalty ablations;
- tail diagnostics for rare extreme weights;
- a versioned script that aggregates run directories into one results table.
