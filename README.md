# ThimbleML: Triangular Holomorphic Contour Flow

[![CI](https://github.com/3thanRam/ThimbleML/actions/workflows/ci.yml/badge.svg)](https://github.com/3thanRam/ThimbleML/actions/workflows/ci.yml)

ThimbleML is a small PyTorch research prototype for learning holomorphic contour deformations for oscillatory complex integrals. It uses triangular affine coupling layers and learned triangular complex-linear maps, so the full complex Jacobian determinant is available analytically.

The repository is designed around a falsifiable question:

> Can a learned holomorphic contour improve phase concentration and effective sample size while preserving observables validated against an exact low-dimensional reference?

This is a research prototype, not a claim that the current architecture is state of the art.

## Curent Status

In a preregistered 3×3 architecture ablation, all nine runs remained numerically finite but eventually underwent held-out importance-weight collapse. Collapse occurred in additive, affine, and triangular-linear variants across all seeds, indicating that the failure was associated with the shared phase-focused objective rather than a single architectural component.

## Core estimator

For

```text
Z = integral_R^n exp(-S(x)) dx
```

we learn a holomorphic deformation `z = f_theta(x + 0i)` and estimate it with

```text
w(x) = exp(-S(z)) det(df/dz) / q(x),  x ~ q(x)
```

The determinant is complex-valued. Its phase is part of the estimator and is tracked exactly by the flow.

Each affine coupling layer has the form

```text
z_a' = z_a
z_b' = z_b * exp(s_theta(z_a)) + t_theta(z_a)
```

so its block-triangular Jacobian has an exact log determinant. The log-scale head is multiplied by a small factor directly; complex `tanh` is not used as a clamp because it has poles and is not bounded over the complex plane.

## What is checked

The implementation tests Jacobian correctness, inverse round trips, exact-grid observable references, and numerical finiteness. Training also measures whether the importance estimator remains statistically supported rather than being dominated by one sample.

Validation records:

- average phase and absolute-weight ESS;
- normalized ESS, `ESS / N`;
- largest normalized absolute weight;
- weight entropy and perplexity;
- real-log-weight quantiles and range;
- exact-reference observable errors in low dimension.

A near-one average phase is **not** treated as success when ESS has collapsed. By default, a run fails when validation ESS falls below 1% of the sample count or one absolute weight exceeds 50% of the total. These thresholds are configurable, recorded in `config.json`, and should be fixed before a comparison.

## Quick start

ThimbleML requires Python 3.10+ and PyTorch 2.1+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Generate the locked proposal data:

```bash
python scripts/make_data.py \
  --out data/proposal_samples.npz \
  --dim 2 \
  --train-samples 80000 \
  --valid-samples 20000 \
  --seed 1234
```

Run the reference configuration:

```bash
python main.py \
  --data data/proposal_samples.npz \
  --steps 800 \
  --batch-size 512 \
  --grid-points 401 \
  --run-dir runs/main
```

The finite-safe defaults use double precision, learning rate `2e-4`, log-scale factor `0.01`, and translation scale `0.02`. They are conservative engineering defaults, not tuned benchmark hyperparameters.

## Run artifacts

A run writes:

```text
runs/main/config.json
runs/main/metrics.csv
runs/main/status.json
runs/main/model_best.pt   # best non-collapsed validation score
runs/main/model_last.pt   # only after successful completion
```

The checkpoint rule is fixed in advance: maximize

```text
valid_avg_phase * valid_ess_fraction
```

among validations that do not trigger collapse thresholds. If training later collapses, `status.json` records failure and `model_last.pt` is not written. An earlier `model_best.pt` may remain for diagnosis, but it does not turn the failed run into a successful result.

## Reproducible ablations

Use the same proposal data, optimization budget, precision, thresholds, and seed set for:

- additive coupling: `--scale-factor 0 --no-triangular-linear`;
- affine coupling: `--no-triangular-linear`;
- affine coupling plus triangular-linear mixing: default settings.

Repeat with seeds `7`, `17`, and `27`. The complete locked policy and interpretation rules are in [`docs/experiment_protocol.md`](docs/experiment_protocol.md).

An optional `--ess-penalty` adds a differentiable `-log(batch ESS / batch size)` penalty. It defaults to zero and should only be used in a separately declared sensitivity study, not introduced after inspecting a failed run.

## Current limitations

- The benchmark is deliberately low-dimensional and synthetic.
- The loss is a practical phase-alignment objective, not a uniquely justified optimum.
- Finite execution does not establish estimator validity or empirical superiority.
- Collapse detection prevents a misleading success signal; it does not solve the underlying contour-learning problem.
- Exact-grid validation scales poorly beyond very small dimension.
- The repository does not yet contain a completed multi-seed benchmark table with uncertainty intervals.

## License

[MIT](LICENSE)
