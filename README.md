# ThimbleML: Triangular Holomorphic Contour Flow

[![CI](https://github.com/3thanRam/ThimbleML/actions/workflows/ci.yml/badge.svg)](https://github.com/3thanRam/ThimbleML/actions/workflows/ci.yml)

ThimbleML is a small PyTorch research prototype for learning holomorphic contour deformations for oscillatory complex integrals. It uses triangular affine coupling layers and learned triangular complex-linear maps, so the full complex Jacobian determinant is available analytically.

The repository is designed around a falsifiable question:

> Can a learned holomorphic contour improve phase concentration and effective sample size while preserving observables validated against an exact low-dimensional reference?

This is a research prototype, not a claim that the current architecture is state of the art. The next research step is a locked multi-seed ablation with uncertainty intervals; see [`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## Core estimator

For an integral

```text
Z = integral_R^n exp(-S(x)) dx
```

we learn a holomorphic deformation

```text
z = f_theta(x + 0i)
```

and estimate it with

```text
w(x) = exp(-S(z)) det(df/dz) / q(x),  x ~ q(x)
```

The determinant is complex-valued. Its phase is part of the estimator and is tracked exactly by the flow.

## Why triangular maps?

Each affine coupling layer has the form

```text
z_a' = z_a
z_b' = z_b * exp(s_theta(z_a)) + t_theta(z_a)
```

where `s_theta` and `t_theta` are holomorphic complex MLPs. The Jacobian is block triangular, so

```text
log det J = sum s_theta(z_a)
```

exactly. Learned lower- and upper-triangular complex-linear layers provide additional mixing while retaining an analytic determinant and inverse.

The contour map deliberately avoids operations such as real/imaginary feature splitting, ReLU, magnitude gates, layer normalization, and softmax attention, because they do not preserve holomorphicity in the required sense. The log-scale head is multiplied by a small factor directly; complex `tanh` is not used as a clamp because it has poles and is not bounded over the complex plane.

## Falsifiable checks

The implementation is built around four independent checks:

1. **Jacobian correctness.** The analytic determinant is compared with a finite-difference complex Jacobian determinant.
2. **Invertibility.** Forward and inverse maps must round-trip in double precision, with inverse log determinant equal to the negative forward value.
3. **Observable preservation.** Low-dimensional observables are compared with exact-grid integration on the original real contour.
4. **Numerical finiteness.** Training stops immediately if contour values, weights, loss, gradients, parameters, validation outputs, or logged metrics become non-finite.

Training additionally records held-out average phase, effective sample size, and absolute observable error. A useful result must improve sampling diagnostics without introducing unacceptable observable error.

## Repository map

```text
src/complex_layers.py             complex-linear and holomorphic MLP components
src/triangular_flow.py            coupling, triangular-linear, inverse, log determinant
src/benchmarks.py                 complex phi^4 sign-problem benchmark
src/estimator.py                  contour weights, phase, ESS, loss, observables
src/data_loading.py               deterministic proposal-data generation and loading
src/train.py                      training loop, validation, finite-value checks
scripts/make_data.py              proposal-sample generation
tests/test_triangular_jacobian.py determinant and inverse tests
tests/test_exact_grid.py          exact-grid smoke test
tests/test_identity_baseline.py   identity and estimator sanity checks
tests/test_training_stability.py  non-finite and complex-scale regression tests
docs/experiment_protocol.md       locked ablation and reporting protocol
```

Generated proposal data and run artifacts are intentionally not committed.

## Quick start

ThimbleML requires Python 3.10+ and PyTorch 2.1+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

A short CPU smoke run:

```bash
python main.py \
  --steps 40 \
  --batch-size 64 \
  --train-samples 1024 \
  --valid-samples 256 \
  --grid-points 51 \
  --valid-batches 1 \
  --device cpu \
  --run-dir runs/smoke
```

The proposal dataset is generated automatically when it does not exist.

A larger reference run using the finite-safe defaults:

```bash
python main.py \
  --steps 800 \
  --batch-size 512 \
  --grid-points 401 \
  --run-dir runs/main
```

The defaults use double precision, learning rate `2e-4`, log-scale factor `0.01`, and translation scale `0.02`. These are conservative engineering defaults, not tuned benchmark hyperparameters.

Outputs include:

```text
runs/main/config.json
runs/main/metrics.csv
runs/main/model.pt
runs/main/contour_projection_z0.png  # only with --plot-contour
```

A checkpoint is written only after all requested training steps finish with finite values. Failed runs raise an error instead of saving a corrupted model.

## Reproducible ablations

The main comparisons use the same fixed proposal dataset and training budget:

- additive coupling: `--scale-factor 0 --no-triangular-linear`
- affine coupling: `--no-triangular-linear`
- affine coupling plus triangular-linear mixing: default settings

The former `--scale-clip` option remains accepted as a hidden compatibility alias, but the value is a multiplicative factor rather than a mathematical clip.

The full seed policy, commands, metrics, and interpretation rules are in [`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## Regenerate proposal data

```bash
python scripts/make_data.py \
  --out data/proposal_samples.npz \
  --dim 2 \
  --train-samples 80000 \
  --valid-samples 20000 \
  --seed 1234
```

The dataset is not external physics data. It is a deterministic set of proposal samples from

```text
x ~ Normal(0, proposal_scale^2 I)
```

The action supplies the physics. Exact-grid integration is used only for low-dimensional validation, not as a supervised training target.

## Current limitations

- The included benchmark is deliberately low-dimensional and synthetic.
- The current loss is a practical phase-alignment objective, not a uniquely justified optimum.
- Reported ESS is based on absolute centered weights and should be interpreted with the accompanying phase and observable errors.
- The repository does not yet contain a completed multi-seed benchmark table with bootstrap or repeated-run uncertainty intervals.
- Exact-grid validation scales poorly beyond very small dimension.
- Finite training is necessary but does not establish statistical correctness or empirical superiority.

These limitations are part of the experimental scope rather than hidden implementation details.

## License

[MIT](LICENSE)
