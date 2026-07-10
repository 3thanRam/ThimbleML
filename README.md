# ThimbleML Triangular Holomorphic Flow

This is a minimal, runnable implementation of a Lefschetz-thimble-inspired neural contour learner using **holomorphic triangular affine coupling layers** with exact complex determinant tracking.

The project deliberately avoids transformer/SSM-style blocks, real/imag feature concatenation, layer norms, softmax attention, or real-valued gates in the actual contour map. Those components are useful ML machinery, but they generally do **not** preserve holomorphicity and do not give the correct contour Jacobian phase.

## Core idea

We estimate integrals of the form

```text
Z = integral_R^n exp(-S(x)) dx
```

by learning a holomorphic contour deformation

```text
z = f_theta(x + 0i)
```

and using the unbiased contour estimator

```text
w(x) = exp(-S(z)) det(df/dz) / q(x),  x ~ q(x)
```

The determinant is complex-valued and its phase is essential.

## Why triangular maps?

The model combines literal learned triangular complex matrices and nonlinear affine coupling layers. Each affine coupling layer is

```text
z_a' = z_a
z_b' = z_b * exp(s_theta(z_a)) + t_theta(z_a)
```

where `s_theta` and `t_theta` are holomorphic complex MLPs. The Jacobian is block triangular, so

```text
log det J = sum s_theta(z_a)
```

exactly. This avoids dense `O(n^3)` determinant tracking.

## Files

```text
src/complex_layers.py          complex-linear + holomorphic MLP components
src/triangular_flow.py         triangular holomorphic affine coupling + triangular linear flow
src/benchmarks.py              complex phi^4 polynomial sign-problem benchmark
src/estimator.py               contour estimator, phase, ESS, loss
src/data_loading.py            synthetic proposal-data generation/loading
src/train.py                   full training loop and validation
scripts/make_data.py           regenerate proposal samples
tests/test_triangular_jacobian.py  determinant and inverse tests
data/proposal_samples.npz      included synthetic training/validation samples
```

## Install

```bash
pip install -r requirements.txt
```

## Run tests

```bash
pytest -q
```

The determinant test compares the analytic triangular determinant against a finite-difference complex Jacobian determinant.

## Smoke train

```bash
python main.py --steps 5 --batch-size 64 --grid-points 51 --valid-batches 1 --run-dir runs/smoke
```

## Main training run

```bash
python main.py --steps 800 --batch-size 512 --grid-points 401 --run-dir runs/main
```

Outputs:

```text
runs/main/config.json
runs/main/metrics.csv
runs/main/model.pt
runs/main/contour_projection_z0.png  # created only when --plot-contour is passed
```

## Regenerate data

```bash
python scripts/make_data.py --out data/proposal_samples.npz --dim 2 --train-samples 80000 --valid-samples 20000
```

The dataset is not external physics data. It is a fixed set of proposal samples from

```text
x ~ Normal(0, proposal_scale^2 I)
```

The action supplies the physics. Exact grid integration validates observables for low dimension.

## CV framing

A good project description:

> Built a holomorphic triangular neural contour flow for oscillatory complex path integrals, with exact complex Jacobian phase tracking and validation against exact-grid observables on a phi^4 sign-problem benchmark.

Useful metrics to report:

- average phase before/after learning
- effective sample size
- observable error against exact grid values
- determinant finite-difference test
- contour plots in 2D (`--plot-contour`)
- ablations: identity contour, additive coupling, affine coupling
