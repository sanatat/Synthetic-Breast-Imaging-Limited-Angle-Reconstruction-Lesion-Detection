# V10 — Cone-beam forward-model and SIRT convergence audit

**Scope:** Numerical diagnostic experiment on small synthetic volumes, **not** calibrated breast tomosynthesis, patient imaging, or clinical validation.

## Methods

The sparse divergent-ray projector from V8 was reused with 9, 15 and 21 views at ±18°. Volumes had shape (12, 18, 18), with two synthetic phantom/noise seeds. Reconstruction was evaluated after 1, 5, 15, 30 and 60 normalized SIRT iterations. Two simplified noise settings were recorded: constant projection Gaussian noise and an approximate *fixed-total-dose proxy* in which noise standard deviation grows with the square root of the relative number of views. This proxy **is not** a photon-counting or radiation-dose model.

The test performed a dot-product adjoint check and estimated the fraction of volume voxels with nonzero column-sum sensitivity. Operator, phantom generator and evaluation metrics are inherited from V8–V9. The reconstruction forward operator and synthetic data generator are identical (**inverse crime**), so this test is less demanding than independent validation.

## Results

| Views | Adjoint relative error | Volume coverage | Lesion MAE (iteration 60) | Projection RMSE (iteration 60) |
|---:|---:|---:|---:|---:|
| 9 | 2.73e-6 | 0.751 | 0.3850 | 0.0635 |
| 15 | 0.00 (float rounding) | 0.751 | 0.3877 | 0.0643 |
| 21 | 2.15e-6 | 0.751 | 0.3890 | 0.0648 |

The tables correspond to **fixed per-view noise** and two synthetic seeds only. The per-iteration, alternative noise-proxy, and per-seed results are in `results_v10/summary.json` and `results_v10/per_seed_metrics.json`. The plotted convergence curves are in `v10_convergence.png`.

## Interpretation

1. The adjoint check passes numerically for each sparse matrix; this verifies implementation consistency of the stored linear operator and its transpose, **not** the real-world correctness of its acquisition geometry.
2. The fraction of voxels receiving any sampled ray is about 75.1% for each view count, exposing a coverage limitation. Zero-sensitivity voxels cannot be recovered from these measurements without assumptions/priors.
3. Increasing views from 9 to 21 does not improve the tested lesion metric under the selected SIRT iterations and synthetic noise setting. This is not evidence that additional views are generally unhelpful.
4. Projection RMSE values are in synthetic line-integral units and are not directly normalized across acquisition counts or comparable to V9 without accounting for the changed grid, phantom, sampling and iteration count.
5. A separate independent projector, geometry calibration, additional phantoms and properly modeled dose/noise are required before general conclusions.

## Recommended next experiment

Inspect detector field-of-view and ray-intersection coverage slice by slice; audit the ray sampling range and numerical quadrature; then perform a same-dose comparison on an independent forward operator with coverage-matched target ROIs. Only after these checks consider regularization or an independent ultrasound measurement.

## Test results

`python -m unittest discover -p 'test_*.py' -q` completed **34 tests, OK** in the build environment. Prior warnings in older suites about Radon-circle support and infinite PSNR still occur and were not concealed or addressed by V10.
