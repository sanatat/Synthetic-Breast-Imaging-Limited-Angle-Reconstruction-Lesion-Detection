# V12: full-FOV cone-beam view-count audit

**Status:** CPU-tested synthetic numerical experiment; not a calibrated digital breast tomosynthesis system.

## Objective
Test whether the previously observed failure of increasing view counts to improve lesion intensity error was solely caused by incomplete detector field of view.

## Configuration
- Phantom: two deterministic independent synthetic seeds, `(y,z,x)=(12,18,18)`.
- Sparse ray-sampled divergent-source / 2D flat-detector geometry, source sweep ±18°.
- Detector: 30×24 cells; 42 samples/ray.
- Views: 9, 15, 21; normalized SIRT, 30 iterations (also saved at iterations 5 and 15).
- Synthetic additive Gaussian noise with constant relative amplitude per measured ray. This is **not** a fixed-total-dose model.
- All metrics based on known synthetic truth. Oracle lesion ROI metrics are not blind diagnostic measures.

## Observed results (means over two seeds, 30 iterations)

| Views | Voxel visibility | Oracle lesion MAE | Projection RMSE |
|---:|---:|---:|---:|
| 9 | 100% | 0.3836 | 0.0811 |
| 15 | 100% | 0.3862 | 0.0805 |
| 21 | 100% | 0.3876 | 0.0794 |

## Interpretation
The larger detector eliminates the zero-sensitivity voxel problem in these settings, but more views do not improve the oracle lesion MAE. Projection consistency improves slightly. Coverage is not equivalent to independent information or well-conditioned inversion: limited angular diversity remains. The results do **not** establish that more projection views are unhelpful generally.

## Critical limitations
- Forward data generation and inversion use the same projector (**inverse crime**).
- Tiny synthetic sample; insufficient for reliable statistical conclusions.
- Simulated noise is Gaussian and normalized per ray, not realistic dose-/photon-dependent noise.
- Ray sampling and SIRT normalization are approximations; none of this is clinically calibrated DBT.
- No ultrasound, motion, registration or measured breast imaging data.
- The system-matrix adjoint check tests numerical consistency, not correctness of physical model.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 python run_v12.py --seeds 2
python -m unittest test_v12 -v
```

Outputs: `results_v12/summary.json`, `per_seed_metrics.json`, `v12_convergence.png`, `v12_reconstructions.png`.

## Suggested follow-up
A held-out projector/discretization for data generation, higher-precision geometry tests, more seeds, and a photon-statistics acquisition simulator; assess convergence at greater SIRT iterations and quantify depth resolution separately from lesion intensity.
