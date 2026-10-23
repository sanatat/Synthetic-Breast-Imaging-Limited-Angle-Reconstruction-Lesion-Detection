# V8: Limited-angle 3D divergent cone-beam feasibility study

## Scope
CPU-friendly synthetic phantom experiment. Unlike V7's stacked parallel-ray slices, this version explicitly creates a rotating point source, a 2D flat detector, and divergent ray paths across a full 3D voxel volume. Trilinear ray integration is represented as a sparse system matrix and the very same matrix transpose is used for backprojection.

**Not** a scanner-calibrated digital breast tomosynthesis system. No detector blur, scatter, polychromatic attenuation, anatomical breast compression, patient scans, or independent ultrasound. Source-to-isocenter and source-to-detector distances are arbitrary simulation units.

## Methods
- 3D synthetic volume `(y,z,x) = (16,24,24)` with known lesion mask (oracle).
- Source follows a short arc about the volume, views at ±18° or ±40°; 11 views, 2D detector.
- Sparse ray-driven trilinear forward matrix; repeated normalized algebraic updates (SIRT-like).
- Optional 3D Chambolle TV denoising after algebraic updates (heuristic, not an exact total-variation optimizer).
- Same random phantom for both angle sweeps. Two random seeds in this fast initial smoke benchmark.

## Initial results (mean over two synthetic seeds)

| Angular range | Method | Lesion MAE ↓ | Whole-volume MAE ↓ | Projection RMSE ↓ |
|---|---|---:|---:|---:|
| ±18° | SIRT | 0.4070 | **0.0717** | **0.0729** |
| ±18° | SIRT+TV | **0.4065** | 0.0906 | 0.1637 |
| ±40° | SIRT | 0.3333 | **0.0839** | **0.0814** |
| ±40° | SIRT+TV | **0.3155** | 0.0939 | 0.2030 |

**Interpretation:** broader angle coverage helped the simulated lesion on this limited test; this cannot be inferred for real clinical DBT geometry. TV marginally reduced lesion MAE but made projection agreement and whole-volume MAE worse. This is a tradeoff, not an unqualified improvement. Two seeds are much too few to infer robustness.

## Reproducibility

```bash
python run_v8.py --seeds 2
python -m unittest test_v8 -v
```

Results: `results_v8/summary.json`, `per_seed_metrics.json`, `v8_conebeam_comparison.png`.

## Next validation work
1. Increase sample count and compare equal projection noise / exposure assumptions.
2. Characterize geometry and ray model with unit-test phantoms (point, uniform sphere) and calibration.
3. Evaluate true lesion detection without using ground-truth ROI, plus physical dimensions and nonrigid misregistration.
4. Study image-quality vs data-fidelity tradeoff over regularization strength, and use independent real ultrasound measurements when accessible.
