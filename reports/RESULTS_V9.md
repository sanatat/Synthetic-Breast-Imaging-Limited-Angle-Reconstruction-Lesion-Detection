# V9 — View count and TV regularization selection in a synthetic cone-beam prototype

## Experimental status
Exploratory CPU-only 3D phantom study. The ray-driven model has a moving divergent source and a 2D detector, but is **not** a calibrated digital breast tomosynthesis system. No patient images or measured ultrasound data were used. It is not a lesion detector.

## Design
- 3D volume: 16 × 24 × 24; angular range ±18°, 9/15/21 projection views
- 25 SIRT iterations; TV denoising interleaved as heuristic splitting (not exact variational TV)
- Development seeds: 2 (91000–91001); untouched evaluation seeds: 3 (99000–99002)
- TV weights: 0, 0.002, 0.008
- Gaussian projection perturbation: 0.5% of max noiseless measurement
- Model selection on development seeds only: minimum *known-lesion* mean absolute error, subject to mean projection RMSE <= 1.35× the zero-TV baseline
- Oracle binary lesion metrics use the known true lesion location and true lesion contrast; the results cannot be interpreted as blind detection performance.

## Held-out mean results

| Views | Method chosen on development | TV weight | Lesion MAE ↓ | Projection RMSE ↓ | Oracle binary centroid error (voxels) ↓ |
|---|---|---:|---:|---:|---:|
| 9 | SIRT | 0 | 0.4158 | 0.0835 | 0.899 |
| 15 | SIRT | 0 | 0.4184 | 0.0857 | 1.536 |
| 21 | SIRT | 0 | 0.4198 | 0.0861 | 1.391 |

TV weight 0.002 slightly improved the *development* lesion intensity error in all view settings but increased projection RMSE by about 35–39%, exceeding the 35% tolerance. Thus all three view settings selected **zero TV**. Weight 0.008 further increased projection RMSE and did not improve development lesion error.

Increasing the number of views **did not** improve held-out lesion MAE with these fixed SIRT iterations and ray normalization. This may indicate differences in solver scaling, discretization, noise, and parameter sensitivity rather than any universal claim that more views are harmful. Cross-angle normalized exposure/noise and tuned iteration/stopping rules require further study.

## Scientific cautions
1. Extremely small held-out set; no uncertainty or generalization claim is justified.
2. The 3D binary lesion evaluation is anchored on ground-truth coordinates (oracle ROI); not a clinical segmentation or localization algorithm.
3. Detector resolution, interpolation, acquisition trajectory and physics are simplified. No scatter, detector response, attenuation spectrum, or dose calibration.
4. TV was implemented as denoising inserted into iterative reconstruction; it does not solve an exact joint data-fidelity/TV objective.
5. The objective used for hyperparameter selection requires known synthetic truth and should never be presented as applicable to real unlabelled patient scans.
6. Projection RMSE and reconstruction error are not directly comparable numerically. The constraint is a prespecified research heuristic, not a clinical performance standard.

## Reproduction
```bash
python run_v9.py --dev-seeds 2 --test-seeds 3
python -m unittest test_v9 -v
```
See `results_v9/summary.json`, `results_v9/per_seed_metrics.json`, and `results_v9/v9_reconstruction_comparison.png`.

## Next technically justified work
Verify forward/adjoint operator and numerical conditioning, enforce dose-matched view comparisons, inspect ray coverage and voxel sensitivity, tune reconstruction iterations on development data, and only then pursue edge-preserving optimization and registered independently acquired ultrasound.
