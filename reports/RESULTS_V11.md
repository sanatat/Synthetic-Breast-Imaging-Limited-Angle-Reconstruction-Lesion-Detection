# V11 — Cone-beam detector field-of-view audit

**Study type:** Synthetic computational feasibility and numerical QA, not calibrated breast tomosynthesis.

## Question
Does the missing voxel sensitivity in V10 reflect the detector field of view, rather than necessarily a failure of the iterative reconstruction?

## Setup
- Volume: 12 × 18 × 18 voxels (y,z,x)
- 15 projections across ±18 degrees
- Identical source/detector distances and ray sampling as V8 (`system_matrix`)
- Detector dimensions (u × v): 14×10, 22×16, and 30×24 pixels
- Coverage defined as positive column-sum sensitivity, with threshold 1e-8

## Observations

| Detector | Covered voxels | Unobserved voxels | Interior coverage |
|---|---:|---:|---:|
| 14×10 | 39.6% | 2,348 | 54.3% |
| 22×16 | 75.1% | 968 | 83.3% |
| 30×24 | 100.0% | 0 | 100.0% |

Widths and heights changed simultaneously, so these results cannot isolate their individual contributions. They also do not establish stable image recoverability: nonzero sensitivity is necessary but insufficient.

## Outputs
- `results_v11/summary.json`: per-detector 3D and per-slice coverage profiles, sensitivity summaries, and numerical adjoint checks
- `results_v11/v11_sensitivity_maps.png`: sensitivity and visibility maps
- `results_v11/v11_coverage_profiles.png`: per-slice coverage for each axis

## Interpretation and limitations
The V10 detector dimensions (22×16 at this volume size) leave 24.9% of voxels unseen. Enlarging the detector can eliminate this geometric truncation in the simplified forward model. **This does not remove limited-angle ill-conditioning or prove lesion recovery.** Detector width and height were varied together. The same approximate discrete projector is used for the forward model and the audit. No measured breast imaging, ultrasound, photon statistics, realistic tissue attenuation, scatter, or calibrated scanner geometry were used.

## Reproduction
```bash
python run_v11.py
python -m unittest test_v11 -v
```
