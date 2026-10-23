# V5 — Controlled synthetic edge-guided reconstruction

**Status:** Exploratory computational pilot. No patients, real X-ray images, acoustic propagation, or clinical claims.

## Method

We interleave SART steps (measured projection-data update) with a four-neighbor anisotropic smoothing operator. Edge conductance is computed from the guidance image. **This is not a mathematical implementation of Directional Total Variation**; it is a simpler controlled edge-guided baseline inspired by structural-prior literature. Ultrasound observations are generated from the reference phantom, so an information advantage is inherent and must not be interpreted as proof of feasibility with real modalities.

## Mean results, 30 deterministic randomized-phantom/noise trials

| Method | SSIM ↑ | Lesion MAE ↓ | Contrast error ↓ | Oracle ROI peak offset (px) ↓ |
|---|---:|---:|---:|---:|
| SART | 0.656 | 0.131 | 0.201 | 3.71 |
| X-ray isotropic smoothing | 0.681 | 0.135 | 0.207 | 3.29 |
| X-ray directional smoothing | 0.686 | 0.126 | 0.197 | 3.12 |
| US correct directional | **0.694** | **0.119** | **0.190** | **2.89** |
| US shifted | 0.676 | 0.134 | 0.206 | 3.33 |
| US unrelated | 0.691 | 0.134 | 0.206 | 3.23 |
| US lesion erased | 0.691 | 0.135 | 0.207 | 3.29 |
| US partial coverage | 0.687 | 0.126 | 0.197 | 2.93 |

Against X-ray directional, the correct-US condition improved mean SSIM by 0.00796 (paired bootstrap 95% CI [0.00774, 0.00820]); mean lesion MAE decreased by 0.00657 (paired bootstrap 95% CI for improvement [0.00529, 0.00799]). These intervals measure within-generator variation only, not clinical external validity. The unrelated-US condition also raises SSIM, but worsens lesion MAE; SSIM by itself is not a reliable indicator of preserved lesions.

## Caveats / reproducibility

- The forward model is **2D parallel-beam Radon** over ±25°, not three-dimensional cone-beam digital breast tomosynthesis.
- US is a synthetic image surrogate **derived from reference truth**, not physical ultrasound simulation. This is privileged information and a major confound.
- Lesion-erasure stress test uses the known phantom lesion mask by design. The peak-offset metric searches inside a ground-truth-centered ROI; it is **not blind detection**.
- No patient data, acquisition model calibration, dose/noise validation or transfer demonstration.
- Hyperparameters here are illustrative, not cross-validated on a separately sampled training set.
- All results saved as raw per-seed metrics and paired bootstrap comparisons in `results_v5/`.
- Prior version tests still emit warnings for edge cases; the full 17-test suite passed in the available runtime.

## Next credible research step

Use genuinely independent modalities acquired on matched specimens or a physical forward simulation for both X-ray and ultrasound, train/tune on separate samples, then test on an untouched set. Include a 3D DBT projection geometry and blinded lesion detection.
