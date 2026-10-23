# V3 — controlled synthetic reconstruction ablation

## Protocol
30 paired phantom/noise seeds; 64×64 synthetic images; 21 limited-angle parallel-beam projections (−25° to +25°); 4 outer SART-plus-diffusion cycles. Every method receives the same X-ray observations and initial SART estimate. Diffusion schedules are identical, while guide sources vary.

## Mean results

| Method | SSIM ↑ | PSNR dB ↑ | Lesion MAE ↓ | Lesion contrast error ↓ | Projection RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| SART | 0.655 | — | 0.148 | 0.270 | 0.389 |
| X-ray-only | 0.689 | — | 0.144 | 0.265 | 0.390 |
| Correct US | 0.695 | — | 0.139 | 0.258 | 0.385 |
| Shifted US (5px diagonal) | 0.682 | — | 0.152 | 0.273 | 0.400 |
| Unrelated rotated synthetic US | 0.692 | — | 0.151 | 0.273 | 0.391 |
| Constant guide | 0.681 | — | 0.154 | 0.276 | 0.406 |

Full PSNR and localization data, paired bootstrap intervals and per-seed values are in `results_v3/summary.json` and `per_seed_metrics.json`; numbers in this report rounded to three decimals.

## Paired bootstrap vs X-ray-only, correct US
- SSIM mean gain: +0.00641; 95% percentile bootstrap CI [0.00621, 0.00660]
- Lesion MAE reduction: 0.00509; 95% CI [0.00485, 0.00533]
- Contrast-error reduction: 0.00688; 95% CI [0.00664, 0.00712]

These intervals quantify variability *within the artificial generator*, not efficacy on new modalities, scanners, patients, or datasets. The narrow intervals may reflect restricted phantom variation rather than robust generalization. No claims of clinical superiority should be drawn.

## Key caveats
- 2D parallel-beam Radon is not a physically faithful model of breast X-ray tomosynthesis.
- The ultrasound appearance surrogate is generated from the **same ground truth** used in reconstruction evaluation, creating a serious source of privileged information/data leakage. It does not model beamforming, sound speed, scattering, transducer aperture or acoustic propagation.
- Unrelated ultrasound is rotated so that its simulated lesion no longer aligns, but its appearance and anatomy remain synthetic and structurally related.
- The lesion localization score searches only near the **known** ground-truth lesion, so it is not a blind lesion-detection task.
- Projection residual is computed against noisy observations and must not be interpreted as proof of diagnostic value.
- Seed-to-seed changes are limited by a largely fixed lesion position, size, phantom family and acquisition geometry.
- Several metrics and controls were evaluated on the same seeds: there is no external held-out real imaging validation.
- Legacy v1/v2 tests can still warn on circular support or infinite PSNR; the v3 forward and residual operators enforce a circle mask.

## Next credible step
Vary lesion location, radius, contrast, missing angular views, random anatomical patterns and modality mismatch; tune on a disjoint training set and report separate holdout results. Next migrate to independently acquired or accepted public ultrasound and mammographic datasets (with proper pairing constraints), and add a physics-aware ultrasound simulation if feasible. Treat this as a coding/research feasibility exercise, not a clinical method.
