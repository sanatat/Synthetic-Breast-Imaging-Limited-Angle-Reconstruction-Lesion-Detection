# Version 2: interleaved SART / guided regularization

All results are **2D synthetic-only**, averaged across four seeds. This is *not* breast DBT or acoustic-wave simulation, and the US surrogate is generated from the same ground truth. That information dependency may make improvements optimistic. No patient images or validation of lesion detection are involved.

| Method | SSIM ↑ | PSNR dB ↑ | Lesion MAE ↓ | Contrast error ↓ | Projection RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| Initial SART | 0.659 | 21.50 | 0.164 | 0.309 | 0.539 |
| X-ray-only iterative | 0.706 | 22.26 | 0.163 | 0.299 | 0.548 |
| US-guided iterative | 0.715 | 22.37 | 0.156 | 0.290 | 0.542 |
| 5-pixel misregistered US | 0.698 | 22.23 | 0.169 | 0.306 | 0.556 |

The differences are descriptive for four synthetic instances; no statistical significance is claimed. The US guided variant has better mean SSIM and lesion MAE than the X-ray-only control in this setup, but the advantage degrades with misregistration. The projection residual of the guided result is **higher than initial SART** (0.542 vs 0.539): the method alternates SART with regularization and does not enforce exact consistency. The residual is included explicitly to prevent misleading claims.

**Limitations:** 2D parallel-beam limited-angle Radon does not model DBT cone-beam geometry, detector physics, attenuation calibration, or 3D tissue. US is an image-like surrogate, not real RF data, beamforming, speed-of-sound modelling, speckle physics or probe geometry. True clinical lesion detection is not measured. Lesion MAE uses the known ground-truth mask (evaluation only); it is not an automated segmentation score. No train/test separation beyond random seeds; parameters are fixed and not tuned on a separate validation set. Do not market this as a clinical imaging method.

Recommended follow-up: broader phantom distributions, holdout testing, projection noise/angle sweeps, realistic forward models, simulated registration errors in multiple directions, and real open data if licensing allows.
