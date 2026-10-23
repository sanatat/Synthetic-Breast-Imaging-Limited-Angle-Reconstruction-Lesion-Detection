# V4 randomized lesion stress test — 23 October 2026

Thirty deterministic *new synthetic* test seeds, random lesion location, size, eccentricity and contrast; settings retained from V3 without V4 tuning. All conditions share the same limited-angle projections within each test seed.

| Method | Mean SSIM ↑ | Lesion MAE ↓ | Contrast error ↓ |
|---|---:|---:|---:|
| SART | 0.657 | 0.133 | 0.198 |
| X-ray-only | 0.687 | 0.125 | 0.189 |
| US-guided | 0.698 | 0.121 | 0.184 |
| Misregistered US | 0.675 | 0.136 | 0.201 |
| Unrelated US | 0.694 | 0.136 | 0.201 |
| Constant guide | 0.668 | 0.140 | 0.205 |

Paired US-guided minus X-ray-only mean improvement: SSIM +0.01040 (bootstrap 95% percentile interval +0.01018 to +0.01060), lesion MAE reduction 0.00418 (bootstrap 95% interval 0.00295 to 0.00526), better lesion MAE in 28/30 synthetic cases.

**Interpretation**: US guidance modestly improves the metrics in this synthetic scenario. Unrelated US also increases global SSIM relative to the X-ray-only reconstruction while worsening lesion MAE, warning against claiming useful lesion recovery from SSIM alone.

**Important limitations**: 2D parallel-beam Radon is not digital breast tomosynthesis. The US-like image is generated from exact ground truth and therefore can leak privileged information. An unrelated synthetic phantom does not establish generalizability. No human imaging, physical acoustic propagation, real-world registration error distribution, blind lesion localization, clinical diagnosis, or external benchmark is present. Bootstrap CI covers variability in the synthetic seed distribution only. The phrase 'held-out' here denotes fresh deterministic seeds with fixed settings, not a genuine external or patient-level test set. 

See `results_v4/summary.json` and `results_v4/per_seed_metrics.json` for precision and settings.
