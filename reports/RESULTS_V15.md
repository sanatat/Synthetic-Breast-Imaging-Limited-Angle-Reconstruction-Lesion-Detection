# V15 — Synthetic hard-negative lesion-screening stress test

## Purpose
Stress-test the V14 blind peak detector with smaller lesions, contrast variation, heterogeneous backgrounds and localized hyperattenuating distractors. This is educational research code **not intended for clinical diagnosis**.

## Methods
- 12 independently seeded **paired** synthetic positive/negative backgrounds; each pair shares its background anatomy.
- 6 seed pairs for development, 6 separate pairs held out.
- Lesion radii randomly varied from 0.85 to 2.2 voxels along each axis and contrast from 0.17 to 0.44.
- Each pair contains 1–3 localized bright distractors and 3D smoothed tissue texture.
- Sparse cone-beam educational forward projector, 15 views across ±18°, Poisson photon-count proxy (30,000 incident counts per ray), 70 SIRT iterations.
- One candidate peak per reconstructed volume; **no reference lesion mask used for reconstruction or candidate detection**.
- Detection threshold fixed using the 95th percentile of *development-negative* candidate scores. All test metrics use this fixed threshold.
- Localization error uses ground-truth lesion center **only for retrospective evaluation**.

## Recorded outcomes

| Cohort | Positive | Hard-negative | True positive | False negative | False positive | True negative | Mean center error (voxels; all positives) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development | 6 | 6 | 3 | 3 | 1 | 5 | 2.98 |
| Held out | 6 | 6 | 2 | 4 | 0 | 6 | 4.27 |

Development negative 95th-percentile threshold = 0.102934. Held-out sensitivity = 2/6 (33.3%); false positive rate = 0/6 (0%). Sample sizes too small for stable estimates.

## Interpretation
Performance drops sharply with small variable-contrast lesions and bright confounding structures, relative to simpler V14 simulations. Detection in the current configuration is not robust. The one-peak method also cannot analyze multiple simultaneous lesions.

## Limitations
- Synthetically generated tissue and lesions; no real patients or clinical labels.
- Even the hard negatives are simplified intensity bumps, not realistic pathology.
- Forward and reconstruction operators match (inverse crime).
- Imaging dose, energy spectrum, scattering, blur, movement and geometry are not clinically calibrated.
- Development/test independence is by procedural seed, not by external dataset or domain.
- Only 6 held-out positive and 6 held-out negative samples; small-sample uncertainty is large.
- No real/independent ultrasound channel or multimodal reconstruction.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 python run_v15.py --seeds 12 --steps 70
python -m unittest test_v15 -v
```

Output: `results_v15/summary.json`, `per_seed_metrics.json`, `v15_example.png`.
