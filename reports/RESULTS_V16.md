# V16 — Held-out Synthetic Lesion Candidate Detection Ablation

## Scope
Exploratory, synthetic, non-clinical experiment. Three single-candidate detection strategies are evaluated on reconstructed limited-angle cone-beam phantom scans with photon-count noise and hard-negative distractors.

## Design
- 12 paired synthetic phantom seeds (12 positive + 12 hard negatives).
- Six paired seeds for development and six paired seeds held out. Negative development scans determine each detector's 95th-percentile score threshold; the held-out set is not used to choose thresholds.
- Identical 15-view cone-beam operator, 70 SIRT iterations and photon noise across methods; predictions depend only on reconstructed scans.
- A true positive requires a score strictly exceeding the development threshold **and** a predicted maximum within 3 voxels of the known positive lesion center; the lesion annotation is used only after prediction to score localization.
- Three methods: Gaussian-smoothed peak intensity; local contrast (narrow minus broad Gaussian smooth); maximum scale-normalized negative Laplacian-of-Gaussian response across three scales.

## Held-out results
| Method | Localized TP / positives | FP / negatives | Mean candidate-to-lesion center error |
|---|---:|---:|---:|
| Intensity | 3 / 6 | 1 / 6 | 3.43 voxels |
| Local contrast | 1 / 6 | 0 / 6 | 4.18 voxels |
| Multiscale LoG | 1 / 6 | 0 / 6 | 6.30 voxels |

The high threshold yields low sensitivity for local contrast and LoG. The intensity baseline recovered more of these simulated lesions but produced a false alarm. These are **tiny synthetic samples**, not an estimate of clinical performance. The displayed FROC-style curve uses one maximum candidate per volume and therefore is not a full multi-mark FROC study.

## Limitations
- The same projector generates and reconstructs the observations (inverse crime).
- Paired, synthetic hard-negative textures may not resemble real tissue.
- One maximum per volume is insufficient for a complete lesion detection study.
- 3-voxel localization radius is an arbitrary simulation tolerance.
- Development/held-out split is independent by seed but not by clinical patient.
- No ultrasound, real patient images, physical calibration, or diagnostic validation.

## Reproduce
```bash
OPENBLAS_NUM_THREADS=1 python run_v16.py --seeds 12 --steps 70
python -m unittest test_v16 -v
```
Outputs: `results_v16/summary.json`, `per_seed_metrics.json`, `v16_froc_and_example.png`.
