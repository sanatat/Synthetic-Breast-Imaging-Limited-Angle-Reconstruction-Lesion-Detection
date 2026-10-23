# V14 — Photon-counting proxy and lesion candidate evaluation

## Motivation
Previous versions relied on additive Gaussian noise and reference-mask-centric lesion metrics. V14 uses a monoenergetic Beer–Lambert Poisson-count proxy and scores a single automatically selected 3D candidate without using the reference mask during inference.

## Setup
- Educational sparse 3D divergent-ray matrix from V8: volume 12 × 18 × 18 voxels, 15 views over ±18°, detector 30 × 24.
- Fixed 30,000 incident photon counts per ray, arbitrary attenuation scale 0.035 per phantom intensity × ray length.
- Photon counts sampled as Poisson(I0 exp(-attenuation × ray integral)); line integrals recovered using the negative logarithm with a half-count numerical floor.
- SIRT: 80 iterations, reconstruction and data generation use the same sparse ray model (**inverse crime still present**).
- 16 seeded synthetic positive phantoms; 16 synthetic negative counterparts with the lesion contrast removed.
- Seed-disjoint development/test split: 8 positive + 8 negative in each set.
- Blind candidate = location of maximum 3D high-pass response on reconstructed image; no reference lesion ROI during prediction.
- Score threshold fixed using 95th percentile of **development negative** scores only. Reference lesion masks used for retrospective evaluation only.

## Held-out results (8 positive, 8 negative)
- Candidate-positive sensitivity at selected threshold: **8/8**.
- False-positive rate: **0/8**.
- Mean candidate localization error (all 8 positive volumes): **2.217 voxels**.
- Mean reference-mask lesion MAE: **0.3754**.
- Mean projection residual RMSE, pooled test volumes: **0.1713**.

These results are exploratory. With only 8 positives and 8 negatives in the test set, 100% observed classification does **not** establish reliable sensitivity/specificity. The negatives share anatomical backgrounds with the positives, making the task much simpler than breast lesion detection. All phantoms are computationally generated; no external independent physical imaging model, real patient data, or independent ultrasound is used. The photon flux and attenuation values are illustrative and **not** linked to a clinical exposure protocol.

## Meaningful next checks
- Hard negatives: high-density normal structures and variable anatomy; controls sharing the same lesion-like contrast but different morphology.
- Subject-independent validation with larger phantom diversity and confidence intervals.
- Evaluate lesion detection using free-response ROC and multiple candidate marks per volume rather than a single global maximum.
- Calibrate attenuation/dose/scatter models or use public DBT projection data where legitimately available.
- Avoid any claim of clinical accuracy or detection performance based on this V14 benchmark.

## Reproducibility
`OPENBLAS_NUM_THREADS=1 python run_v14.py --seeds 16 --steps 80`

`python -m unittest test_v14 -v`

Artifacts: `results_v14/summary.json`, `results_v14/per_seed_metrics.json`, `results_v14/v14_reconstruction.png`.
