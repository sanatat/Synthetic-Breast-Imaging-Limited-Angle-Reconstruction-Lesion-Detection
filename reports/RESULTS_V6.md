# V6 — Limited-angle X-ray baseline and TV stress test

## Purpose
Isolate the effect of angular range on lesion degradation before attempting multimodal reconstruction. **This experiment does not use ultrasound** and does not simulate clinical breast tomosynthesis.

## Methods
12 independent synthetic phantom seeds, 64×64 images, 21 equally spaced parallel-beam angles per range (±15°, ±25°, ±45°, ±75°), Gaussian noise proportional to maximum noiseless sinogram value (fraction .015), eight SART iterations with relaxation .12. TV baseline alternates SART updates and `skimage.restoration.denoise_tv_chambolle` (weight .035); this is a TV proximal heuristic rather than a solver proven to minimize a combined objective. No separate development set was used to tune these hyperparameters.

## Results (mean across 12 synthetic seeds)

| Angular range | Method | SSIM ↑ | Lesion intensity MAE ↓ | Contrast error ↓ | Oracle-local peak offset (px) ↓ |
|---|---|---:|---:|---:|---:|
| ±15° | SART | 0.602 | 0.156 | 0.236 | 4.87 |
| ±15° | TV prox | 0.637 | 0.169 | 0.246 | 4.53 |
| ±25° | SART | 0.651 | 0.136 | 0.202 | 2.99 |
| ±25° | TV prox | 0.706 | 0.152 | 0.215 | 2.20 |
| ±45° | SART | 0.719 | 0.129 | 0.143 | 1.09 |
| ±45° | TV prox | 0.862 | 0.142 | 0.154 | 0.80 |
| ±75° | SART | 0.782 | 0.049 | 0.041 | 1.28 |
| ±75° | TV prox | 0.958 | 0.074 | 0.075 | 0.81 |

## Interpretation
- Wider angular coverage improves synthetic reconstruction strongly under these chosen configurations. The number of views stays fixed, so view spacing changes as well; this is not an isolated geometry effect.
- TV proximal steps consistently improve global SSIM but **worsen lesion intensity MAE** in these configurations; smoothing/contrast loss is relevant to small targets.
- Oracle-local peak offset uses the true lesion center to restrict the search. It is **not** an independent detection metric.
- The shape proxy is stored in JSON; it can be influenced by background and is not a validated lesion boundary measure.

## Next step
Use development/holdout separation to tune TV weight only on development phantoms, then test lesion-aware fidelity metrics on unseen held-out phantoms. For a Radboud-relevant next milestone, implement a small 3D limited-angle cone-beam forward projector and physically separated ultrasound observation (not ultrasound generated from an image being reconstructed). Compare X-ray-only and guidance with registered, misregistered, and unrelated cues.

## Reproduction
`python run_v6.py --seeds 12`
`python -m unittest test_v6 -v`
Outputs: `results_v6/summary.json`, `per_seed_metrics.json`, `v6_angle_tv_comparison.png`.
