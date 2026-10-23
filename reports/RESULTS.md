# Results — baseline pilot (4 seeds)

This is an exploratory **2D** pilot with synthetic images, not a clinical result.

| Method | SSIM (mean ± std) | PSNR (dB, mean ± std) | Lesion MAE (mean ± std) |
|---|---:|---:|---:|
| `fbp` | 0.331 ± 0.007 | 17.84 ± 0.15 | 0.146 ± 0.013 |
| `sart` | 0.659 ± 0.008 | 21.50 ± 0.23 | 0.164 ± 0.009 |
| `xray_guided_diffusion` | 0.647 ± 0.010 | 21.53 ± 0.23 | 0.192 ± 0.009 |
| `ultrasound_guided_diffusion` | 0.647 ± 0.010 | 21.55 ± 0.23 | 0.186 ± 0.010 |
| `misregistered_ultrasound` | 0.639 ± 0.010 | 21.48 ± 0.22 | 0.195 ± 0.009 |

## Interpretation

In this small experiment, SART exceeded FBP on SSIM. Ultrasound-guided diffusion did **not** clearly beat the SART reference and lesion error increased compared with SART. Introducing a 5-pixel mismatch generally worsened reconstruction. These trends are limited to four synthetic examples; no inference about clinical performance is justified.

## Next experiments

1. Evaluate angular span, noise level and SART iteration count with seeds held out.
2. Improve independent ultrasound forward modelling to avoid using the same ground truth as a near-perfect structural guide.
3. Implement physically motivated multimodal regularization of projection-domain reconstruction rather than post-processing only.
4. Test correct and incorrect anatomical edges, lesion detectability and registration error.
