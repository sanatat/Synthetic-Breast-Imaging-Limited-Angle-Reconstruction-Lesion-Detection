# V13: SIRT Convergence and Forward-Model Discretization Mismatch

**Scope:** CPU-only exploratory synthetic cone-beam-like geometry. Not calibrated breast tomosynthesis and not clinical evidence.

## Protocol
- 3D phantom: 12 × 18 × 18 voxels; two independent random seeds.
- Cone-beam-like divergent rays: 9, 15, or 21 projections across ±18°; 30 × 24 detector pixels.
- SIRT: 15, 30, 60, 120 iterations, same stopping points for all views.
- Two experiment arms: **matched** synthetic measurements from 42-sample ray integration and **quadrature-mismatched** measurements from 77-sample integration. Both are reconstructed using the 42-sample system matrix.
- Gaussian projection noise is independently seeded and scaled to clean projection peak. This is **not** a physical dose/Poisson model.
- Lesion MAE is measured with a ground-truth oracle mask; this is not automated lesion detection.

## Quantitative results at 120 iterations

| Simulation arm | Projections | Lesion MAE ↓ | Volume MAE ↓ | Projection RMSE ↓ |
|---|---:|---:|---:|---:|
| Matched | 9 | 0.3640 | 0.0636 | 0.0613 |
| Matched | 15 | 0.3688 | 0.0640 | 0.0610 |
| Matched | 21 | 0.3710 | 0.0645 | 0.0608 |
| Quadrature mismatch | 9 | 0.3638 | 0.0637 | 0.0615 |
| Quadrature mismatch | 15 | 0.3688 | 0.0640 | 0.0610 |
| Quadrature mismatch | 21 | 0.3711 | 0.0645 | 0.0608 |

At nine views (matched), mean lesion MAE was 0.4010 at 15 iterations, 0.3908 at 30, 0.3791 at 60, and 0.3640 at 120. Thus more iterations improved the lesion error over the tested interval, but we have not established convergence at 120 iterations.

Discretization mismatch was nonzero: relative Frobenius differences for projection operators were 0.1164 / 0.1127 / 0.1114 for 9 / 15 / 21 views. However, the measured reconstruction differences were very small in this pilot. **Different quadrature is not equivalent to independent image formation physics**; this only partially addresses inverse crime.

## Interpretation
- Increased iterations reduce the oracle lesion intensity error in these settings, even when additional views do not.
- Increasing views from 9 to 21 marginally decreases the measurement residual but does not improve oracle lesion MAE in these *two* examples. Do not extrapolate this result to clinical tomosynthesis or conclude that adding views harms reconstruction in general.
- A changing sample count does not remove model assumptions shared by the forward/inverse operators. A real inverse-crime test would include independent geometry/physics and ideally real measurement data.

## Files
- `results_v13/summary.json`: settings, operator diagnostics, aggregated metrics
- `results_v13/per_seed_metrics.json`: per-seed data
- `results_v13/v13_convergence.png`: full convergence comparison
- `results_v13/v13_reconstructions.png`: two orthogonal slices for mismatched arm

## Next step
Evaluate a physically different forward operator, fixed-total-dose Poisson statistics, independent phantom distribution, and blind lesion localization with larger sample counts. Preserve negative findings and model assumptions.
