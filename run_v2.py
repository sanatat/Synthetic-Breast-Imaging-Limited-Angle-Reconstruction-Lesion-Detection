"""CPU-only v2: SART projection consistency interleaved with edge-aware regularization.
Not physical DBT/ultrasound, not clinical; US is a truth-derived image surrogate.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import iradon_sart, radon
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_experiment import phantom, ultrasound_surrogate, xray_recon


def edge_diffusion(image, guide, steps=2, rate=.12, scale=.035):
    """Edge-directed 4-neighbor smoothing (guide governs conductance)."""
    g = gaussian_filter(guide, 0.65)
    hx = np.exp(-np.square(np.diff(g, axis=1)/scale))
    hy = np.exp(-np.square(np.diff(g, axis=0)/scale))
    out = image.copy()
    for _ in range(steps):
        fx = hx * np.diff(out, axis=1)
        fy = hy * np.diff(out, axis=0)
        div = np.zeros_like(out)
        div[:, :-1] += fx
        div[:, 1:] -= fx
        div[:-1, :] += fy
        div[1:, :] -= fy
        out = np.clip(out + rate * div, 0, 1)
    return out


def reconstruct(sino, angles, initial, guide=None, outer=8, scale=.035):
    """Interleaved projection-data SART updates and edge-aware regularization.
    X-ray-only baseline uses a fixed, Gaussian-smoothed SART guide (not ground truth).
    US condition uses simulated-US guide. Both have same smoothing schedule.
    """
    x = initial.copy()
    for _ in range(outer):
        x = np.clip(iradon_sart(sino, theta=angles, image=x, relaxation=.08),0,1)
        if guide is not None:
            x = edge_diffusion(x, guide, steps=2, rate=.12, scale=scale)
    return x


def evaluate(truth, x, lesion, sino, angles):
    normal = np.ones_like(x, dtype=bool)
    # Evaluate a local annulus instead of averaging all heterogeneous breast tissue.
    from scipy.ndimage import binary_dilation
    ring = binary_dilation(lesion, iterations=9) & ~binary_dilation(lesion, iterations=2)
    if not np.any(ring): ring = ~lesion
    lesion_mean=float(np.mean(x[lesion])); bg=float(np.mean(x[ring]))
    gt_contrast=float(np.mean(truth[lesion])-np.mean(truth[ring]))
    pred_contrast=lesion_mean-bg
    residual=radon(x,theta=angles,circle=True)-sino
    return {'ssim':float(structural_similarity(truth,x,data_range=1)),
            'psnr_db':float(peak_signal_noise_ratio(truth,x,data_range=1)),
            'lesion_mae':float(np.mean(np.abs(x[lesion]-truth[lesion]))),
            'lesion_contrast':pred_contrast,
            'lesion_contrast_error':abs(pred_contrast-gt_contrast),
            'projection_rmse':float(np.sqrt(np.mean(residual**2)))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,default=4)
    ap.add_argument('--size',type=int,default=96)
    ap.add_argument('--output',default='results_v2')
    ap.add_argument('--outer',type=int,default=8)
    args=ap.parse_args()
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    records=[];visual=None
    for seed in range(args.seeds):
        rng=np.random.default_rng(1000+seed)
        truth,lesion=phantom(args.size,seed+12)
        angles,sino,fbp,sart=xray_recon(truth,rng)
        us=ultrasound_surrogate(truth,rng)
        shifted=ultrasound_surrogate(truth,rng,misregistration=5)
        x_guide=gaussian_filter(sart,1.0)
        results={'sart_initial':sart,
                 'xray_only_iterative':reconstruct(sino,angles,sart,x_guide,args.outer),
                 'ultrasound_guided_iterative':reconstruct(sino,angles,sart,us,args.outer),
                 'misregistered_ultrasound_iterative':reconstruct(sino,angles,sart,shifted,args.outer)}
        for key,img in results.items():records.append({'seed':seed,'method':key,**evaluate(truth,img,lesion,sino,angles)})
        if seed==0:visual=(truth,us,sart,*results.values())
    methods=list(dict.fromkeys(r['method'] for r in records))
    keys=['ssim','psnr_db','lesion_mae','lesion_contrast','lesion_contrast_error','projection_rmse']
    summary={method:{key:{'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0}
                    for key in keys for vals in [[r[key] for r in records if r['method']==method]]} for method in methods}
    (dest/'per_seed_metrics.json').write_text(json.dumps(records,indent=2))
    (dest/'summary.json').write_text(json.dumps({'seeds':args.seeds,'outer':args.outer,'limitations':'Synthetic 2D Radon, non-physical US surrogate; phantom-derived US can leak truth; no clinical evidence','methods':summary},indent=2))
    fig,axes=plt.subplots(2,3,figsize=(12,8));
    ims=[visual[0],visual[1],visual[2],visual[4],visual[5],visual[6]]
    titles=['Ground truth','US surrogate','Initial SART','X-ray only iterative','US-guided iterative','US shifted 5px']
    for ax,im,title in zip(axes.ravel(),ims,titles):
        ax.imshow(im,cmap='gray',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(dest/'v2_comparison.png',dpi=150);plt.close(fig)
    print('V2 mean over',args.seeds,'seeds; synthetic feasibility only')
    for method in methods:
        m=summary[method];print(f'{method:38} SSIM {m["ssim"]["mean"]:.3f}  PSNR {m["psnr_db"]["mean"]:.2f}  lesionMAE {m["lesion_mae"]["mean"]:.3f}  contrastErr {m["lesion_contrast_error"]["mean"]:.3f}  projRMSE {m["projection_rmse"]["mean"]:.3f}')
    print('Saved to',dest.resolve())

if __name__=='__main__':main()
