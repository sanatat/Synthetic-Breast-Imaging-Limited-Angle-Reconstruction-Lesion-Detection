"""V3 controlled synthetic ablations. Not real breast imaging or independent clinical evidence."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, shift, binary_dilation, maximum_filter
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage.transform import radon, iradon_sart
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_experiment import phantom, ultrasound_surrogate
from run_v2 import reconstruct

METHODS = ('sart', 'xray_only', 'ultrasound', 'us_shifted', 'us_unrelated', 'us_constant')


def circular_mask(n):
    yy, xx = np.ogrid[:n, :n]
    return (xx-(n-1)/2)**2 + (yy-(n-1)/2)**2 <= (n/2-2)**2


def synthesize(seed, n, n_angles, angle_extent, noise_fraction):
    rng = np.random.default_rng(1000 + seed)
    truth, lesion = phantom(n, seed + 12)
    mask = circular_mask(n)
    truth = truth * mask
    lesion = lesion & mask
    theta = np.linspace(-angle_extent, angle_extent, n_angles)
    clean = radon(truth, theta=theta, circle=True)
    sino = clean + rng.normal(0, noise_fraction * max(float(clean.max()), 1e-12), clean.shape)
    sart = np.zeros_like(truth)
    for _ in range(2):
        sart = np.clip(iradon_sart(sino, theta=theta, image=sart, relaxation=.12), 0, 1) * mask
    us = ultrasound_surrogate(truth, rng) * mask
    # Shift precisely the same observation; do NOT redraw noise.
    us_shift = shift(us, (5, -5), order=1, mode='constant', prefilter=False) * mask
    unrelated_truth, _ = phantom(n, seed+100012)
    unrelated_truth *= mask
    us_unrelated = np.rot90(ultrasound_surrogate(unrelated_truth, np.random.default_rng(900000+seed)), k=1) * mask
    return truth, lesion, mask, theta, sino, sart, us, us_shift, us_unrelated


def lesion_statistics(truth, x, lesion, sino, theta, mask):
    ring = binary_dilation(lesion, iterations=9) & ~binary_dilation(lesion, iterations=2) & mask
    if not ring.any():
        raise ValueError('Empty lesion background ring; choose larger phantom')
    target_contrast = float(np.mean(truth[lesion]) - np.mean(truth[ring]))
    contrast = float(np.mean(x[lesion]) - np.mean(x[ring]))
    # Localization by brightest local peak around target, not a full lesion detector.
    ys, xs = np.where(lesion)
    true_center = np.array([ys.mean(), xs.mean()])
    radius = 13
    yy, xx = np.indices(x.shape)
    roi = (yy-true_center[0])**2 + (xx-true_center[1])**2 <= radius**2
    roi &= mask
    peak_index = np.argmax(np.where(roi, gaussian_filter(x, .85), -np.inf))
    peak_center = np.array(np.unravel_index(peak_index, x.shape))
    # Projection residual includes noise; lower is not automatically better clinically.
    rmse = float(np.sqrt(np.mean((radon(x*mask, theta=theta, circle=True)-sino)**2)))
    return {'ssim':float(structural_similarity(truth,x,data_range=1)),
            'psnr_db':float(peak_signal_noise_ratio(truth,x,data_range=1)),
            'lesion_mae':float(np.mean(np.abs(x[lesion]-truth[lesion]))),
            'lesion_contrast_error':abs(contrast-target_contrast),
            'lesion_peak_offset_px':float(np.linalg.norm(peak_center-true_center)),
            'projection_rmse':rmse}


def paired_stats(records, metric, comparator='xray_only', target='ultrasound', boots=2000):
    by_method = {m:np.array([r[metric] for r in records if r['method']==m]) for m in METHODS}
    direction = 1 if metric in ('ssim', 'psnr_db') else -1
    # Positive improvement always means target is preferable.
    differences = direction*(by_method[target]-by_method[comparator])
    rng = np.random.default_rng(20261023)
    samples = rng.choice(differences, size=(boots,len(differences)), replace=True).mean(axis=1)
    ci=np.percentile(samples, [2.5,97.5])
    return {'metric':metric,'target':target,'comparator':comparator,
            'mean_improvement':float(differences.mean()),
            'bootstrap_95_ci':[float(ci[0]),float(ci[1])],
            'fraction_seed_improved':float(np.mean(differences>0)),
            'n':len(differences)}


def run(seeds=30, size=64, outer=4, n_angles=21, angle_extent=25, noise_fraction=.015, output='results_v3'):
    if seeds < 2 or size < 48 or outer < 1 or n_angles < 3:
        raise ValueError('Require seeds>=2, size>=48, outer>=1, angles>=3')
    out = Path(output); out.mkdir(parents=True,exist_ok=True)
    records=[]; example=None
    for seed in range(seeds):
        truth, lesion, mask, theta, sino, sart, us, shifted, unrelated = synthesize(seed, size, n_angles, angle_extent, noise_fraction)
        guides = {'xray_only':gaussian_filter(sart, 1.0)*mask,
                  'ultrasound':us,'us_shifted':shifted,'us_unrelated':unrelated,
                  'us_constant':np.full_like(us, float(us[mask].mean()))*mask}
        outputs={'sart':sart}
        for method, guide in guides.items():
            outputs[method]=reconstruct(sino,theta,sart,guide,outer=outer)*mask
        for method in METHODS:
            records.append({'seed':seed,'method':method,**lesion_statistics(truth, outputs[method],lesion,sino,theta,mask)})
        if seed == 0:
            example=[(truth,'Synthetic ground truth'),(us,'US appearance surrogate'),(sart,'SART')]+[(outputs[m],m.replace('_',' ')) for m in METHODS[1:]]
        if (seed+1)%10 == 0: print('Completed',seed+1,'/',seeds,flush=True)
    metrics=['ssim','psnr_db','lesion_mae','lesion_contrast_error','lesion_peak_offset_px','projection_rmse']
    summary={m:{k:{'mean':float(np.mean([r[k] for r in records if r['method']==m])),
                   'std':float(np.std([r[k] for r in records if r['method']==m],ddof=1))} for k in metrics} for m in METHODS}
    paired=[paired_stats(records,metric,target=method) for method in ('ultrasound','us_shifted','us_unrelated','us_constant') for metric in metrics]
    settings=dict(seeds=seeds,size=size,outer=outer,n_angles=n_angles,angle_extent=angle_extent,noise_fraction=noise_fraction)
    caveats=['2D circular parallel-beam Radon is not digital breast tomosynthesis.',
             'Ultrasound appearance surrogate is derived from same ground-truth phantom and may leak its structure.',
             'Unrelated US is rotated 90 degrees to move its lesion, but still shares synthetic generation rules and gross geometry.',
             'Lesion localization metric searches around known truth lesion; not blind clinical lesion detection.',
             'Bootstrap CI describes variability across synthetic seeds, not external generalization.',
             'Parameters not tuned on held-out realistic phantoms or clinical data.']
    (out/'per_seed_metrics.json').write_text(json.dumps(records,indent=2))
    (out/'summary.json').write_text(json.dumps({'settings':settings,'methods':summary,'paired_improvements_vs_xray_only':paired,'limitations':caveats},indent=2))
    fig, axes=plt.subplots(2,4,figsize=(13,7))
    for ax,(im,name) in zip(axes.ravel(),example):
        ax.imshow(im,cmap='gray',vmin=0,vmax=1);ax.set_title(name,fontsize=10);ax.axis('off')
    fig.tight_layout();fig.savefig(out/'v3_comparison.png',dpi=135);plt.close(fig)
    print('V3 controlled synthetic ablations; means across',seeds,'seeds:')
    for m in METHODS:
        d=summary[m]
        print(f'{m:19s} SSIM {d["ssim"]["mean"]:.3f}  lesion MAE {d["lesion_mae"]["mean"]:.3f}  contrast error {d["lesion_contrast_error"]["mean"]:.3f}  proj RMSE {d["projection_rmse"]["mean"]:.3f}')
    for p in paired:
        if p['target']=='ultrasound' and p['metric'] in ('ssim','lesion_mae','lesion_contrast_error'):
            print('Paired US improvement',p['metric'],round(p['mean_improvement'],5),'CI',p['bootstrap_95_ci'])
    print('Saved to',out.resolve())
    return {'settings':settings,'methods':summary,'paired_improvements_vs_xray_only':paired,'limitations':caveats}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',type=int,default=30);p.add_argument('--size',type=int,default=64)
    p.add_argument('--outer',type=int,default=4);p.add_argument('--angles',type=int,default=21)
    p.add_argument('--angle-extent',type=float,default=25);p.add_argument('--noise',type=float,default=.015)
    p.add_argument('--output',default='results_v3')
    a=p.parse_args()
    run(a.seeds,a.size,a.outer,a.angles,a.angle_extent,a.noise,a.output)
