"""V6: angle sensitivity and genuine total-variation proximal steps (2D toy tomography).
No clinical data, no real DBT/US. US is deliberately excluded from this geometry diagnostic.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter,binary_dilation, label
from skimage.transform import radon,iradon_sart
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import structural_similarity
from run_v4 import phantom_v4
from run_v3 import circular_mask
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def reconstruct(sino,theta,n,mask,method='tv',iterations=8,tv_weight=.035):
    x=np.zeros((n,n),dtype=float)
    for _ in range(iterations):
        x=np.clip(iradon_sart(sino,theta=theta,image=x,relaxation=.12),0,1)*mask
        if method=='tv':
            x=np.clip(denoise_tv_chambolle(x,weight=tv_weight,channel_axis=None),0,1)*mask
    return x

def evaluate(truth,lesion,pred,sino,theta,mask):
    ring=binary_dilation(lesion,iterations=7)&~binary_dilation(lesion,iterations=2)&mask
    gt_contrast=truth[lesion].mean()-truth[ring].mean()
    measured=pred[lesion].mean()-pred[ring].mean()
    yy,xx=np.where(lesion)
    target=np.array([yy.mean(),xx.mean()])
    # Oracle-local peak is a diagnostic using truth location; NOT lesion detection.
    y,x=np.indices(truth.shape)
    local=((y-target[0])**2+(x-target[1])**2<=12**2)&mask
    peak=np.array(np.unravel_index(np.argmax(np.where(local,gaussian_filter(pred,1),-np.inf)),pred.shape))
    # Shape proxy: weighted covariance of contrast-enhanced ROI; undefined if there is no signal.
    roi=((y-target[0])**2+(x-target[1])**2<=12**2)&mask
    background=float(np.median(pred[ring]))
    weights=np.maximum(pred-background,0)*roi
    if weights.sum()>1e-9:
        cy=(weights*y).sum()/weights.sum();cx=(weights*x).sum()/weights.sum()
        var_y=(weights*(y-cy)**2).sum()/weights.sum();var_x=(weights*(x-cx)**2).sum()/weights.sum()
        cov=(weights*(y-cy)*(x-cx)).sum()/weights.sum()
        ev=np.linalg.eigvalsh([[var_y,cov],[cov,var_x]])
        elongation=float(np.sqrt(max(ev[-1],1e-9)/max(ev[0],1e-9)))
    else:elongation=float('nan')
    return dict(ssim=float(structural_similarity(truth,pred,data_range=1)),lesion_mae=float(np.abs(pred[lesion]-truth[lesion]).mean()),contrast_error=float(abs(measured-gt_contrast)),projection_rmse=float(np.sqrt(np.mean((radon(pred,theta=theta,circle=True)-sino)**2))),oracle_peak_offset=float(np.linalg.norm(peak-target)),roi_elongation=elongation)

def run(seeds=12,size=64,views=21,extents=(15,25,45,75),noise=.015,iterations=8,tv_weight=.035,output='results_v6'):
    if seeds<2 or size<48 or views<5 or any(e<=0 or e>90 for e in extents):raise ValueError('Invalid input')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    mask=circular_mask(size);records=[];panels=[]
    for j in range(seeds):
        seed=800000+j
        truth,lesion,_=phantom_v4(size,seed)
        for extent in extents:
            theta=np.linspace(-extent,extent,views)
            clean=radon(truth,theta=theta,circle=True)
            # Same-seed noise generator per extent, reproducible comparison.
            rng=np.random.default_rng(seed+round(extent*100)+200000)
            sino=clean+rng.normal(0,noise*max(float(clean.max()),1e-9),clean.shape)
            for method in ('sart','tv'):
                pred=reconstruct(sino,theta,size,mask,method,iterations,tv_weight)
                records.append(dict(seed=seed,extent=float(extent),method=method,**evaluate(truth,lesion,pred,sino,theta,mask)))
                if j==0:panels.append((f'{method.upper()} ±{extent}°',pred))
    metrics=('ssim','lesion_mae','contrast_error','projection_rmse','oracle_peak_offset','roi_elongation')
    means={str(e):{m:{k:float(np.nanmean([r[k] for r in records if r['extent']==e and r['method']==m])) for k in metrics} for m in ('sart','tv')} for e in extents}
    # Paired comparisons by seed, TV vs SART at same extent.
    paired={}
    rng=np.random.default_rng(9234)
    for e in extents:
        paired[str(e)]={}
        for k in ('ssim','lesion_mae','contrast_error','oracle_peak_offset'):
            aa=np.array([r[k] for r in records if r['extent']==e and r['method']=='tv'])
            bb=np.array([r[k] for r in records if r['extent']==e and r['method']=='sart'])
            delta=(aa-bb)*(1 if k=='ssim' else -1)
            draws=rng.choice(delta,(1000,len(delta)),replace=True).mean(axis=1)
            paired[str(e)][k]={'improvement_mean':float(delta.mean()),'bootstrap_ci95':np.percentile(draws,[2.5,97.5]).tolist(),'fraction_better':float((delta>0).mean())}
    summary=dict(description='2D limited-angle parallel-beam sensitivity study with actual isotropic TV proximal steps; not DBT, not joint US reconstruction',parameters=dict(seeds=seeds,size=size,views=views,extents=list(extents),noise=noise,iterations=iterations,tv_weight=tv_weight),means=means,paired_tv_vs_sart=paired,limitations=['Comparison is synthetic only and not cone-beam digital breast tomosynthesis','TV uses SART alternation followed by a Chambolle TV denoiser; it is not an exact optimizer of the globally combined fidelity + TV objective','Angular extent is varied while number of views remains fixed; angular sampling density therefore also changes','Oracle lesion peak and contrast metrics use known ground-truth mask, not blinded detection','No ultrasound data are used: first isolate the X-ray reconstruction failure mode','No hyperparameter optimization on separate development set'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'per_seed_metrics.json').write_text(json.dumps(records,indent=2))
    fig,ax=plt.subplots(3,3,figsize=(10,10))
    images=[('Ground truth',truth)] + panels[:8]
    # Plot first seed only; truth variable is from last seed so fix using first seed.
    images[0]=('Ground truth',phantom_v4(size,800000)[0])
    for a,(name,im) in zip(ax.flat,images):a.imshow(im,cmap='gray',vmin=0,vmax=1);a.set_title(name);a.axis('off')
    fig.tight_layout();fig.savefig(out/'v6_angle_tv_comparison.png',dpi=140);plt.close(fig)
    print('V6: angle sweep and TV proximal baseline (synthetic 2D, NO US)')
    for e in extents:
        for m in ('sart','tv'):
            v=means[str(e)][m]
            print(f'±{e:2d}° {m:4s} SSIM={v["ssim"]:.3f} lesionMAE={v["lesion_mae"]:.3f} contrastErr={v["contrast_error"]:.3f} peakOffset={v["oracle_peak_offset"]:.2f}')
    print('Saved',out.resolve());return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=12);p.add_argument('--size',type=int,default=64);p.add_argument('--views',type=int,default=21);p.add_argument('--extents',type=float,nargs='+',default=[15,25,45,75]);p.add_argument('--noise',type=float,default=.015);p.add_argument('--iterations',type=int,default=8);p.add_argument('--tv-weight',type=float,default=.035);p.add_argument('--output',default='results_v6')
    run(**vars(p.parse_args()))
