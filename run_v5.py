"""V5 exploratory prior-informed anisotropic diffusion with SART data steps.
NOT a validated DTV solver, clinical DBT model, or physical ultrasound simulation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, shift
from skimage.transform import iradon_sart, radon
from skimage.metrics import structural_similarity
from run_v4 import synthesize, circular_mask
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=('sart','xray_isotropic','xray_directional','us_directional','us_shifted','us_unrelated','us_lesion_erased','us_missing_coverage')

def guided_step(x, guide, rate=.11, edge_scale=.055, directional=True, support=None):
    """Stable 4-neighbor anisotropic diffusivity controlled by guide gradients.
    This is NOT a numerical directional total variation optimizer.
    """
    guide=gaussian_filter(guide,.7)
    if directional:
        cx=np.exp(-np.square(np.diff(guide,axis=1)/edge_scale))
        cy=np.exp(-np.square(np.diff(guide,axis=0)/edge_scale))
    else:
        cx=np.ones((x.shape[0],x.shape[1]-1));cy=np.ones((x.shape[0]-1,x.shape[1]))
    if support is not None:
        cx=np.where(support[:,:-1] & support[:,1:],cx,1.)
        cy=np.where(support[:-1,:] & support[1:,:],cy,1.)
    dx=cx*np.diff(x,axis=1);dy=cy*np.diff(x,axis=0)
    divergence=np.zeros_like(x)
    divergence[:,:-1]+=dx;divergence[:,1:]-=dx
    divergence[:-1,:]+=dy;divergence[1:,:]-=dy
    return np.clip(x+rate*divergence,0,1)

def reconstruct_v5(sino,theta,initial,guide,mask,outer=4,directional=True,support=None):
    x=initial.copy()
    for _ in range(outer):
        x=np.clip(iradon_sart(sino,theta=theta,image=x,relaxation=.08),0,1)*mask
        for __ in range(2):x=guided_step(x,guide,directional=directional,support=support)*mask
    return x

def lesion_metrics(truth,lesion,pred,theta,sino,mask):
    from run_v4 import metrics
    vals=metrics(truth,lesion,pred,sino,theta)
    ys,xs=np.where(lesion);center=np.array([ys.mean(),xs.mean()])
    # Oracle-bounded search: uses known lesion location, NOT independent detection.
    radius=12; yy,xx=np.indices(pred.shape)
    roi=((yy-center[0])**2+(xx-center[1])**2<=radius**2)&mask
    sm=gaussian_filter(pred,1.)
    maxpos=np.unravel_index(np.argmax(np.where(roi,sm,-np.inf)),sm.shape)
    vals['oracle_local_peak_offset_px']=float(np.linalg.norm(np.array(maxpos)-center))
    vals['lesion_contrast_signed']=float(pred[lesion].mean()-pred[binary_dilation(lesion,iterations=8)&~binary_dilation(lesion,iterations=2)&mask].mean())
    return vals

def run(seeds=30,size=64,angles=21,extent=25,noise=.015,outer=4,output='results_v5'):
    if seeds<4 or size<48 or angles<3 or outer<1:raise ValueError('Invalid experiment size')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    records=[];show=None;mask=circular_mask(size)
    for j in range(seeds):
        seed=600000+j
        truth,lesion,details,theta,sino,sart,us,shifted,unrelated=synthesize(seed,size,angles,extent,noise)
        # Lesion-erased US: locally substitute a smoothed background estimate.
        replace=binary_dilation(lesion,iterations=4)
        background=gaussian_filter(us,7.)
        erased=np.where(replace,background,us)
        # Missing field of view: leave one side unsupported and fall back to unguided smoothing.
        support=np.broadcast_to(np.arange(size)[None,:]<size//2,(size,size)).copy()
        missing=np.where(support,us,0)
        outputs={'sart':sart,
          'xray_isotropic':reconstruct_v5(sino,theta,sart,sart,mask,outer,directional=False),
          'xray_directional':reconstruct_v5(sino,theta,sart,gaussian_filter(sart,1),mask,outer),
          'us_directional':reconstruct_v5(sino,theta,sart,us,mask,outer),
          'us_shifted':reconstruct_v5(sino,theta,sart,shifted,mask,outer),
          'us_unrelated':reconstruct_v5(sino,theta,sart,unrelated,mask,outer),
          'us_lesion_erased':reconstruct_v5(sino,theta,sart,erased,mask,outer),
          'us_missing_coverage':reconstruct_v5(sino,theta,sart,missing,mask,outer,support=support)}
        for name,img in outputs.items():
            records.append({'seed':seed,'method':name,**details,**lesion_metrics(truth,lesion,img,theta,sino,mask)})
        if j==0:show=[('Ground truth',truth),('US surrogate',us),('X-ray directional',outputs['xray_directional']),('US directional',outputs['us_directional']),('US unrelated',outputs['us_unrelated']),('US erased lesion',outputs['us_lesion_erased']),('US shifted',outputs['us_shifted']),('US partial FOV',outputs['us_missing_coverage'])]
        if (j+1)%10==0:print(f'Completed {j+1}/{seeds}',flush=True)
    keys=('ssim','lesion_mae','contrast_error','projection_rmse','oracle_local_peak_offset_px')
    mean={method:{key:float(np.mean([r[key] for r in records if r['method']==method])) for key in keys} for method in METHODS}
    pair={}; rng=np.random.default_rng(773)
    for method in METHODS:
        if method=='xray_directional':continue
        pair[method]={}
        for key in keys:
            a=np.array([r[key] for r in records if r['method']==method]);b=np.array([r[key] for r in records if r['method']=='xray_directional'])
            improvements=(a-b)*(1 if key=='ssim' else -1)
            bootstrap=rng.choice(improvements,(2000,seeds),replace=True).mean(axis=1)
            pair[method][key]={'mean_improvement':float(improvements.mean()),'ci95':np.percentile(bootstrap,[2.5,97.5]).tolist(),'fraction_better':float(np.mean(improvements>0))}
    summary={'parameters':dict(seeds=seeds,size=size,angles=angles,extent=extent,noise=noise,outer=outer), 'mean_metrics':mean,'paired_vs_xray_directional':pair,
       'method':'SART alternating with guide-gradient weighted 4-neighbor anisotropic diffusion (not mathematically DTV)',
       'limitations':['2D parallel-beam limited-angle is not DBT geometry','US surrogate originates from ground-truth phantom; privileged-information leakage','No real ultrasound physics, RF data, or patients','Lesion-derived erasure and oracle-constrained localization use ground-truth mask for evaluation/stress-test construction','No independent detection metric','All same-seed observations are paired, synthetic variability is restricted; bootstrap CI is not clinical generalization']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'per_seed_metrics.json').write_text(json.dumps(records,indent=2))
    fig,axs=plt.subplots(2,4,figsize=(12,6))
    for ax,(title,im) in zip(axs.flat,show):ax.imshow(im,cmap='gray',vmin=0,vmax=1);ax.set_title(title,fontsize=9);ax.axis('off')
    fig.tight_layout();fig.savefig(out/'v5_comparison.png',dpi=140);plt.close(fig)
    print('V5 synthetic stress test; NOT directional TV optimization or medical validation')
    for m in METHODS:print(f'{m:21s} SSIM {mean[m]["ssim"]:.3f} lesion MAE {mean[m]["lesion_mae"]:.3f} contrastErr {mean[m]["contrast_error"]:.3f} peakOffset {mean[m]["oracle_local_peak_offset_px"]:.2f}')
    print('Saved',out.resolve());return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=30);p.add_argument('--size',type=int,default=64);p.add_argument('--angles',type=int,default=21);p.add_argument('--extent',type=float,default=25);p.add_argument('--noise',type=float,default=.015);p.add_argument('--outer',type=int,default=4);p.add_argument('--output',default='results_v5')
    run(**vars(p.parse_args()))
