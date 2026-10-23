"""V4 held-out randomized-lesion stress test; purely synthetic, nonclinical."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, shift, binary_dilation
from skimage.transform import radon, iradon_sart
from skimage.metrics import structural_similarity
from run_experiment import ultrasound_surrogate
from run_v2 import reconstruct
from run_v3 import circular_mask, paired_stats as _unused
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=('sart','xray_only','us_correct','us_shifted','us_unrelated','us_constant')

def phantom_v4(n, seed):
    rng=np.random.default_rng(seed)
    yy,xx=np.indices((n,n)); cx=(n-1)/2; cy=cx
    X=(xx-cx)/(n/2); Y=(yy-cy)/(n/2)
    breast=(X*X/.85**2+Y*Y/.84**2)<1
    img=np.where(breast,.26+.07*np.exp(-((X+.23)**2+(Y-.1)**2)/.16),0.)
    for _ in range(14):
        px,py=rng.uniform(-.48,.48,2); sx,sy=rng.uniform(.05,.19,2)
        img+=breast*rng.uniform(.01,.075)*np.exp(-((X-px)**2/sx**2+(Y-py)**2/sy**2))
    # Lesion position, radius, contrast, and aspect ratio vary independently across seeds.
    lx,ly=rng.uniform(-.42,.42,2); r=rng.uniform(.065,.16); ratio=rng.uniform(.7,1.35)
    lesion=((X-lx)/(r*ratio))**2+((Y-ly)/r)**2<1
    amplitude=rng.uniform(.22,.52)
    img=np.where(lesion,np.minimum(img+amplitude,.95),img)
    return np.clip(gaussian_filter(img,.5)*circular_mask(n),0,1),lesion & circular_mask(n), {'x':float(lx),'y':float(ly),'r':float(r),'amplitude':float(amplitude)}

def synthesize(seed,n,angles,extent,noise):
    rng=np.random.default_rng(seed+10000)
    truth,lesion,details=phantom_v4(n,seed+4000)
    theta=np.linspace(-extent,extent,angles)
    sino_clean=radon(truth,theta=theta,circle=True)
    sino=sino_clean+rng.normal(0,noise*max(sino_clean.max(),1e-10),sino_clean.shape)
    sart=np.zeros_like(truth)
    for _ in range(2):
        sart=np.clip(iradon_sart(sino,theta=theta,image=sart,relaxation=.12),0,1)*circular_mask(n)
    us=ultrasound_surrogate(truth,rng)*circular_mask(n)
    # Independent sample, not a rotation of the same physical anatomy
    other,_,_=phantom_v4(n,seed+200000)
    unrelated=ultrasound_surrogate(other,np.random.default_rng(seed+900000))*circular_mask(n)
    shifted=shift(us,(6,-6),order=1,mode='constant',prefilter=False)*circular_mask(n)
    return truth,lesion,details,theta,sino,sart,us,shifted,unrelated

def metrics(truth,lesion,pred,sino,theta):
    mask=circular_mask(len(truth)); ring=binary_dilation(lesion,iterations=8)&~binary_dilation(lesion,iterations=2)&mask
    gt_contrast=truth[lesion].mean()-truth[ring].mean()
    contrast=pred[lesion].mean()-pred[ring].mean()
    return {'ssim':float(structural_similarity(truth,pred,data_range=1.)),
            'lesion_mae':float(np.abs(pred[lesion]-truth[lesion]).mean()),
            'contrast_error':float(abs(contrast-gt_contrast)),
            'projection_rmse':float(np.sqrt(np.mean((radon(pred*mask,theta=theta,circle=True)-sino)**2)))}

def run(seeds=30, size=64, angles=21, extent=25, noise=.015, outer=4, output='results_v4'):
    if seeds<4 or size<48 or angles<3 or outer<1:raise ValueError('Require seeds>=4,size>=48,angles>=3,outer>=1')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    records=[]; example=None
    # Dedicated, deterministic held-out IDs. V4 does not tune model parameters against them.
    for j in range(seeds):
        seed=500000+j
        truth,lesion,details,theta,sino,sart,us,shifted,unrelated=synthesize(seed,size,angles,extent,noise)
        mask=circular_mask(size)
        guides={'xray_only':gaussian_filter(sart,1)*mask,'us_correct':us,'us_shifted':shifted,
                'us_unrelated':unrelated,'us_constant':np.full_like(us,float(us[mask].mean()))*mask}
        predictions={'sart':sart}
        for name,guide in guides.items():predictions[name]=reconstruct(sino,theta,sart,guide,outer=outer)*mask
        for method in METHODS: records.append({'seed':seed,'method':method,**details,**metrics(truth,lesion,predictions[method],sino,theta)})
        if example is None: example=[('Ground truth',truth),('US surrogate',us),('X-ray only',predictions['xray_only']),('US correct',predictions['us_correct']),('US unrelated',predictions['us_unrelated']),('US shifted',predictions['us_shifted'])]
        if (j+1)%10==0:print(f'Completed {j+1}/{seeds}',flush=True)
    keys=('ssim','lesion_mae','contrast_error','projection_rmse')
    avg={m:{k:float(np.mean([r[k] for r in records if r['method']==m])) for k in keys} for m in METHODS}
    differences={};rng=np.random.default_rng(42)
    for m in METHODS:
        if m=='xray_only':continue
        differences[m]={}
        for k in keys:
            d=np.array([r[k] for r in records if r['method']==m])-np.array([r[k] for r in records if r['method']=='xray_only'])
            if k!='ssim':d=-d
            bootstrap=rng.choice(d,(2000,seeds),replace=True).mean(axis=1)
            differences[m][k]={'mean_improvement':float(d.mean()),'ci95':np.percentile(bootstrap,[2.5,97.5]).tolist(), 'fraction_better':float(np.mean(d>0))}
    summary={'design':'Held-out deterministic seed IDs (500000+); lesion location/shape/contrast randomized; no V4 hyperparameter tuning',
             'parameters':dict(seeds=seeds,size=size,angles=angles,extent=extent,noise=noise,outer=outer),
             'mean_metrics':avg,'paired_vs_xray_only':differences,
             'limitations':['2D parallel-beam geometry is NOT DBT','US is generated from ground truth; potential privileged-information leakage','Unrelated anatomy generated independently from same synthetic family','No patient images; not clinical validation','Metrics use known lesion masks, not blinded detection']}
    (out/'per_seed_metrics.json').write_text(json.dumps(records,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axs=plt.subplots(2,3,figsize=(10,7))
    for ax,(name,img) in zip(axs.ravel(),example):ax.imshow(img,cmap='gray',vmin=0,vmax=1);ax.set_title(name);ax.axis('off')
    fig.tight_layout();fig.savefig(out/'v4_comparison.png',dpi=120);plt.close(fig)
    print('V4 randomized-lesion synthetic holdout (not clinical evidence):')
    for m in METHODS:print(f'{m:15s} SSIM {avg[m]["ssim"]:.3f} lesion MAE {avg[m]["lesion_mae"]:.3f} contrast error {avg[m]["contrast_error"]:.3f}')
    for k in ('ssim','lesion_mae'):print('US correct vs Xray',k,differences['us_correct'][k])
    print('Saved to',out.resolve());return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=30);p.add_argument('--size',type=int,default=64);p.add_argument('--angles',type=int,default=21);p.add_argument('--extent',type=float,default=25);p.add_argument('--noise',type=float,default=.015);p.add_argument('--outer',type=int,default=4);p.add_argument('--output',default='results_v4');a=p.parse_args();run(**vars(a))
