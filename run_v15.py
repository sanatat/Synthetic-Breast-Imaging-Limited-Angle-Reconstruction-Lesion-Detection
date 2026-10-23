"""V15: hard-negative synthetic lesion-screening stress test (NOT clinical detection).

No reference lesion mask is used for reconstruction/candidate inference. Ground truth
is used only to construct simulations and compute retrospective metrics.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_v7 import phantom
from run_v8 import system_matrix
from run_v14 import poisson_line_integrals, blind_candidate, remove_lesion


def stress_pair(seed, ny=12, n=18):
    """Paired lesion-positive and hard-negative volumes sharing tissue background.

    Hard negatives contain benign-looking localized hyperattenuating distractors.
    This is a synthetic appearance test, not pathological ground truth.
    """
    rng=np.random.default_rng(750000+seed)
    initial, initial_mask,_=phantom(750000+seed,ny,n)
    baseline=remove_lesion(initial,initial_mask)
    y,z,x=np.ogrid[:ny,:n,:n]
    body=baseline>.12
    # Smooth heterogeneous parenchyma, no global geometry leakage from lesion seed.
    texture=gaussian_filter(rng.normal(size=baseline.shape),sigma=(1.0,1.0,1.0))
    texture=texture/(np.std(texture[body])+1e-8)
    baseline=np.clip(baseline+body*texture*.025,0,1).astype(np.float32)
    cy=(ny-1)/2; cz=(n-1)/2; cx=(n-1)/2
    def center():return (cy+rng.uniform(-1.6,1.6),cz+rng.uniform(-2.0,2.0),cx+rng.uniform(-2.0,2.0))
    # Distractors exist in BOTH classes, so simple bright-spot thresholding may fail.
    for _ in range(int(rng.integers(1,4))):
        a,b,c=center(); rad=rng.uniform(.65,1.8)
        d=((y-a)/rad)**2+((z-b)/rad)**2+((x-c)/rad)**2
        baseline=np.clip(baseline+body*np.exp(-d*1.4)*rng.uniform(.10,.27),0,1).astype(np.float32)
    # Lesion center/radius independently varied, including subresolution lesions.
    a,b,c=center(); ry=rng.uniform(.85,2.2); rz=rng.uniform(.85,2.2); rx=rng.uniform(.85,2.2)
    lesion=((y-a)/ry)**2+((z-b)/rz)**2+((x-c)/rx)**2<=1
    lesion=lesion&body
    # Guarantee nonempty even with small radii.
    if not lesion.any():
        coord=np.array([a,b,c]).round().astype(int)
        lesion[tuple(coord)]=True
    amplitude=rng.uniform(.17,.44)
    positive=np.clip(baseline+amplitude*lesion,0,1).astype(np.float32)
    return positive,baseline,lesion,{'lesion_radius_vox':[ry,rz,rx], 'lesion_contrast':float(amplitude), 'lesion_voxels':int(lesion.sum())}


def sirt_blind(A,y,shape,steps=70,relaxation=.85):
    """Reconstruct without truth volume or lesion mask."""
    if steps<1:raise ValueError('steps must be >= 1')
    x=np.zeros(A.shape[1],np.float32)
    rows=np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    cols=np.asarray(A.sum(axis=0)).ravel().astype(np.float32)
    ir=np.divide(1,rows,out=np.zeros_like(rows),where=rows>1e-8)
    ic=np.divide(1,cols,out=np.zeros_like(cols),where=cols>1e-8)
    for _ in range(steps):
        x += relaxation*ic*(A.T@(ir*(y-A@x)))
        np.clip(x,0,1,out=x)
    return x.reshape(shape)


def classification_metrics(rows,threshold):
    pos=[r for r in rows if r['label']=='positive']
    neg=[r for r in rows if r['label']=='negative']
    tp=sum(r['candidate_score']>threshold for r in pos)
    fp=sum(r['candidate_score']>threshold for r in neg)
    return {'n_positive':len(pos),'n_hard_negative':len(neg),
            'true_positive':tp,'false_negative':len(pos)-tp,
            'false_positive':fp,'true_negative':len(neg)-fp,
            'sensitivity':tp/len(pos) if pos else None,
            'false_positive_rate':fp/len(neg) if neg else None,
            'mean_localization_error_all_positive_vox':float(np.mean([r['center_error_vox'] for r in pos])) if pos else None}


def run(seeds=12,steps=70,views=15,photons=30000,output='results_v15'):
    if seeds<6 or steps<1 or views<3 or photons<=0:raise ValueError('invalid settings')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    ny,n=12,18
    A=system_matrix(ny=ny,n=n,views=views,extent=18,detector_u=30,detector_v=24,samples=42)
    records=[];example=None
    for seed in range(seeds):
        positive,negative,mask,info=stress_pair(seed,ny,n)
        true_center=np.argwhere(mask).mean(axis=0)
        for j,(label,truth) in enumerate((('positive',positive),('negative',negative))):
            rng=np.random.default_rng(880000+seed*2+j)
            measurement,_=poisson_line_integrals(A,truth,rng,photons,.035)
            rec=sirt_blind(A,measurement,truth.shape,steps)
            candidate=blind_candidate(rec)
            records.append({'seed':seed,'label':label,'candidate_score':candidate['score'],
                'candidate_center':candidate['center'],
                'center_error_vox':float(np.linalg.norm(np.array(candidate['center'])-true_center)) if j==0 else None,
                'lesion_voxels':info['lesion_voxels'] if j==0 else None,
                'projection_rmse':float(np.sqrt(np.mean((A@rec.ravel()-measurement)**2)))})
            if seed==seeds-1 and label=='positive':example=(truth,rec,candidate)
    split=seeds//2
    dev=[r for r in records if r['seed']<split]
    hold=[r for r in records if r['seed']>=split]
    neg_scores=[r['candidate_score'] for r in dev if r['label']=='negative']
    threshold=float(np.quantile(neg_scores,.95))
    summary={'scope':'Synthetic hard-negative stress test; NOT a medical diagnostic model',
      'settings':{'seeds':seeds,'steps':steps,'views':views,'photons_per_ray':photons},
      'threshold_development_negative_95pct':threshold,
      'development':classification_metrics(dev,threshold),
      'held_out':classification_metrics(hold,threshold),
      'limitations':['Small synthetic data with paired tissue backgrounds; not patient independent data.',
        'Bright distractors and lesions differ by a simulated contrast bump, not pathology.',
        'Same ray operator used to simulate and reconstruct; inverse crime remains.',
        'Threshold selected on development negatives only; confidence intervals are wide.',
        'One candidate per image, not a validated lesion detector.',
        'No independent ultrasound, clinical data, or dose-calibrated imaging physics.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'per_seed_metrics.json').write_text(json.dumps(records,indent=2))
    truth,rec,candidate=example
    c=int(candidate['center'][0]);fig,axs=plt.subplots(1,2,figsize=(7.2,3.4))
    axs[0].imshow(truth[c],cmap='gray',vmin=0,vmax=.8)
    axs[1].imshow(rec[c],cmap='gray',vmin=0,vmax=.8)
    axs[1].plot(candidate['center'][2],candidate['center'][1],'rx',ms=9)
    for ax,title in zip(axs,['Synthetic truth','Blind candidate (red x)']):ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(out/'v15_example.png',dpi=150);plt.close(fig)
    print('V15 synthetic stress test — NOT clinical accuracy')
    print(json.dumps(summary,indent=2));print('Saved to',out.resolve())
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=12)
    p.add_argument('--steps',type=int,default=70)
    p.add_argument('--views',type=int,default=15)
    p.add_argument('--photons',type=float,default=30000)
    p.add_argument('--output',default='results_v15')
    run(**vars(p.parse_args()))
