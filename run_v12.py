"""V12 full-FOV, limited-angle cone-beam view-count experiment; synthetic CPU-only.

Not calibrated DBT. Forward and inverse use same system matrix (inverse crime).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_v7 import phantom
from run_v8 import system_matrix
from run_v10 import audit_operator, sirt_trace


def coverage_fraction(A):
    col = np.asarray(A.sum(axis=0)).ravel()
    return float(np.mean(col > 1e-8))


def run(seeds=2, views=(9,15,21), iterations=(5,15,30), ny=12, size=18,
        extent=18., noise=0.005, detector_u=30, detector_v=24, samples=42,
        output='results_v12'):
    if seeds < 1 or any(v<3 for v in views) or any(i<1 for i in iterations) or noise<0:
        raise ValueError('Invalid experiment parameters')
    if sorted(set(iterations)) != list(iterations):
        raise ValueError('Iterations must be sorted unique integers')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    records=[]; coverage={}; examples={}
    for nviews in views:
        A=system_matrix(ny=ny,n=size,views=nviews,extent=extent,
                        detector_u=detector_u,detector_v=detector_v,samples=samples)
        coverage[str(nviews)]={'fraction':coverage_fraction(A),**audit_operator(A)}
        for seed in range(seeds):
            truth,lesion,_=phantom(120000+seed,ny,size)
            clean=(A@truth.ravel()).astype(np.float32)
            rng=np.random.default_rng(124000+seed*17+nviews)
            # Same relative per-ray noise, not a constant total-exposure dose model.
            measured=(clean+rng.normal(0,noise*max(float(clean.max()),1e-8),clean.shape)).astype(np.float32)
            trace,pred=sirt_trace(A,measured,truth,lesion,checkpoints=tuple(iterations))
            records.extend({'views':nviews,'seed':seed,**r} for r in trace)
            if seed==0: examples[nviews]=(truth,pred)
    means=[]
    for nv in views:
        for itr in iterations:
            subset=[r for r in records if r['views']==nv and r['iteration']==itr]
            means.append({'views':nv,'iteration':itr,**{k:float(np.mean([r[k] for r in subset]))
                         for k in ('lesion_mae','volume_mae','projection_rmse','contrast_error')}})
    summary={'title':'V12 full field-of-view toy cone-beam SIRT view-count audit',
             'settings':{'seeds':seeds,'views':list(views),'iterations':list(iterations),
                         'shape':[ny,size,size],'angle_extent_degrees':extent,
                         'detector':[detector_u,detector_v],'samples':samples,'relative_noise':noise},
             'coverage':coverage,'means':means,
             'limitations':['Synthetic ray-sampling geometry is not calibrated clinical DBT.',
                            'Forward and reconstruction operators identical (inverse crime).',
                            'Oracle lesion mask used for scoring, not blind detection.',
                            'Per-ray Gaussian noise with fixed relative amplitude is not fixed-dose photon statistics.',
                            'Visibility does not establish stability, unique recovery, or statistical significance.']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    (output/'per_seed_metrics.json').write_text(json.dumps(records,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for nv in views:
        q=[r for r in means if r['views']==nv]
        axs[0].plot(iterations,[r['lesion_mae'] for r in q],marker='o',label=f'{nv} views')
        axs[1].plot(iterations,[r['projection_rmse'] for r in q],marker='o',label=f'{nv} views')
    axs[0].set_ylabel('Oracle lesion MAE (lower better)')
    axs[1].set_ylabel('Projection RMSE (lower better)')
    for ax in axs:
        ax.set_xlabel('SIRT iterations');ax.grid(alpha=.25);ax.legend()
    fig.suptitle('V12: full-FOV synthetic cone-beam audit (not clinical DBT)')
    fig.tight_layout();fig.savefig(output/'v12_convergence.png',dpi=130);plt.close(fig)
    fig,axs=plt.subplots(2,len(views)+1,figsize=(3*(len(views)+1),6),squeeze=False)
    truth=examples[views[0]][0]
    imgs=[('Truth',truth)]+[(f'{v} views',examples[v][1]) for v in views]
    for j,(label,vol) in enumerate(imgs):
        axs[0,j].imshow(vol[ny//2],cmap='gray',vmin=0,vmax=.85)
        axs[1,j].imshow(vol[:,size//2,:],cmap='gray',vmin=0,vmax=.85,aspect='auto')
        axs[0,j].set_title(label)
        for ax in axs[:,j]:ax.axis('off')
    fig.suptitle('V12: synthetic full-FOV cone-beam, central slices')
    fig.tight_layout();fig.savefig(output/'v12_reconstructions.png',dpi=130);plt.close(fig)
    for nv in views:
        q=next(r for r in means if r['views']==nv and r['iteration']==iterations[-1])
        print(f'{nv:2d} views: coverage {coverage[str(nv)]["fraction"]:.1%}, lesion MAE {q["lesion_mae"]:.4f}, projection RMSE {q["projection_rmse"]:.4f}')
    print('Saved to',output.resolve())
    return summary

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=2)
    p.add_argument('--views',nargs='+',type=int,default=[9,15,21])
    p.add_argument('--iterations',nargs='+',type=int,default=[5,15,30])
    p.add_argument('--output',default='results_v12')
    a=p.parse_args();run(seeds=a.seeds,views=tuple(a.views),iterations=tuple(a.iterations),output=a.output)
