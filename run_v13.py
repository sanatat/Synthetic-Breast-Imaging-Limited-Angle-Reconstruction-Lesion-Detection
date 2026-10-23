"""V13: SIRT convergence and discretization-mismatch audit (synthetic, CPU).

Data generation uses a denser ray quadrature than inverse reconstruction.
This reduces identical-matrix inverse crime, but does not create an independent
physical imaging model, nor constitute clinical validation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_v7 import phantom
from run_v8 import system_matrix, evaluate
from run_v10 import sirt_trace, audit_operator


def matrices(views, ny=12, size=18, extent=18., detector_u=30, detector_v=24,
             reconstruction_samples=42, simulation_samples=77):
    if reconstruction_samples == simulation_samples:
        raise ValueError('Simulation/reconstruction sampling must differ')
    common=dict(ny=ny,n=size,views=views,extent=extent,
                detector_u=detector_u,detector_v=detector_v)
    return (system_matrix(**common,samples=reconstruction_samples),
            system_matrix(**common,samples=simulation_samples))


def run(seeds=2, views=(9,15,21), iterations=(15,30,60,120),
        output='results_v13', noise=.005, ny=12, size=18,
        reconstruction_samples=42, simulation_samples=77):
    if seeds<1 or noise<0 or min(iterations)<1 or list(iterations)!=sorted(set(iterations)):
        raise ValueError('Invalid experiment settings')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records=[]; audits={}; example={}
    for v in views:
        A_rec,A_gen=matrices(v,ny=ny,size=size,reconstruction_samples=reconstruction_samples,
                             simulation_samples=simulation_samples)
        audits[str(v)]={**audit_operator(A_rec),
                         'normalized_operator_mismatch':float(np.sqrt((A_gen-A_rec).power(2).sum()) / max(np.sqrt(A_gen.power(2).sum()),1e-12))}
        for seed in range(seeds):
            truth,lesion,_=phantom(130000+seed,ny,size)
            scale_truth=truth.ravel().astype(np.float32)
            for regime,A_forward in [('matched',A_rec),('different_quadrature',A_gen)]:
                clean=np.asarray(A_forward@scale_truth,dtype=np.float32)
                rng=np.random.default_rng(20261300+v*1000+seed*3)
                sigma=noise*max(float(clean.max()),1e-8)
                measured=clean+rng.normal(0,sigma,clean.shape).astype(np.float32)
                trace,pred=sirt_trace(A_rec,measured,truth,lesion,checkpoints=tuple(iterations))
                for r in trace:
                    records.append({'views':v,'seed':seed,'regime':regime,'noise_sigma':sigma,**r})
                if seed==0 and regime=='different_quadrature':example[v]=(truth,pred,lesion)
    means=[]
    for regime in ('matched','different_quadrature'):
        for v in views:
            for iteration in iterations:
                subset=[r for r in records if r['regime']==regime and r['views']==v and r['iteration']==iteration]
                means.append({'regime':regime,'views':v,'iteration':iteration,
                    **{k:float(np.mean([r[k] for r in subset])) for k in
                    ('lesion_mae','volume_mae','projection_rmse','contrast_error')}})
    summary={'description':'V13 synthetic CPU SIRT convergence & ray quadrature mismatch',
             'settings':{'seeds':seeds,'views':list(views),'iterations':list(iterations),
                         'shape':[ny,size,size],'extent_degrees':18,'detector':[30,24],
                         'reconstruction_samples':reconstruction_samples,
                         'simulation_samples':simulation_samples,'noise':noise},
             'audits':audits,'means':means,
             'limitations':['Both operators implement the same idealized ray physics; quadrature mismatch only, not a fully independent physical forward model.',
                            'Per-projection Gaussian noise is not an X-ray photon dose model.',
                            'Only synthetic data and oracle lesion-mask-based scoring.',
                            'Few seeds; no demonstrated significance or clinical transfer.',
                            'SIRT projection RMSE refers to reconstruction-operator residual, not independent detector validation.']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    (output/'per_seed_metrics.json').write_text(json.dumps(records,indent=2))
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for regime,style in [('matched','-'),('different_quadrature','--')]:
        for v in views:
            q=[r for r in means if r['regime']==regime and r['views']==v]
            axs[0,0].plot(iterations,[r['lesion_mae'] for r in q],style,marker='o',label=f'{v} {regime}')
            axs[0,1].plot(iterations,[r['volume_mae'] for r in q],style,marker='o',label=f'{v} {regime}')
            axs[1,0].plot(iterations,[r['projection_rmse'] for r in q],style,marker='o',label=f'{v} {regime}')
            axs[1,1].plot(iterations,[r['contrast_error'] for r in q],style,marker='o',label=f'{v} {regime}')
    for ax,title in zip(axs.flat,['Oracle lesion MAE','Volume MAE','Projection RMSE','Oracle contrast error']):
        ax.set_title(title);ax.set_xlabel('SIRT iteration');ax.grid(alpha=.25)
    axs[1,1].legend(fontsize=7,ncol=2)
    fig.suptitle('V13 SIRT convergence: matched vs quadrature-mismatched synthetic operators')
    fig.tight_layout();fig.savefig(output/'v13_convergence.png',dpi=140);plt.close(fig)
    truth,_,lesion=example[views[0]]
    center=np.argwhere(lesion).mean(axis=0).round().astype(int)
    fig,axs=plt.subplots(2,len(views)+1,figsize=(3*(len(views)+1),6))
    for j,(name,vol) in enumerate([('Truth',truth)]+[(f'{v} views',example[v][1]) for v in views]):
        axs[0,j].imshow(vol[center[0]],cmap='gray',vmin=0,vmax=.85)
        axs[1,j].imshow(vol[:,center[1],:],cmap='gray',vmin=0,vmax=.85,aspect='auto')
        axs[0,j].set_title(name)
        for ax in axs[:,j]:ax.axis('off')
    fig.suptitle('V13 different-quadrature reconstruction; synthetic volume')
    fig.tight_layout();fig.savefig(output/'v13_reconstructions.png',dpi=140);plt.close(fig)
    print('V13 synthetic convergence + mismatch audit (not clinical DBT)')
    for regime in ('matched','different_quadrature'):
        for v in views:
            q=next(x for x in means if x['regime']==regime and x['views']==v and x['iteration']==iterations[-1])
            print(f'{regime:21} {v:2} views / {iterations[-1]:3} steps: lesionMAE {q["lesion_mae"]:.4f} volumeMAE {q["volume_mae"]:.4f} projRMSE {q["projection_rmse"]:.4f}')
    print('Saved:',output.resolve())
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=2)
    parser.add_argument('--views',type=int,nargs='+',default=[9,15,21])
    parser.add_argument('--iterations',type=int,nargs='+',default=[15,30,60,120])
    parser.add_argument('--output',default='results_v13')
    args=parser.parse_args();run(**vars(args))
