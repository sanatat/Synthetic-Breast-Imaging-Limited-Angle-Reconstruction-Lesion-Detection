"""V10: diagnostic audit of the synthetic V8 cone-beam forward model.

Same sparse projector for simulation/inversion ('inverse crime'); NOT clinical DBT.
The fixed-total-dose noise model is a deliberately simplified Gaussian proxy.
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


def audit_operator(A, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=A.shape[1]).astype(np.float32)
    y = rng.normal(size=A.shape[0]).astype(np.float32)
    lhs = float(np.dot(A@x, y))
    rhs = float(np.dot(x, A.T@y))
    relative = abs(lhs-rhs)/max(abs(lhs), abs(rhs), 1.)
    sensitivity = np.asarray(A.sum(axis=0)).ravel()
    coverage = float(np.mean(sensitivity > 1e-8))
    return dict(adjoint_relative_error=relative, voxel_coverage_fraction=coverage,
                voxel_sensitivity_min=float(sensitivity.min()),
                voxel_sensitivity_median=float(np.median(sensitivity)),
                voxel_sensitivity_max=float(sensitivity.max()))


def sirt_trace(A, measurement, truth, lesion, checkpoints=(1,5,15,30,60), relaxation=.85):
    if min(checkpoints) < 1 or sorted(set(checkpoints)) != list(checkpoints):
        raise ValueError('Checkpoints must be increasing positive integers')
    shape = truth.shape
    x = np.zeros(A.shape[1], dtype=np.float32)
    rows = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    cols = np.asarray(A.sum(axis=0)).ravel().astype(np.float32)
    ir = np.divide(1., rows, out=np.zeros_like(rows), where=rows>1e-8)
    ic = np.divide(1., cols, out=np.zeros_like(cols), where=cols>1e-8)
    series=[]
    for iteration in range(1, max(checkpoints)+1):
        residual = measurement-A@x
        x += relaxation * ic * (A.T@(ir*residual))
        np.clip(x,0,1,out=x)
        if iteration in checkpoints:
            metrics = evaluate(truth, lesion, x.reshape(shape), A, measurement)
            series.append(dict(iteration=iteration,lesion_mae=metrics['lesion_mae'],
                               volume_mae=metrics['volume_mae'],
                               projection_rmse=metrics['projection_rmse'],
                               contrast_error=metrics['contrast_error']))
    return series, x.reshape(shape)


def run(views=(9,15,21),seeds=2,checkpoints=(1,5,15,30,60),ny=12,size=18,
        extent=18.,noise=.005,output='results_v10'):
    if seeds<1 or any(v<3 for v in views) or noise<0:
        raise ValueError('Invalid settings')
    out=Path(output);out.mkdir(parents=True, exist_ok=True)
    records=[]; audits={}; examples={}; reference_views=max(views)
    for v in views:
        A=system_matrix(ny=ny,n=size,views=v,extent=extent,detector_u=size+4,
                        detector_v=ny+4,samples=42)
        audits[str(v)]=audit_operator(A)
        for seed in range(seeds):
            truth,lesion,_=phantom(230000+seed,ny,size)
            clean=A@truth.ravel()
            # Matched per-view amplitude: fixed noise on each acquired projection.
            # Total-dose proxy: Gaussian sigma grows as sqrt(number_of_views).
            scale=noise*max(float(clean.max()),1e-8)
            rng=np.random.default_rng(88000+seed*997+v)
            z=rng.normal(size=clean.size).astype(np.float32)
            for regime, sigma in [('per_view_fixed',scale),
                                  ('fixed_total_dose_proxy',scale*np.sqrt(v/9))]:
                y=(clean+z*sigma).astype(np.float32)
                trace,pred=sirt_trace(A,y,truth,lesion,checkpoints)
                for r in trace:
                    records.append(dict(views=v,seed=seed,regime=regime,noise_sigma=sigma,**r))
                if seed==0 and regime=='per_view_fixed':
                    examples[v]=dict(truth=truth,pred=pred,lesion=lesion)
    aggregates=[]
    for regime in ['per_view_fixed','fixed_total_dose_proxy']:
        for v in views:
            for step in checkpoints:
                rows=[r for r in records if r['views']==v and r['regime']==regime and r['iteration']==step]
                aggregates.append(dict(regime=regime,views=v,iteration=step,
                            lesion_mae=float(np.mean([r['lesion_mae'] for r in rows])),
                            projection_rmse=float(np.mean([r['projection_rmse'] for r in rows])),
                            volume_mae=float(np.mean([r['volume_mae'] for r in rows]))))
    summary=dict(description='V10 synthetic cone-beam projector QA; NOT clinical or calibrated DBT',
        settings=dict(views=list(views),seeds=seeds,checkpoints=list(checkpoints),ny=ny,
                      size=size,extent=extent,noise=noise),
        operator_audits=audits,mean_convergence=aggregates,
        limitations=['Identical forward and inverse operator (inverse crime)',
          'Small synthetic phantoms; oracle lesion mask for metrics',
          'Approximate Gaussian noise proxy, not Poisson photon/dose physics',
          'No independent ultrasound, no real X-ray measurements',
          'Convergence measured over a limited number of iterations, not a proof of optimum'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'per_seed_metrics.json').write_text(json.dumps(records,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for v in views:
        rows=[r for r in aggregates if r['views']==v and r['regime']=='per_view_fixed']
        axes[0].plot([r['iteration'] for r in rows],[r['lesion_mae'] for r in rows],marker='o',label=f'{v} views')
        axes[1].plot([r['iteration'] for r in rows],[r['projection_rmse'] for r in rows],marker='o',label=f'{v} views')
    for ax in axes:
        ax.set_xlabel('SIRT iterations');ax.legend();ax.grid(alpha=.25)
    axes[0].set_ylabel('Oracle lesion MAE (lower better)')
    axes[1].set_ylabel('Projection RMSE (lower better)')
    fig.suptitle('V10 synthetic cone-beam convergence audit (fixed per-view noise)')
    fig.tight_layout();fig.savefig(out/'v10_convergence.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,len(views)+1,figsize=(3*(len(views)+1),6))
    example=examples[reference_views];center=np.argwhere(example['lesion']).mean(axis=0).round().astype(int)
    panels=[('Truth',example['truth'])]+[(f'{v} views',examples[v]['pred']) for v in views]
    for i,(label,volume) in enumerate(panels):
        axes[0,i].imshow(volume[center[0]],cmap='gray',vmin=0,vmax=.8)
        axes[1,i].imshow(volume[:,center[1],:],cmap='gray',vmin=0,vmax=.8,aspect='auto')
        axes[0,i].set_title(label)
        for ax in axes[:,i]:ax.axis('off')
    fig.suptitle('V10 synthetic reconstructions after final iteration')
    fig.tight_layout();fig.savefig(out/'v10_reconstructions.png',dpi=140);plt.close(fig)
    print('V10 diagnostics: synthetic, not clinical DBT')
    for v in views:
        audit=audits[str(v)]
        row=next(r for r in aggregates if r['views']==v and r['iteration']==max(checkpoints) and r['regime']=='per_view_fixed')
        print(f'{v:2} views  adjoint_relerr={audit["adjoint_relative_error"]:.2e} coverage={audit["voxel_coverage_fraction"]:.3f} '
              f'lesionMAE={row["lesion_mae"]:.4f} projectionRMSE={row["projection_rmse"]:.4f}')
    print('Saved:',out.resolve())
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--views',nargs='+',type=int,default=[9,15,21])
    p.add_argument('--seeds',type=int,default=2)
    p.add_argument('--checkpoints',nargs='+',type=int,default=[1,5,15,30,60])
    p.add_argument('--ny',type=int,default=12);p.add_argument('--size',type=int,default=18)
    p.add_argument('--extent',type=float,default=18.)
    p.add_argument('--noise',type=float,default=.005)
    p.add_argument('--output',default='results_v10')
    run(**vars(p.parse_args()))
