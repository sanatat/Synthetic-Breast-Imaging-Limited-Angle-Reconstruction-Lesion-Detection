"""V11: detector field-of-view and 3D voxel sensitivity audit; synthetic geometry only."""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_v8 import system_matrix
from run_v10 import audit_operator


def sensitivity_volume(A, shape, threshold=1e-8):
    if A.shape[1] != int(np.prod(shape)):
        raise ValueError('Matrix-volume size mismatch')
    s = np.asarray(A.sum(axis=0)).ravel().reshape(shape)
    observed = s > threshold
    return s, observed


def slice_coverage(observed):
    if observed.ndim != 3:
        raise ValueError('Expected three-dimensional coverage mask')
    return {'y': observed.mean(axis=(1, 2)).tolist(),
            'z': observed.mean(axis=(0, 2)).tolist(),
            'x': observed.mean(axis=(0, 1)).tolist()}


def run(ny=12, size=18, views=15, extent=18., detector_widths=(14, 22, 30),
        detector_heights=(10, 16, 24), samples=42, output='results_v11'):
    if len(detector_widths) != len(detector_heights) or not detector_widths:
        raise ValueError('Paired detector sizes required')
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    shape=(ny,size,size); rows=[]; images=[]
    for du,dv in zip(detector_widths,detector_heights):
        A=system_matrix(ny=ny,n=size,views=views,extent=extent,detector_u=int(du),detector_v=int(dv),samples=samples)
        s,obs=sensitivity_volume(A,shape)
        profiles=slice_coverage(obs)
        interior=np.zeros(shape, dtype=bool)
        interior[:,size//4:3*size//4,size//4:3*size//4]=True
        row={'detector_u':int(du),'detector_v':int(dv),'coverage':float(obs.mean()),
             'interior_coverage':float(obs[interior].mean()),
             'sensitivity_positive_mean':float(s[obs].mean()) if obs.any() else 0.,
             'unobserved_voxels':int((~obs).sum()),'total_voxels':int(obs.size),
             'slice_coverage':profiles, 'adjoint_relative_error':audit_operator(A)['adjoint_relative_error']}
        rows.append(row);images.append((du,dv,s,obs))
    # Detector heights and widths increase together by design; cannot attribute change to just one dimension.
    (out/'summary.json').write_text(json.dumps({'model':'synthetic ray-driven cone-beam-like sparse projector, not calibrated DBT',
      'settings':{'shape':shape,'views':views,'extent_degrees':extent,'samples':samples},
      'results':rows,'limitations':['Sensitivity>0 is geometric visibility, not stable recoverability',
        'Detector width and height change together in this experiment',
        'Finite ray quadrature and interpolation affect coverage',
        'Same projector used for audit; no clinical or measured data']},indent=2))
    fig,axs=plt.subplots(2,len(images),figsize=(4*len(images),7),squeeze=False)
    for i,(du,dv,s,obs) in enumerate(images):
        axs[0,i].imshow(s[ny//2],cmap='magma',vmin=0,vmax=max(float(s.max()),1e-8))
        axs[1,i].imshow(obs[:,size//2,:],cmap='gray',vmin=0,vmax=1,aspect='auto')
        axs[0,i].set_title(f'Detector {du}×{dv}\nSensitivity at y middle')
        axs[1,i].set_title(f'Observed mask at z middle\nCoverage {obs.mean():.1%}')
        for ax in axs[:,i]:ax.set_xlabel('x');ax.set_ylabel('z' if ax==axs[0,i] else 'y')
    fig.suptitle('V11: synthetic cone-beam detector FOV and voxel coverage')
    fig.tight_layout();fig.savefig(out/'v11_sensitivity_maps.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,3.5))
    for row in rows:
        for ax,axis in zip(axs,('y','z','x')):
            ax.plot(row['slice_coverage'][axis],marker='.',label=f"{row['detector_u']}×{row['detector_v']}")
            ax.set_title(f'{axis} slice coverage');ax.set_ylim(0,1.05);ax.set_xlabel('Slice index');ax.set_ylabel('Covered fraction')
    axs[0].legend();fig.tight_layout();fig.savefig(out/'v11_coverage_profiles.png',dpi=150);plt.close(fig)
    for r in rows:
        print(f"Detector {r['detector_u']}x{r['detector_v']}: covered {r['coverage']:.1%}, "
              f"interior {r['interior_coverage']:.1%}, unobserved {r['unobserved_voxels']}")
    print('Saved',out.resolve())
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--ny',type=int,default=12);p.add_argument('--size',type=int,default=18)
    p.add_argument('--views',type=int,default=15);p.add_argument('--extent',type=float,default=18.)
    p.add_argument('--detector-widths',nargs='+',type=int,default=[14,22,30]);p.add_argument('--detector-heights',nargs='+',type=int,default=[10,16,24]);p.add_argument('--samples',type=int,default=42)
    p.add_argument('--output',default='results_v11');a=p.parse_args()
    run(a.ny,a.size,a.views,a.extent,a.detector_widths,a.detector_heights,a.samples,a.output)
