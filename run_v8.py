"""V8 synthetic 3D limited-angle cone-beam prototype (CPU, sparse ray-driven model).
Educational geometry, NOT calibrated DBT, no ultrasound, no patient data.
Coordinates of volumes: (y,z,x). Source moves in x-z plane, planar 2D detector.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.ndimage import binary_dilation
from skimage.restoration import denoise_tv_chambolle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_v7 import phantom


def system_matrix(ny=16, n=24, views=11, extent=18, detector_u=28, detector_v=20,
                  source_radius=60., detector_radius=55., samples=52):
    """Sparse trilinear ray integrals: source at distance source_radius, detector
    centered detector_radius past origin. Volume voxel pitch=1. Detector pitch=1.
    Ray sampling approximates integral; same A used for forward and adjoint.
    """
    if not (ny>=8 and n>=12 and views>=3 and 0 < extent < 85 and samples>=8):
        raise ValueError('Invalid acquisition geometry')
    rows=[]; cols=[]; vals=[]
    angles=np.deg2rad(np.linspace(-extent,extent,views))
    nr=views*detector_v*detector_u
    # volume physical axes: x,z,y with center at origin
    for iv,theta in enumerate(angles):
        radial=np.array([np.sin(theta), 0., np.cos(theta)])  # (x,y,z)
        tangent=np.array([np.cos(theta), 0., -np.sin(theta)])
        source=source_radius*radial
        detector_center=-detector_radius*radial
        for vy in range(detector_v):
            dy=(vy-(detector_v-1)/2)
            for ux in range(detector_u):
                du=(ux-(detector_u-1)/2)
                target=detector_center + tangent*du + np.array([0.,dy,0.])
                delta=target-source
                ray_length=np.linalg.norm(delta)
                # sample only central fraction encompassing entire volume, avoid wasting samples
                ta=(source_radius-(n*.78))/ (source_radius+detector_radius)
                tb=(source_radius+(n*.78))/ (source_radius+detector_radius)
                ts=np.linspace(ta,tb,samples)
                coords=source[None,:]+ts[:,None]*delta[None,:]
                xx=coords[:,0]+(n-1)/2
                yy=coords[:,1]+(ny-1)/2
                zz=coords[:,2]+(n-1)/2
                valid=(xx>=0)&(xx<n-1)&(yy>=0)&(yy<ny-1)&(zz>=0)&(zz<n-1)
                rid=(iv*detector_v+vy)*detector_u+ux
                dx=xx[valid];dyv=yy[valid];dz=zz[valid]
                if not len(dx): continue
                x0=np.floor(dx).astype(int); y0=np.floor(dyv).astype(int); z0=np.floor(dz).astype(int)
                fx=dx-x0;fy=dyv-y0;fz=dz-z0
                ds=ray_length*(tb-ta)/(samples-1)
                for iy in (0,1):
                    wy=fy if iy else 1-fy
                    for iz in (0,1):
                        wz=fz if iz else 1-fz
                        for ix in (0,1):
                            wx=fx if ix else 1-fx
                            w=(wx*wy*wz*ds).astype(np.float32)
                            rows.extend([rid]*len(w))
                            cols.extend((((y0+iy)*n+(z0+iz))*n+x0+ix).tolist())
                            vals.extend(w.tolist())
    A=sparse.coo_matrix((np.asarray(vals,np.float32),(rows,cols)),shape=(nr,ny*n*n)).tocsr()
    A.sum_duplicates()
    return A


def reconstruct(A,measurement,shape,steps=40,tv_weight=0.):
    """Normalized simultaneous iterative algebraic update + optional TV prox.
    TV step is heuristic proximal splitting, NOT exact constrained TV optimization.
    """
    nvox=np.prod(shape)
    x=np.zeros(nvox,dtype=np.float32)
    row_sum=np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    col_sum=np.asarray(A.sum(axis=0)).ravel().astype(np.float32)
    inv_row=np.divide(1.,row_sum,out=np.zeros_like(row_sum),where=row_sum>1e-8)
    inv_col=np.divide(1.,col_sum,out=np.zeros_like(col_sum),where=col_sum>1e-8)
    for _ in range(steps):
        x += .85*inv_col*(A.T@(inv_row*(measurement-A@x)))
        x=np.clip(x,0,1)
        if tv_weight:
            x=denoise_tv_chambolle(x.reshape(shape),weight=tv_weight,channel_axis=None).astype(np.float32).ravel()
    return x.reshape(shape)


def evaluate(truth,lesion,pred,A,y):
    ring=binary_dilation(lesion,iterations=4)&~binary_dilation(lesion,iterations=1)
    # oracle mask: lesion localization is known for diagnostic metrics
    contrast_true=truth[lesion].mean()-truth[ring].mean()
    contrast_pred=pred[lesion].mean()-pred[ring].mean()
    pts=np.argwhere(lesion)
    center=pts.mean(axis=0)
    # center and anisotropy computed on positive contrast inside fixed oracle neighborhood
    yy,zz,xx=np.indices(truth.shape)
    near=((yy-center[0])/7)**2+((zz-center[1])/9)**2+((xx-center[2])/9)**2<=1
    base=np.median(pred[ring])
    weights=np.maximum(pred-base,0)*near
    if weights.sum() <1e-8:
        center_err=None;depth_to_x=None
    else:
        centroid=np.array([(weights*a).sum()/weights.sum() for a in (yy,zz,xx)])
        var=[float((weights*(a-centroid[i])**2).sum()/weights.sum()) for i,a in enumerate((yy,zz,xx))]
        center_err=float(np.linalg.norm(centroid-center))
        depth_to_x=float(np.sqrt(var[1]/max(var[2],1e-8)))
    return dict(volume_mae=float(np.abs(pred-truth).mean()),lesion_mae=float(np.abs(pred[lesion]-truth[lesion]).mean()),contrast_error=float(abs(contrast_pred-contrast_true)),oracle_center_error=center_err,depth_to_x_spread=depth_to_x,projection_rmse=float(np.sqrt(np.mean((A@pred.ravel()-y)**2))))


def run(seeds=2,ny=16,size=24,views=11,extents=(18,40),steps=40,tv_weight=.008,noise=.005,output='results_v8'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    records=[]; examples={}
    for extent in extents:
        A=system_matrix(ny,size,views,extent,detector_u=size+4,detector_v=ny+4,samples=52)
        for seed in range(seeds):
            truth,lesion,_=phantom(seed+81000,ny,size)
            clean=A@truth.ravel()
            rng=np.random.default_rng(810000+seed+int(extent*10))
            observed=(clean+rng.normal(0,noise*max(float(clean.max()),1e-9),clean.shape)).astype(np.float32)
            for method,weight in [('sirt',0.),('sirt_tv',tv_weight)]:
                pred=reconstruct(A,observed,truth.shape,steps=steps,tv_weight=weight)
                records.append(dict(seed=seed,angle_extent=extent,method=method,**evaluate(truth,lesion,pred,A,observed)))
                if seed==0: examples[(extent,method)]=pred
            if seed==0: examples[(extent,'truth')]=truth
    keys=['volume_mae','lesion_mae','contrast_error','oracle_center_error','depth_to_x_spread','projection_rmse']
    means={str(e):{m:{k:float(np.mean([r[k] for r in records if r['angle_extent']==e and r['method']==m])) for k in keys} for m in ('sirt','sirt_tv')} for e in extents}
    summary={'model':'Synthetic 3D cone-beam-like line-integral projector with 2D flat detector; no scatter, polychromatic physics, exposure model, ultrasound, or clinical DBT calibration', 'shape':[ny,size,size],'parameters':dict(seeds=seeds,views=views,extents=list(extents),steps=steps,tv_weight=tv_weight,noise=noise),'means':means,'caveat':'Oracle lesion mask used for lesion-level diagnostics. TV prox does not guarantee improved lesion fidelity.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'per_seed_metrics.json').write_text(json.dumps(records,indent=2))
    truth=examples[(extents[0],'truth')]
    center=np.argwhere(phantom(81000,ny,size)[1]).mean(axis=0).round().astype(int)
    fig,axs=plt.subplots(2,5,figsize=(14,6))
    panels=[('Truth',truth)]+[(f'{m} ±{e}°',examples[(e,m)]) for e in extents for m in ('sirt','sirt_tv')]
    for i,(name,im) in enumerate(panels):
        axs[0,i].imshow(im[center[0]],vmin=0,vmax=.8,cmap='gray')
        axs[1,i].imshow(im[:,center[1],:],vmin=0,vmax=.8,cmap='gray',aspect='auto')
        axs[0,i].set_title(name,fontsize=10)
        for ax in axs[:,i]:ax.axis('off')
    fig.suptitle('V8: synthetic divergent cone-beam, 2D flat detector — not a calibrated DBT system')
    fig.tight_layout();fig.savefig(out/'v8_conebeam_comparison.png',dpi=150);plt.close(fig)
    print('V8 synthetic divergent cone-beam experiment; NOT clinical DBT')
    for e in extents:
        for m in ('sirt','sirt_tv'):
            s=means[str(e)][m]
            print(f'±{e:g}° {m:7} lesion MAE {s["lesion_mae"]:.4f} volume MAE {s["volume_mae"]:.4f} projRMSE {s["projection_rmse"]:.4f}')
    print('Saved to',out.resolve())
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=2);p.add_argument('--ny',type=int,default=16)
    p.add_argument('--size',type=int,default=24);p.add_argument('--views',type=int,default=11)
    p.add_argument('--extents',nargs='+',type=float,default=[18,40]);p.add_argument('--steps',type=int,default=40)
    p.add_argument('--tv-weight',type=float,default=.008);p.add_argument('--noise',type=float,default=.005)
    p.add_argument('--output',default='results_v8');run(**vars(p.parse_args()))
