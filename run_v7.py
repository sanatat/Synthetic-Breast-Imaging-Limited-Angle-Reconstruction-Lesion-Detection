"""V7: CPU-friendly volumetric *parallel-ray* limited-angle tomography.
3D volume (y,z,x), 2D detector (y,u), rotating rays in x-z plane.
Not cone-beam DBT; not a clinical breast model or ultrasound reconstruction.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.transform import radon, iradon_sart
from skimage.restoration import denoise_tv_chambolle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def phantom(seed=0, ny=32, n=48):
    rng=np.random.default_rng(seed)
    yy,zz,xx=np.meshgrid(np.arange(ny),np.arange(n),np.arange(n),indexing='ij')
    cy=(ny-1)/2; cz=(n-1)/2; cx=(n-1)/2
    # Every x-z slice remains inside the Radon reconstruction circle.
    body=((xx-cx)/(.41*n))**2+((zz-cz)/(.41*n))**2+((yy-cy)/(.47*ny))**2<1
    vol=body*(.20+.025*np.sin(xx/6)+.025*np.cos(yy/5))
    # Randomized lesion center and physical radii across seeds.
    ly=cy+rng.uniform(-.14*ny,.14*ny)
    lz=cz+rng.uniform(-.10*n,.10*n)
    lx=cx+rng.uniform(-.10*n,.10*n)
    ry=rng.uniform(2.3,3.2); rz=rng.uniform(2.8,3.7); rx=rng.uniform(2.8,3.7)
    lesion=((yy-ly)/ry)**2+((zz-lz)/rz)**2+((xx-lx)/rx)**2<=1
    vol=np.clip(vol+lesion*.55,0,1).astype(np.float32)
    return vol, lesion, (ly,lz,lx)


def forward(vol,theta):
    # skimage radon operates on each 2D x-z slab; stacking gives detector (y,u,angle)
    return np.stack([radon(sl,theta=theta,circle=True).astype(np.float32) for sl in vol],axis=0)


def reconstruct(sino,theta,method='sart',iterations=3,tv_weight=.025):
    ny,n,views=sino.shape
    vol=np.zeros((ny,n,n),dtype=np.float32)
    zz,xx=np.ogrid[:n,:n]; mask=(xx-(n-1)/2)**2+(zz-(n-1)/2)**2<=(n/2-1)**2
    for _ in range(iterations):
        for i in range(ny):
            vol[i]=np.clip(iradon_sart(sino[i],theta=theta,image=vol[i],relaxation=.15),0,1)
        vol*=mask[None]
        if method=='tv':
            vol=np.clip(denoise_tv_chambolle(vol,weight=tv_weight,channel_axis=None),0,1).astype(np.float32)
    return vol


def metrics(truth,lesion,pred,theta,sino):
    # This is an oracle ROI diagnostic; NOT a blind detection result.
    ring=binary_dilation(lesion,iterations=5)&~binary_dilation(lesion,iterations=2)
    true_contrast=float(truth[lesion].mean()-truth[ring].mean())
    contrast=float(pred[lesion].mean()-pred[ring].mean())
    # Weighted lesion shape with local background removed. Report depth/in-plane spread ratio.
    pts=np.argwhere(lesion)
    c=pts.mean(axis=0)
    yy,zz,xx=np.ogrid[:truth.shape[0],:truth.shape[1],:truth.shape[2]]
    roi=((yy-c[0])/8)**2+((zz-c[1])/9)**2+((xx-c[2])/9)**2<1
    background=float(np.median(pred[ring]))
    w=np.where(roi,np.maximum(pred-background,0),0)
    if w.sum()>1e-8:
        wc=np.array([(w*v).sum()/w.sum() for v in (yy,zz,xx)])
        var=np.array([(w*(v-wc[j])**2).sum()/w.sum() for j,v in enumerate((yy,zz,xx))])
        aspect=float(np.sqrt(var[1]/max(var[2],1e-8))) # z depth vs x lateral
        localization=float(np.linalg.norm(wc-c))
    else:
        aspect=float('nan');localization=float('nan')
    return dict(volume_mae=float(np.abs(pred-truth).mean()),lesion_mae=float(np.abs(pred[lesion]-truth[lesion]).mean()),contrast_error=float(abs(contrast-true_contrast)),depth_to_lateral_spread=aspect,oracle_roi_centroid_error=localization,projection_rmse=float(np.sqrt(np.mean((forward(pred,theta)-sino)**2))))


def run(seeds=4,ny=24,size=40,views=17,extents=(18,60),iterations=3,tv_weight=.025,noise=.01,output='results_v7'):
    if not (1<=seeds<=100 and ny>=12 and size>=24 and views>=5 and iterations>=1 and all(0<e<=89 for e in extents)):
        raise ValueError('Invalid parameters')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    records=[];example=None
    for seed in range(seeds):
        truth,lesion,_=phantom(seed+70000,ny,size)
        for extent in extents:
            theta=np.linspace(-extent,extent,views)
            clean=forward(truth,theta)
            rng=np.random.default_rng(seed+400000+round(extent*100))
            observed=clean+rng.normal(0,noise*max(clean.max(),1e-9),clean.shape)
            for method in ('sart','tv'):
                pred=reconstruct(observed,theta,method,iterations,tv_weight)
                records.append(dict(seed=seed,extent=float(extent),method=method,**metrics(truth,lesion,pred,theta,observed)))
                if seed==0 and (extent==extents[0] or extent==extents[-1]):
                    if example is None:example={'truth':truth,'lesion':lesion,'recons':{}}
                    example['recons'][f'{method} ±{extent}°']=pred
    numeric=('volume_mae','lesion_mae','contrast_error','depth_to_lateral_spread','oracle_roi_centroid_error','projection_rmse')
    means={str(e):{m:{k:float(np.nanmean([r[k] for r in records if r['extent']==e and r['method']==m])) for k in numeric} for m in ('sart','tv')} for e in extents}
    summary={'description':'3D volume / 2D detector LIMITED-ANGLE PARALLEL-RAY toy model; not cone-beam clinical DBT','parameters':dict(seeds=seeds,ny=ny,size=size,views=views,extents=list(extents),iterations=iterations,tv_weight=tv_weight,noise=noise),'means':means,'limitations':['Parallel ray slices are separable across detector rows; this does NOT model cone beam divergence, rotation or detector motion of DBT','Synthetic volumes and known ground-truth lesion masks only','Oracle ROI shape/localization are not blinded lesion detection','TV is SART interleaved with Chambolle TV denoising, not exact minimization of a joint objective','No independent ultrasound measurements','Only a few seeds by default; do not generalize to patient imaging']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'per_seed_metrics.json').write_text(json.dumps(records,indent=2))
    truth=example['truth'];lesion=example['lesion']; p=np.argwhere(lesion).mean(axis=0).round().astype(int)
    fig,axs=plt.subplots(2,5,figsize=(14,6.1))
    stack=[('Truth',truth)]+list(example['recons'].items())
    for j,(name,vol) in enumerate(stack[:5]):
        axs[0,j].imshow(vol[p[0]],vmin=0,vmax=.8,cmap='gray');axs[0,j].set_title(name,fontsize=10)
        axs[1,j].imshow(vol[:,p[1],:],vmin=0,vmax=.8,cmap='gray',aspect='auto')
        for a in (axs[0,j],axs[1,j]):a.axis('off')
    axs[0,0].set_ylabel('XY plane');axs[1,0].set_ylabel('Depth view')
    fig.suptitle('V7: 3D parallel-ray limited-angle toy experiment (not cone-beam DBT)',fontsize=11)
    fig.tight_layout();fig.savefig(out/'v7_volume_comparison.png',dpi=140);plt.close(fig)
    print('V7: CPU 3D parallel-ray toy model, NOT cone-beam DBT')
    for e in extents:
        for m in ('sart','tv'):
            a=means[str(e)][m]
            print(f'±{e:g}° {m.upper():4} volumeMAE={a["volume_mae"]:.4f} lesionMAE={a["lesion_mae"]:.4f} depth/x spread={a["depth_to_lateral_spread"]:.3f} projRMSE={a["projection_rmse"]:.3f}')
    print('Saved',out.resolve())
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=4);parser.add_argument('--ny',type=int,default=24);parser.add_argument('--size',type=int,default=40);parser.add_argument('--views',type=int,default=17);parser.add_argument('--extents',type=float,nargs='+',default=[18,60]);parser.add_argument('--iterations',type=int,default=3);parser.add_argument('--tv-weight',type=float,default=.025);parser.add_argument('--noise',type=float,default=.01);parser.add_argument('--output',default='results_v7')
    run(**vars(parser.parse_args()))
