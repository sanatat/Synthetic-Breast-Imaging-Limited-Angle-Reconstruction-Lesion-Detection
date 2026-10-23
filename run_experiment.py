"""CPU-only synthetic limited-angle X-ray / ultrasound-guided reconstruction pilot.

Educational feasibility study, NOT a clinical or physics-faithful DBT/US simulator.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, shift
from skimage.draw import ellipse
from skimage.transform import radon, iradon, iradon_sart, resize
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def phantom(n=96, seed=12):
    rng=np.random.default_rng(seed)
    yy,xx=np.mgrid[:n,:n]; x=(xx-(n-1)/2)/(n/2); y=(yy-(n-1)/2)/(n/2)
    breast=(x*x/0.86**2+y*y/0.9**2)<1
    img=np.zeros((n,n),float)
    img[breast]=0.27+0.10*np.exp(-((x[breast]+0.25)**2+(y[breast]-0.1)**2)/0.2)
    tissue=np.zeros_like(img)
    for _ in range(12):
        cx,cy=rng.uniform(-0.5,0.5,2); sx,sy=rng.uniform(0.07,0.22,2)
        tissue+=rng.uniform(0.02,0.085)*np.exp(-((x-cx)**2/sx**2+(y-cy)**2/sy**2))
    img+=tissue*breast
    lesion=((x-0.19)**2/0.12**2+(y+0.11)**2/0.095**2)<1
    img[lesion]=0.83
    img=gaussian_filter(img,0.65)
    return np.clip(img,0,1),lesion.astype(bool)


def ultrasound_surrogate(truth,rng,misregistration=0.0):
    # Image-like *surrogate*, not acoustic wave propagation, reflection or beamforming.
    us=gaussian_filter(truth,sigma=(1.8,0.9))
    speckle=rng.normal(0,0.16,truth.shape)
    speckle=gaussian_filter(speckle,0.65)
    us=np.clip(us*(1+speckle)+rng.normal(0,0.025,truth.shape),0,1)
    if misregistration:
        us=shift(us,shift=(misregistration,-misregistration),order=1,mode='constant')
    return us


def xray_recon(truth,rng,angle_extent=25,n_angles=21,noise_fraction=0.015,sart_iterations=2):
    angles=np.linspace(-angle_extent,angle_extent,n_angles)
    sino=radon(truth,theta=angles,circle=True)
    noisy=sino+rng.normal(0, noise_fraction*max(float(sino.max()),1e-12),sino.shape)
    fbp=iradon(noisy,theta=angles,filter_name='ramp',circle=True,output_size=truth.shape[0])
    sart=np.zeros_like(truth)
    for _ in range(sart_iterations):
        sart=iradon_sart(noisy,theta=angles,image=sart,relaxation=0.12)
        sart=np.maximum(sart,0)
    return angles,noisy,np.clip(fbp,0,1),np.clip(sart,0,1)


def diffuse(img,guide,steps=22,step_size=0.13,edge_strength=0.12):
    """Diffusion with conductance determined by guide image gradients.

    Baseline uses reconstructed X-ray as guide; joint uses ultrasound as guide.
    This is ultrasound-guided post-processing, NOT joint raw-data reconstruction.
    """
    out=img.copy()
    guide=gaussian_filter(guide,1.0)
    diff_h=np.diff(guide,axis=1); diff_v=np.diff(guide,axis=0)
    conduct_h=np.exp(-(diff_h/edge_strength)**2)
    conduct_v=np.exp(-(diff_v/edge_strength)**2)
    for _ in range(steps):
        dh=conduct_h*np.diff(out,axis=1)
        dv=conduct_v*np.diff(out,axis=0)
        flux=np.zeros_like(out)
        flux[:,:-1]+=dh; flux[:,1:]-=dh
        flux[:-1,:]+=dv; flux[1:,:]-=dv
        out=np.clip(out+step_size*flux,0,1)
    return out


def metrics(truth, pred, lesion):
    bg=(truth>0.15)&(~lesion)
    return {'ssim':float(structural_similarity(truth,pred,data_range=1.0)),
            'psnr_db':float(peak_signal_noise_ratio(truth,pred,data_range=1.0)),
            'mae':float(np.mean(np.abs(truth-pred))),
            'lesion_mean_abs_error':float(np.mean(np.abs(truth[lesion]-pred[lesion]))),
            'lesion_background_contrast':float(np.mean(pred[lesion])-np.mean(pred[bg]))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='results'); ap.add_argument('--seeds',type=int,default=4)
    ap.add_argument('--size',type=int,default=96)
    args=ap.parse_args(); outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)
    records=[]; example=None
    for seed in range(args.seeds):
        rng=np.random.default_rng(1000+seed)
        truth,lesion=phantom(args.size,seed=seed+12)
        angles,sinogram,fbp,sart=xray_recon(truth,rng)
        us=ultrasound_surrogate(truth,rng)
        us_shift=ultrasound_surrogate(truth,rng,misregistration=5)
        xonly=diffuse(sart,sart)
        guided=diffuse(sart,us)
        shifted=diffuse(sart,us_shift)
        preds={'fbp':fbp,'sart':sart,'xray_guided_diffusion':xonly,
               'ultrasound_guided_diffusion':guided,'misregistered_ultrasound':shifted}
        for name,pred in preds.items():
            records.append({'seed':seed,'method':name,**metrics(truth,pred,lesion)})
        if seed==0: example=(truth,lesion,us,fbp,sart,xonly,guided,shifted,sinogram)
    (outdir/'per_seed_metrics.json').write_text(json.dumps(records,indent=2))
    summary={}
    for name in sorted(set(x['method'] for x in records)):
        rows=[r for r in records if r['method']==name]
        summary[name]={k:{'mean':float(np.mean([r[k] for r in rows])), 'std':float(np.std([r[k] for r in rows],ddof=1)) if len(rows)>1 else 0.0} for k in ('ssim','psnr_db','mae','lesion_mean_abs_error','lesion_background_contrast')}
    (outdir/'summary.json').write_text(json.dumps({'n_seeds':args.seeds,'model':'2D limited-angle Radon; ultrasound appearance surrogate, not physics-based','methods':summary},indent=2))
    images=[example[i] for i in (0,2,3,4,5,6,7)]
    names=['Ground truth','US surrogate','Limited-angle FBP','SART','X-ray-guided diffusion','US-guided diffusion','US with 5px misregistration']
    fig,axes=plt.subplots(2,4,figsize=(14,7)); axes=axes.ravel()
    for a,im,name in zip(axes,images,names):
        a.imshow(im,cmap='gray',vmin=0,vmax=1); a.set_title(name,fontsize=10); a.axis('off')
    axes[7].imshow(example[-1],cmap='magma',aspect='auto'); axes[7].set_title('Noisy X-ray projections',fontsize=10)
    fig.tight_layout(); fig.savefig(outdir/'reconstruction_comparison.png',dpi=150);plt.close(fig)
    print('Metric averages across',args.seeds,'independent phantom/noise seeds:')
    for method in ['fbp','sart','xray_guided_diffusion','ultrasound_guided_diffusion','misregistered_ultrasound']:
        m=summary[method]; print(f'{method:31s} SSIM {m["ssim"]["mean"]:.3f}  PSNR {m["psnr_db"]["mean"]:.2f} dB  lesion MAE {m["lesion_mean_abs_error"]["mean"]:.3f}')
    print('Saved to',outdir.resolve())

if __name__=='__main__':main()
