"""V9: cone-beam toy phantom sweep. Synthetic, oracle lesion ROI, not clinical DBT.
Select TV weight only on development seeds; assess held-out seeds separately.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_v7 import phantom
from run_v8 import system_matrix, reconstruct, evaluate


def lesion_diagnostics(truth,mask,pred):
    """Oracle-centered local positive-contrast moment/threshold metrics, not blind detection."""
    pts=np.argwhere(mask); target=pts.mean(axis=0)
    yy,zz,xx=np.indices(truth.shape)
    roi=((yy-target[0])/6)**2+((zz-target[1])/7)**2+((xx-target[2])/7)**2<=1
    shell=binary_dilation(mask,iterations=3)&~binary_dilation(mask,iterations=1)
    bg=float(np.median(pred[shell]))
    reference_contrast=float(truth[mask].mean()-np.median(truth[shell]))
    # Fixed threshold derived from synthetic truth: known lesion intensity is an oracle.
    binary=(pred>bg+0.5*reference_contrast)&roi
    if binary.sum()<2:
        return {'oracle_detected_volume_ratio':0.,'oracle_binary_centroid_error':None,'oracle_depth_lateral_ratio':None}
    where=np.argwhere(binary); centroid=where.mean(axis=0)
    spread=where.std(axis=0)
    return {'oracle_detected_volume_ratio':float(binary.sum()/mask.sum()),
            'oracle_binary_centroid_error':float(np.linalg.norm(centroid-target)),
            'oracle_depth_lateral_ratio':float(spread[1]/max(spread[2],1e-6))}


def select_weight(records,weights,projection_tolerance=1.35):
    """Development-only selection: smallest lesion MAE, subject to data fidelity tolerance.
    If no TV candidate satisfies projection tolerance, select zero weight.
    This uses synthetic truth in DEVELOPMENT only, so not deployable for patient data.
    """
    summaries={}
    for w in weights:
        rows=[r for r in records if r['weight']==w]
        summaries[str(w)]={'lesion_mae':float(np.mean([r['lesion_mae'] for r in rows])),
                           'projection_rmse':float(np.mean([r['projection_rmse'] for r in rows]))}
    baseline=summaries[str(weights[0])]['projection_rmse']
    valid=[w for w in weights if summaries[str(w)]['projection_rmse']<=max(baseline,1e-10)*projection_tolerance]
    best=min(valid,key=lambda w:summaries[str(w)]['lesion_mae']) if valid else weights[0]
    return best,summaries


def run(dev_seeds=2,test_seeds=3, views=(9,15,21),weights=(0.,.002,.008),steps=25,
        extent=18.,ny=16,size=24,noise=.005,output='results_v9'):
    if dev_seeds<1 or test_seeds<1 or len(views)<1 or weights[0]!=0 or sorted(weights)!=list(weights):
        raise ValueError('Invalid experimental settings')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    records=[];final=[];selected={};figure=None
    for nview in views:
        A=system_matrix(ny=ny,n=size,views=nview,extent=extent,detector_u=size+4,detector_v=ny+4,samples=52)
        development=[]
        for split,count in [('development',dev_seeds),('holdout',test_seeds)]:
            for seed in range(count):
                label=91000+seed if split=='development' else 99000+seed
                truth,lesion,_=phantom(label,ny,size)
                clean=A@truth.ravel()
                rng=np.random.default_rng(label+int(nview)*100)
                measured=(clean+rng.normal(0,noise*max(clean.max(),1e-8),clean.shape)).astype(np.float32)
                for w in weights:
                    pred=reconstruct(A,measured,truth.shape,steps=steps,tv_weight=w)
                    d=evaluate(truth,lesion,pred,A,measured)
                    diag=lesion_diagnostics(truth,lesion,pred)
                    record={'split':split,'seed':seed,'views':nview,'weight':w,**d,**diag}
                    records.append(record)
                    if split=='development':development.append(record)
                    if split=='holdout' and seed==0 and nview==max(views) and w in (0.,weights[-1]):
                        if figure is None:figure={'truth':truth,'lesion':lesion}
                        figure[f'weight_{w}']=pred
        chosen,detail=select_weight(development,list(weights))
        selected[str(nview)]={'chosen_tv_weight':chosen,'development_means':detail}
        rows=[r for r in records if r['views']==nview and r['split']=='holdout' and r['weight']==chosen]
        base=[r for r in records if r['views']==nview and r['split']=='holdout' and r['weight']==0]
        for kind,subset in [('sirt',base),('selected',rows)]:
            means={k:float(np.mean([r[k] for r in subset])) for k in ('lesion_mae','volume_mae','contrast_error','projection_rmse')}
            means['oracle_binary_detection_fraction']=float(np.mean([r['oracle_detected_volume_ratio']>0 for r in subset]))
            finite=[r['oracle_binary_centroid_error'] for r in subset if r['oracle_binary_centroid_error'] is not None]
            means['oracle_binary_centroid_error']=float(np.mean(finite)) if finite else None
            finite=[r['oracle_depth_lateral_ratio'] for r in subset if r['oracle_depth_lateral_ratio'] is not None]
            means['oracle_depth_lateral_ratio']=float(np.mean(finite)) if finite else None
            final.append({'views':nview,'method':kind,'weight':0. if kind=='sirt' else chosen,**means})
    result={'description':'CPU synthetic divergent-ray cone beam, 2D detector; NOT calibrated breast tomosynthesis',
            'settings':dict(dev_seeds=dev_seeds,test_seeds=test_seeds,views=list(views),weights=list(weights),steps=steps,extent=extent,noise=noise,shape=[ny,size,size]),
            'selection':selected,'holdout_means':final,
            'limitations':['Tiny synthetic ground truths; no real US, patient images, calibrated geometry or dose physics',
                           'TV is interleaved Chambolle denoising, not exact TV minimization',
                           'Lesion geometry metrics depend on ground-truth oracle ROI and threshold',
                           'Holdout seeds differ but share the same phantom generating family',
                           'Weights chosen using known lesion truth on development seeds only']}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'per_seed_metrics.json').write_text(json.dumps(records,indent=2))
    if figure is not None:
        loc=np.argwhere(figure['lesion']).mean(axis=0).round().astype(int)
        images=[('Truth',figure['truth']),('SIRT',figure['weight_0.0']),('TV weight '+str(weights[-1]),figure['weight_'+str(weights[-1])])]
        fig,ax=plt.subplots(2,3,figsize=(9,6))
        for i,(name,im) in enumerate(images):
            ax[0,i].imshow(im[loc[0]],cmap='gray',vmin=0,vmax=.8)
            ax[1,i].imshow(im[:,loc[1],:],cmap='gray',vmin=0,vmax=.8,aspect='auto')
            ax[0,i].set_title(name)
            for a in ax[:,i]:a.axis('off')
        fig.suptitle('V9: held-out synthetic phantom, %d views, ±%.0f° (not clinical DBT)'%(max(views),extent))
        fig.tight_layout();fig.savefig(out/'v9_reconstruction_comparison.png',dpi=135);plt.close(fig)
    print('V9 holdout results (synthetic only)')
    for row in final:print(f"{row['views']:2d} views {row['method']:8} w={row['weight']:.4f} lesionMAE={row['lesion_mae']:.4f} projRMSE={row['projection_rmse']:.4f} centroid={row['oracle_binary_centroid_error']}")
    print('Saved:',out.resolve())
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dev-seeds',type=int,default=2);p.add_argument('--test-seeds',type=int,default=3)
    p.add_argument('--views',nargs='+',type=int,default=[9,15,21]);p.add_argument('--weights',nargs='+',type=float,default=[0.,.002,.008])
    p.add_argument('--steps',type=int,default=25);p.add_argument('--extent',type=float,default=18.)
    p.add_argument('--output',default='results_v9');p.add_argument('--ny',type=int,default=16);p.add_argument('--size',type=int,default=24);p.add_argument('--noise',type=float,default=.005)
    run(**vars(p.parse_args()))
