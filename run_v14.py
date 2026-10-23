"""V14 synthetic photon-counting proxy + blind candidate evaluation.

Physics scope: monoenergetic Beer-Lambert Poisson counts on the V8 educational
ray matrix. NOT a calibrated DBT model or clinical sensitivity assessment.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_v7 import phantom
from run_v8 import system_matrix
from run_v10 import sirt_trace


def poisson_line_integrals(A, volume, rng, photons=30000, attenuation=0.035):
    """Photon counts then negative log; zeros handled by half-count floor."""
    if photons <= 0 or attenuation <= 0: raise ValueError('photons and attenuation must be positive')
    length=np.asarray(A @ volume.ravel(),dtype=np.float32)
    transmission=np.exp(-attenuation*np.maximum(length,0))
    counts=rng.poisson(photons*transmission)
    measured=-np.log(np.maximum(counts,0.5)/photons)/attenuation
    return measured.astype(np.float32), counts


def remove_lesion(truth, lesion):
    """Remove added 0.55 contrast without using the mask during detection."""
    negative=truth.copy()
    negative[lesion]=np.maximum(negative[lesion]-.55,0)
    return negative


def blind_candidate(volume):
    """One global candidate per volume, no reference lesion mask or location.

    Exclude outer border and subtract broad smoothed background. This baseline
    can find anatomy rather than lesions; we measure its false alarms explicitly.
    """
    vol=np.asarray(volume,dtype=np.float32)
    smooth=gaussian_filter(vol, .85)
    background=gaussian_filter(vol, 3.0)
    response=smooth-background
    margin=2
    response[:margin]=-np.inf; response[-margin:]=-np.inf
    response[:,:margin]=-np.inf;response[:,-margin:]=-np.inf
    response[:,:,:margin]=-np.inf;response[:,:,-margin:]=-np.inf
    center=np.unravel_index(int(np.argmax(response)),response.shape)
    return {'center':list(map(int,center)), 'score':float(response[center])}


def run(seeds=16, steps=80, views=15, photons=30000, attenuation=.035,
        size=18, ny=12, extent=18., output='results_v14'):
    if seeds<6 or steps<1 or views<3 or size<12 or ny<8: raise ValueError('invalid run parameters')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    A=system_matrix(ny=ny,n=size,views=views,extent=extent,
                    detector_u=30,detector_v=24,samples=42)
    records=[]; example=None
    for seed in range(seeds):
        truth,mask,_=phantom(140000+seed,ny,size)
        truth_negative=remove_lesion(truth,mask)
        true_center=np.argwhere(mask).mean(axis=0)
        for label,vol in [('positive',truth),('negative',truth_negative)]:
            rng=np.random.default_rng(1400000+2*seed+(label=='negative'))
            measurement, counts=poisson_line_integrals(A,vol,rng,photons,attenuation)
            # No oracle mask enters reconstruction or candidate identification;
            # one mask passed solely for post-hoc metrics within sirt_trace.
            trace,recon=sirt_trace(A,measurement,vol,mask,(steps,))
            candidate=blind_candidate(recon)
            center_err=float(np.linalg.norm(np.array(candidate['center'])-true_center)) if label=='positive' else None
            records.append({'seed':seed,'label':label,'candidate':candidate,'center_error_vox':center_err,
                            'oracle_lesion_mae':trace[-1]['lesion_mae'] if label=='positive' else None,
                            'projection_rmse':trace[-1]['projection_rmse'],
                            'zero_count_fraction':float(np.mean(counts==0))})
            if seed==seeds-1 and label=='positive':example=(truth,recon,mask,candidate)
    split=seeds//2
    train=[r for r in records if r['seed']<split]
    test=[r for r in records if r['seed']>=split]
    negatives=[r['candidate']['score'] for r in train if r['label']=='negative']
    threshold=float(np.quantile(negatives,.95))
    def stats(group):
        pos=[r for r in group if r['label']=='positive'];neg=[r for r in group if r['label']=='negative']
        det=[r for r in pos if r['candidate']['score']>threshold]
        return {'positives':len(pos),'negatives':len(neg),
                'sensitivity_at_threshold':float(len(det)/len(pos)),
                'false_positive_rate':float(sum(r['candidate']['score']>threshold for r in neg)/len(neg)),
                'mean_localization_error_vox_all_positive':float(np.mean([r['center_error_vox'] for r in pos])),
                'mean_oracle_lesion_mae':float(np.mean([r['oracle_lesion_mae'] for r in pos])),
                'mean_projection_rmse':float(np.mean([r['projection_rmse'] for r in group]))}
    summary={'scope':'Educational sparse cone-beam, Poisson photon-count proxy; no real DBT/ultrasound/patient data.',
             'settings':{'seeds':seeds,'train_seed_count':split,'test_seed_count':seeds-split,
                         'steps':steps,'views':views,'photons_per_ray':photons,
                         'linear_attenuation_scale':attenuation,'shape':[ny,size,size],'angle_extent_deg':extent},
             'threshold_from_train_negative_95pct':threshold,
             'development':stats(train),'held_out':stats(test),
             'limitations':['Candidate location is chosen without reference mask but phantom construction remains simplistic.',
                            'One maximum per volume; this is NOT a full lesion detector.',
                            'Threshold estimated from small negative development set; false positive rate imprecise.',
                            'Photon flux fixed per ray, not a calibrated dose/exposure model.',
                            'No scatter, beam hardening, detector blur, anatomical variability, independent projection model or external dataset.',
                            'Oracle lesion mask used solely for retrospective error metrics.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'per_seed_metrics.json').write_text(json.dumps(records,indent=2))
    truth,recon,mask,candidate=example
    c=int(np.argwhere(mask).mean(axis=0)[0])
    fig,axes=plt.subplots(1,3,figsize=(11,3.8))
    axes[0].imshow(truth[c],cmap='gray',vmin=0,vmax=.8)
    axes[1].imshow(recon[c],cmap='gray',vmin=0,vmax=.8)
    axes[2].imshow(recon[c],cmap='gray',vmin=0,vmax=.8)
    axes[2].plot(candidate['center'][2],candidate['center'][1],'rx',markersize=10,label='blind candidate')
    for ax,t in zip(axes,['Synthetic truth','SIRT reconstruction','Blind candidate (3D peak)']):
        ax.set_title(t);ax.axis('off')
    axes[2].legend(loc='lower center',fontsize=8)
    fig.suptitle('V14: synthetic photon-counting proxy — NOT a clinical detector')
    fig.tight_layout();fig.savefig(out/'v14_reconstruction.png',dpi=145);plt.close(fig)
    print('V14 synthetic Poisson experiment (not clinical DBT)')
    print('threshold from development negatives:',round(threshold,5))
    print('held-out metrics:',json.dumps(summary['held_out'],indent=2))
    print('Saved:',out.resolve())
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=16)
    p.add_argument('--steps',type=int,default=80)
    p.add_argument('--views',type=int,default=15)
    p.add_argument('--photons',type=float,default=30000)
    p.add_argument('--attenuation',type=float,default=.035)
    p.add_argument('--output',default='results_v14')
    run(**vars(p.parse_args()))
