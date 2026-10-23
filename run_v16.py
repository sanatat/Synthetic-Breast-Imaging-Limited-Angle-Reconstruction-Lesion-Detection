"""V16: synthetic lesion candidate detector ablation, with held-out evaluation.
Not a clinical model. One candidate per scan; no reference masks for prediction.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_v8 import system_matrix
from run_v14 import poisson_line_integrals
from run_v15 import stress_pair, sirt_blind

METHODS=('intensity', 'local_contrast', 'multiscale_log')

def candidate(volume, method):
    v=np.asarray(volume, dtype=np.float64)
    if v.ndim!=3 or min(v.shape)<8: raise ValueError('expected 3D volume dimensions >=8')
    if method=='intensity': response=gaussian_filter(v, 0.8)
    elif method=='local_contrast': response=gaussian_filter(v, 0.8)-gaussian_filter(v, 2.8)
    elif method=='multiscale_log':
        response=np.maximum.reduce([-s*s*gaussian_laplace(v,s) for s in (0.85,1.25,1.8)])
    else: raise ValueError('unknown method')
    margin=2
    response[:margin,:,:]=-np.inf; response[-margin:,:,:]=-np.inf
    response[:,:margin,:]=-np.inf; response[:,-margin:,:]=-np.inf
    response[:,:,:margin]=-np.inf; response[:,:,-margin:]=-np.inf
    center=np.array(np.unravel_index(np.argmax(response),v.shape))
    return {'center':center.tolist(), 'score':float(response[tuple(center)])}

def confusions(rows, method, threshold, tolerance=3.0):
    pos=[r for r in rows if r['label']=='positive']; neg=[r for r in rows if r['label']=='negative']
    # A positive scan counts as correctly detected only with a candidate near the true lesion.
    tp=sum(r['candidates'][method]['score']>threshold and r['candidates'][method]['distance_to_lesion_vox']<=tolerance for r in pos)
    fp_scans=sum(r['candidates'][method]['score']>threshold for r in neg)
    return {'positive_scans':len(pos),'negative_scans':len(neg),'localized_true_positives':tp,
        'missed_or_mislocalized_positives':len(pos)-tp,'false_positive_negative_scans':fp_scans,
        'localized_sensitivity':tp/len(pos) if pos else None,
        'fp_per_negative_scan':fp_scans/len(neg) if neg else None,
        'mean_candidate_center_error_positive_vox':float(np.mean([r['candidates'][method]['distance_to_lesion_vox'] for r in pos])) if pos else None}

def run(seeds=12,steps=70,views=15,photons=30000,output='results_v16'):
    if seeds<8 or steps<1 or views<3 or photons<=0: raise ValueError('invalid settings')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    A=system_matrix(ny=12,n=18,views=views,extent=18,detector_u=30,detector_v=24,samples=42)
    rows=[]; example=None
    for seed in range(seeds):
        positive,negative,mask,info=stress_pair(seed,12,18)
        center=np.argwhere(mask).mean(axis=0)
        for j,(label,truth) in enumerate((('positive',positive),('negative',negative))):
            y,_=poisson_line_integrals(A,truth,np.random.default_rng(880000+seed*2+j),photons,.035)
            rec=sirt_blind(A,y,truth.shape,steps)
            candidates={}
            for method in METHODS:
                cand=candidate(rec,method)
                cand['distance_to_lesion_vox']=float(np.linalg.norm(np.array(cand['center'])-center)) if label=='positive' else None
                candidates[method]=cand
            rows.append({'seed':seed,'label':label,'candidates':candidates,
                'lesion_voxels':info['lesion_voxels'] if label=='positive' else None})
            if seed==seeds-1 and label=='positive': example=(truth,rec,mask,candidates)
    split=seeds//2
    development=[r for r in rows if r['seed']<split]; held=[r for r in rows if r['seed']>=split]
    summary={'scope':'Synthetic paired hard-negative screening; NOT clinical evaluation',
       'settings':{'seeds':seeds,'steps':steps,'views':views,'photons':photons,'localization_tolerance_vox':3.0},
       'split':{'development_seed_ids':list(range(split)),'held_out_seed_ids':list(range(split,seeds))},'methods':{}}
    fig,axs=plt.subplots(1,3,figsize=(10,3.5))
    for method in METHODS:
        devneg=np.array([r['candidates'][method]['score'] for r in development if r['label']=='negative'])
        # Fixed 95th percentile of negative dev scores, no held-out label involved.
        threshold=float(np.quantile(devneg, .95))
        # FROC-style operating curve: thresholds are FIXED from development negative quantiles.
        thresholds=np.unique(np.quantile(devneg,[0,.25,.5,.75,.9,.95,1]))
        roc=[dict(threshold=float(t), **confusions(held,method,float(t))) for t in thresholds]
        summary['methods'][method]={'threshold_development_neg_95pct':threshold,
             'development':confusions(development,method,threshold),
             'held_out':confusions(held,method,threshold), 'froc_style_curve_held_out':roc}
        axs[0].plot([v['fp_per_negative_scan'] for v in roc], [v['localized_sensitivity'] for v in roc], marker='o',label=method)
    axs[0].set(xlabel='FP per negative scan',ylabel='Localized sensitivity',title='Held-out, candidate-level FROC-style')
    axs[0].set_xlim(-.03,1.05);axs[0].set_ylim(-.03,1.05);axs[0].legend(fontsize=7)
    truth,rec,mask,cands=example
    z=int(np.round(np.argwhere(mask).mean(axis=0)[0]))
    axs[1].imshow(truth[z],cmap='gray',vmin=0,vmax=.8);axs[1].set_title('Example: synthetic truth')
    axs[2].imshow(rec[z],cmap='gray',vmin=0,vmax=.8);axs[2].set_title('Example: reconstruction')
    for ax in axs[1:]:ax.axis('off')
    fig.tight_layout(); fig.savefig(output/'v16_froc_and_example.png',dpi=150);plt.close(fig)
    summary['limitations']=['Same projector used for simulation and reconstruction (inverse crime)',
      'Synthetic simplistic backgrounds and paired negative/positive volumes',
      'Only one maximum candidate per scan; curve is FROC-style, not full multiple-mark FROC',
      'Localization tolerance is an arbitrary 3 voxel radius, not a clinical criterion',
      'Small sample; ROC thresholds selected only from development negatives',
      'No clinical data, independent ultrasound, or proof of generalization']
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    (output/'per_seed_metrics.json').write_text(json.dumps(rows,indent=2))
    print('V16 synthetic held-out comparison (NOT clinical accuracy)')
    for method,res in summary['methods'].items():
        x=res['held_out'];print(f"{method:19} localized TP {x['localized_true_positives']}/{x['positive_scans']}  FP {x['false_positive_negative_scans']}/{x['negative_scans']}  mean center error {x['mean_candidate_center_error_positive_vox']:.2f}")
    print('Saved to',output.resolve())
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=12);p.add_argument('--steps',type=int,default=70)
    p.add_argument('--views',type=int,default=15);p.add_argument('--photons',type=float,default=30000)
    p.add_argument('--output',default='results_v16');run(**vars(p.parse_args()))
