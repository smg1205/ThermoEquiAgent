"""Phase-invariant set metrics; failures remain in aggregate coverage/error."""
import numpy as np

def distance(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    direct=((a[:,None]-b[None])**2).mean((-1,-2))
    swapped=((a[:,None]-b[None,:,::-1])**2).mean((-1,-2))
    return np.minimum(direct,swapped),swapped<direct

def score(records,field='predicted'):
    distances=[]; coverage=[]; y=[]; pred=[]; success=0; maxeq=[]; mintpd=[]
    for r in records:
        truth=np.asarray(r['observed']); estimate=np.asarray(r[field])
        if not len(estimate): distances.append(1.0); coverage.append(0.0); continue
        success+=1
        d,swap=distance(estimate,truth)
        distances.append(float(0.5*(np.sqrt(d.min(0)).mean()+np.sqrt(d.min(1)).mean())))
        coverage.append(float((np.sqrt(d.min(0))<=0.02).mean()))
        ii=d.argmin(0)
        for j,i in enumerate(ii):
            p=estimate[i][::-1] if swap[i,j] else estimate[i]
            # Only independent mole fractions enter R2, without padded/complement coordinates.
            y.extend(truth[j,:,:-1].flatten().tolist()); pred.extend(p[:,:-1].flatten().tolist())
        if field=='predicted' and r.get('solver',{}).get('pairs'):
            maxeq.extend(p['equilibrium_rms'] for p in r['solver']['pairs'])
            mintpd.extend(p['minimum_grid_tpd'] for p in r['solver']['pairs'])
    y=np.asarray(y); pred=np.asarray(pred)
    sst=((y-y.mean())**2).sum() if len(y) else 0
    r2=float(1-((y-pred)**2).sum()/sst) if sst>1e-12 else None
    return {'conditions':len(records),'successful_conditions':success,'success_fraction':success/len(records) if records else None,'failure_penalized_set_distance':float(np.mean(distances)) if distances else None,'observed_coverage_at_0.02':float(np.mean(coverage)) if coverage else None,'matched_endpoint_r2_successful_only':r2,'matched_endpoint_mae_successful_only':float(np.abs(y-pred).mean()) if len(y) else None,'matched_endpoint_rmse_successful_only':float(np.sqrt(np.mean((y-pred)**2))) if len(y) else None,'max_equilibrium_rms':max(maxeq) if maxeq else None,'min_grid_tpd':min(mintpd) if mintpd else None,'r2_target_met':bool(field=='predicted' and records and success==len(records) and r2 is not None and r2>0.95)}
