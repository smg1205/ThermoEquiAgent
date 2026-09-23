"""Accept already converged sampled roots only after full thermodynamic checks.

The constructive decoder supplies roots analytically. Running Newton again is
unnecessary when activities and the original stability thresholds already pass.
If none pass, retain the original multi-start solver as a fallback. This routine
returns the certified sampled curve rather than adding extra generic-start roots.
"""
import numpy as np,torch
from .solver_cached import solve_many as fallback
from .solver import simplex_grid
from .model import repeat_context,select_context

def solve_many(model,ctx,proposals,iterations=28,equilibrium_tol=1e-3,tpd_tol=1e-4,grid_resolution=18):
    if int(ctx[-1][0].sum())!=3 or not getattr(model,'supports_cached_binary',False):
        return fallback(model,ctx,proposals,iterations,equilibrium_tol,tpd_tol,grid_resolution)
    if not bool((ctx[-1].sum(-1)==3).all()):raise ValueError('Homogeneous component count required')
    seeds=proposals.detach();batch,k=seeds.shape[:2];device=seeds.device;dtype=seeds.dtype
    extra=[]
    for i,j in [(0,1),(0,2),(1,2)]:
        for fraction in (.02,.1,.25,.4):
            a=torch.full((3,),fraction,device=device,dtype=dtype);a[i]=1-2*fraction
            b=torch.full((3,),fraction,device=device,dtype=dtype);b[j]=1-2*fraction;extra.append(torch.stack([a,b]))
    grid=torch.tensor(simplex_grid(3,grid_resolution),device=device,dtype=dtype)
    grid=torch.cat([grid[None].expand(batch,-1,-1),seeds.reshape(batch,-1,3),torch.stack(extra).reshape(1,-1,3).expand(batch,-1,-1)],1);g=grid.shape[1]
    gg=model.thermo(repeat_context(ctx,g),grid.reshape(-1,3),False)[3].detach().reshape(batch,g)
    ec=repeat_context(ctx,k)
    ma=model.thermo(ec,seeds[:,:,0].reshape(-1,3),False)[2].detach().reshape(batch,k,3)
    mb=model.thermo(ec,seeds[:,:,1].reshape(-1,3),False)[2].detach().reshape(batch,k,3)
    mm=model.thermo(ec,seeds.mean(2).reshape(-1,3),False)[2].detach().reshape(batch,k,3)
    mid=(gg[:,None]-torch.einsum('bki,bgi->bkg',mm,grid)).min(-1).values
    tpa=(gg[:,None]-torch.einsum('bki,bgi->bkg',ma,grid)).min(-1).values
    tpb=(gg[:,None]-torch.einsum('bki,bgi->bkg',mb,grid)).min(-1).values
    rms=(ma-mb).square().mean(-1).sqrt();tpd=torch.minimum(tpa,tpb)
    good=(mid < -tpd_tol)&(rms<=equilibrium_tol)&(tpd>=-tpd_tol)&((seeds[:,:,0]-seeds[:,:,1]).abs().sum(-1)>.005)
    arrays=[v.cpu().numpy() for v in (seeds,rms,tpd,mid,good)]
    pairs,rms,tpd,mid,good=arrays;results=[]
    for i in range(batch):
        if not good[i].any():
            results.append(fallback(model,select_context(ctx,slice(i,i+1)),proposals[i:i+1],iterations,equilibrium_tol,tpd_tol,grid_resolution)[0]);continue
        accepted=[]
        for j in np.where(good[i])[0]:
            pair=pairs[i,j]
            if tuple(pair[0])>tuple(pair[1]):pair=pair[::-1].copy()
            if any(np.linalg.norm(pair-np.array(v['endpoints']))<1e-4 for v in accepted):continue
            accepted.append({'endpoints':pair.tolist(),'equilibrium_rms':float(rms[i,j]),'minimum_grid_tpd':float(tpd[i,j])})
        results.append({'status':'converged','pairs':accepted,'attempts':k,'unstable_starts':int((mid[i]<-tpd_tol).sum()),'min_midpoint_tpd':float(mid[i].min()),'grid_resolution':grid_resolution,'iterations_used':0,'method':'certified_sampled_roots'})
    return results
