"""Batch-check exact binary candidates; retain the original solver as fallback."""
import numpy as np,torch
from .solver_batch import solve_many as original_solve_many
from .solver import simplex_grid
from .model import repeat_context,select_context

def solve_many(model,ctx,proposals,iterations=28,equilibrium_tol=1e-3,tpd_tol=1e-4,grid_resolution=18):
    if int(ctx[-1][0].sum())!=2 or not getattr(model,'supports_cached_binary',False):
        return original_solve_many(model,ctx,proposals,iterations,equilibrium_tol,tpd_tol,grid_resolution)
    if not bool((ctx[-1].sum(-1)==2).all()):raise ValueError('Homogeneous component count required')
    proposals=proposals.detach()
    batch=len(ctx[0]);device=ctx[0].device;dtype=ctx[0].dtype
    extra=torch.tensor([[[.01,.99,0],[.99,.01,0]],[[.1,.9,0],[.9,.1,0]],[[.3,.7,0],[.7,.3,0]]],device=device,dtype=dtype)
    seeds=torch.cat([proposals[:,:1],extra[None].expand(batch,-1,-1,-1)],1)
    grid=torch.tensor(np.pad(simplex_grid(2,grid_resolution),((0,0),(0,1))),device=device,dtype=dtype)
    grid=torch.cat([grid[None].expand(batch,-1,-1),seeds.reshape(batch,-1,3)],1);g=grid.shape[1]
    gg=model.thermo(repeat_context(ctx,g),grid.reshape(-1,3),False)[3].detach().reshape(batch,g)
    mm=model.thermo(repeat_context(ctx,4),seeds.mean(2).reshape(-1,3),False)[2].detach().reshape(batch,4,3)
    mt=(gg[:,None]-torch.einsum('bki,bgi->bkg',mm,grid)).min(-1).values
    ma=model.thermo(ctx,seeds[:,0,0],False)[2].detach();mb=model.thermo(ctx,seeds[:,0,1],False)[2].detach()
    rms=(ma[:,:2]-mb[:,:2]).square().mean(-1).sqrt()
    tpd=torch.minimum((gg-torch.einsum('bi,bgi->bg',ma,grid)).min(-1).values,(gg-torch.einsum('bi,bgi->bg',mb,grid)).min(-1).values)
    good=(mt[:,0]<-1e-10)&(rms<=equilibrium_tol)&(tpd>=-tpd_tol)&((seeds[:,0,0]-seeds[:,0,1]).abs().sum(-1)>.005)
    results=[]
    for i in range(batch):
        if not bool(good[i]):
            results.append(original_solve_many(model,select_context(ctx,slice(i,i+1)),proposals[i:i+1],iterations,equilibrium_tol,tpd_tol,grid_resolution)[0]);continue
        pair=seeds[i,0,:,:2].cpu().numpy()
        if tuple(pair[0])>tuple(pair[1]):pair=pair[::-1].copy()
        results.append({'status':'converged','pairs':[{'endpoints':pair.tolist(),'equilibrium_rms':float(rms[i]),'minimum_grid_tpd':float(tpd[i])}],'attempts':4,'unstable_starts':int((mt[i]<-1e-10).sum()),'min_midpoint_tpd':float(mt[i].min()),'grid_resolution':grid_resolution})
    return results
