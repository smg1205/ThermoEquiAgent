"""Batched numerical equivalent of the independent-condition LLE solver."""
import numpy as np
import torch
from .model import repeat_context,select_context
from .solver import simplex_grid,_pack,_unpack,_pad,solve

def solve_many(model,ctx,proposals,iterations=28,equilibrium_tol=1e-3,tpd_tol=1e-4,grid_resolution=18):
    n=int(ctx[-1][0].sum()); batch=len(ctx[0])
    if not bool((ctx[-1].sum(-1)==n).all()): raise ValueError('Homogeneous component count required')
    if n==2:
        return [solve(model,select_context(ctx,slice(i,i+1)),proposals[i,:1],iterations,equilibrium_tol,tpd_tol,grid_resolution) for i in range(batch)]
    device=ctx[0].device; dtype=ctx[0].dtype
    extra=[]
    for i,j in [(0,1),(0,2),(1,2)]:
        for fraction in (0.02,0.1,0.25,0.4):
            a=torch.full((3,),fraction,device=device,dtype=dtype); a[i]=1-2*fraction
            b=torch.full((3,),fraction,device=device,dtype=dtype); b[j]=1-2*fraction
            extra.append(torch.stack([a,b]))
    seeds=torch.cat([proposals.detach(),torch.stack(extra)[None].expand(batch,-1,-1,-1)],1).clamp_min(1e-8)
    seeds=seeds/seeds.sum(-1,keepdim=True); k=seeds.shape[1]
    base=torch.as_tensor(simplex_grid(n,grid_resolution),device=device,dtype=dtype)
    grids=torch.cat([base[None].expand(batch,-1,-1),seeds.reshape(batch,-1,n)],1)
    g=grids.shape[1]
    _,_,_,gg=model.thermo(repeat_context(ctx,g),grids.reshape(-1,n),False)
    gg=gg.detach().reshape(batch,g)
    ec=repeat_context(ctx,k)
    _,_,mm,_=model.thermo(ec,seeds.mean(2).reshape(-1,n),False)
    mid_tpd=(gg[:,None,:]-torch.einsum('bki,bgi->bkg',mm.detach().reshape(batch,k,n),grids)).min(-1).values
    unstable=mid_tpd < -tpd_tol
    results=[{'status':'no_instability_found','pairs':[],'attempts':k,'unstable_starts':int(v.sum()),'min_midpoint_tpd':float(t.min()),'grid_resolution':grid_resolution} for v,t in zip(unstable,mid_tpd)]
    if not unstable.any(): return results
    parent=torch.arange(batch,device=device).repeat_interleave(k)[unstable.flatten()]
    seeds=seeds.reshape(-1,2,n)[unstable.flatten()]
    ec=select_context(ec,unstable.flatten())
    # Match scalar selection: variance over unstable initial alpha endpoints.
    scan=torch.zeros(len(seeds),device=device,dtype=torch.long)
    for i in range(batch):
        active=parent==i
        if active.any():
            scan[active]=seeds[active,0].var(0).argmax() if int(active.sum())>1 else 0
    pin=seeds[:,0].gather(1,scan[:,None]).squeeze(-1).detach()
    z=_pack(seeds).detach()
    def residual(q,graph):
        pairs=_unpack(q,n)
        _,_,ma,_=model.thermo(ec,pairs[:,0],graph)
        _,_,mb,_=model.thermo(ec,pairs[:,1],graph)
        return torch.cat([ma-mb,(pairs[:,0].gather(1,scan[:,None]).squeeze(-1)-pin)[:,None]],-1)
    for _ in range(iterations):
        z=z.detach().requires_grad_(True); r=residual(z,True)
        jac=torch.stack([torch.autograd.grad(r[:,i].sum(),z,retain_graph=True)[0] for i in range(4)],1)
        jt=jac.transpose(1,2)
        lhs=jt@jac+1e-5*torch.eye(4,device=device,dtype=dtype)[None]
        delta=torch.linalg.solve(lhs,jt@r.detach().unsqueeze(-1)).squeeze(-1).clamp(-2,2)
        best=z.detach(); best_score=r.detach().square().sum(-1)
        for damping in (1.0,0.3,0.1):
            trial=(z.detach()-damping*delta).clamp(-18,18)
            value=residual(trial,False).detach().square().sum(-1)
            better=value<best_score; best=torch.where(better[:,None],trial,best); best_score=torch.minimum(best_score,value)
        z=best
        if float(best_score.max())<equilibrium_tol**2*0.1: break
    pairs=_unpack(z.detach(),n)
    _,_,ma,_=model.thermo(ec,pairs[:,0],False)
    _,_,mb,_=model.thermo(ec,pairs[:,1],False)
    rms=((ma-mb)**2).mean(-1).sqrt().detach()
    tpa=(gg[parent]-torch.einsum('ki,kgi->kg',ma.detach(),grids[parent])).min(-1).values
    tpb=(gg[parent]-torch.einsum('ki,kgi->kg',mb.detach(),grids[parent])).min(-1).values
    good=(rms<=equilibrium_tol)&(tpa>=-tpd_tol)&(tpb>=-tpd_tol)&((pairs[:,0]-pairs[:,1]).abs().sum(-1)>0.005)
    arrays=[v.detach().cpu().numpy() for v in (parent,pairs,rms,torch.minimum(tpa,tpb),good)]
    parent,pairs,rms,min_tpd,good=arrays
    for i in range(batch):
        active=parent==i
        if not active.any(): continue
        results[i]['status']='no_valid_coexistence'; results[i]['best_residual']=float(rms[active].min())
        for j in np.where(active&good)[0]:
            pair=pairs[j]
            if tuple(pair[0])>tuple(pair[1]): pair=pair[::-1].copy()
            if any(np.linalg.norm(pair-np.array(r['endpoints']))<1e-4 for r in results[i]['pairs']): continue
            results[i]['pairs'].append({'endpoints':pair.tolist(),'equilibrium_rms':float(rms[j]),'minimum_grid_tpd':float(min_tpd[j])})
        if results[i]['pairs']: results[i]['status']='converged'
    return results
