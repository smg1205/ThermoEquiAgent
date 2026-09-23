"""Label-free multi-start TPD screening and common-tangent endpoint correction."""
import numpy as np
import torch
from .model import repeat_context,select_context

def simplex_grid(n,resolution=18):
    if n==2:
        u=np.unique(np.r_[np.linspace(1e-5,1-1e-5,resolution*4),np.geomspace(1e-8,0.01,12),1-np.geomspace(1e-8,0.01,12)])
        return np.stack([u,1-u],-1)
    g=np.array([[i,j,resolution-i-j] for i in range(resolution+1) for j in range(resolution+1-i)],float)/resolution
    return (g+1e-8)/(1+3e-8)

def _pack(x): return (x[...,:-1].clamp_min(1e-8).log()-x[...,-1:].clamp_min(1e-8).log()).flatten(1)
def _unpack(z,n):
    z=z.reshape(-1,2,n-1)
    return torch.cat([z,torch.zeros_like(z[...,:1])],-1).softmax(-1)
def _pad(x): return torch.nn.functional.pad(x,(0,3-x.shape[-1]))

def solve(model,ctx,seeds,iterations=28,equilibrium_tol=1e-3,tpd_tol=1e-4,grid_resolution=18):
    """One condition. Seeds and scan coordinates must be independent of labels.

    TPD checks use a deterministic dense simplex grid: this is a numerical
    stability screen, not a proof of global stability or three-phase exclusion.
    """
    n=int(ctx[-1].sum().item()); device=ctx[0].device; dtype=ctx[0].dtype
    grid=torch.as_tensor(simplex_grid(n,grid_resolution),device=device,dtype=dtype)
    if n==2:
        extra=torch.tensor([[[.01,.99],[.99,.01]],[[.1,.9],[.9,.1]],[[.3,.7],[.7,.3]]],device=device,dtype=dtype)
    else:
        extra=[]
        for i,j in [(0,1),(0,2),(1,2)]:
            for fraction in (0.02,0.1,0.25,0.4):
                a=torch.full((3,),fraction,device=device,dtype=dtype); a[i]=1-2*fraction
                b=torch.full((3,),fraction,device=device,dtype=dtype); b[j]=1-2*fraction
                extra.append(torch.stack([a,b]))
        extra=torch.stack(extra)
    seeds=torch.cat([seeds.detach()[...,:n],extra]).clamp_min(1e-8)
    seeds=seeds/seeds.sum(-1,keepdim=True)
    grid=torch.cat([grid,seeds.reshape(-1,n)],0)
    k=len(seeds); ec=repeat_context(ctx,k)
    _,_,_,gg=model.thermo(repeat_context(ctx,len(grid)),_pad(grid),False)
    gg=gg.detach().squeeze(-1)
    mid=seeds.mean(1)
    _,_,mm,_=model.thermo(ec,_pad(mid),False)
    mid_tpd=(gg[None,:]-mm.detach()[:,:n]@grid.T).min(-1).values
    unstable=mid_tpd < -(1e-10 if n==2 else tpd_tol)
    if not unstable.any():
        return {'status':'no_instability_found','pairs':[],'attempts':k,'unstable_starts':0,'min_midpoint_tpd':float(mid_tpd.min()),'grid_resolution':grid_resolution}
    if n==2:
        # A constrained energy parameterization can supply an already converged
        # root. Accept only after the same activity and supporting-plane checks.
        _,_,ma,_=model.thermo(ec,_pad(seeds[:,0]),False)
        _,_,mb,_=model.thermo(ec,_pad(seeds[:,1]),False)
        rms=((ma[:,:n]-mb[:,:n])**2).mean(-1).sqrt().detach()
        tpa=(gg[None,:]-ma.detach()[:,:n]@grid.T).min(-1).values
        tpb=(gg[None,:]-mb.detach()[:,:n]@grid.T).min(-1).values
        valid=unstable&(rms<=equilibrium_tol)&(tpa>=-tpd_tol)&(tpb>=-tpd_tol)&((seeds[:,0]-seeds[:,1]).abs().sum(-1)>0.005)
        if valid.any():
            i=rms.masked_fill(~valid,float('inf')).argmin().item()
            pair=seeds[i].cpu().numpy()
            if tuple(pair[0])>tuple(pair[1]): pair=pair[::-1].copy()
            return {'status':'converged','pairs':[{'endpoints':pair.tolist(),'equilibrium_rms':float(rms[i]),'minimum_grid_tpd':float(torch.minimum(tpa[i],tpb[i]))}],'attempts':k,'unstable_starts':int(unstable.sum()),'min_midpoint_tpd':float(mid_tpd.min()),'grid_resolution':grid_resolution}
    seeds=seeds[unstable]; ec=select_context(ec,unstable)
    k=len(seeds); z=_pack(seeds).detach()
    # One independent continuation coordinate is required for a ternary curve.
    scan_index=(seeds[:,0].var(0)).argmax().item() if n==3 else 0
    pin=seeds[:,0,scan_index].detach()
    def residual(q,context,pins,graph):
        pair=_unpack(q,n)
        _,_,ma,_=model.thermo(context,_pad(pair[:,0]),graph)
        _,_,mb,_=model.thermo(context,_pad(pair[:,1]),graph)
        r=(ma-mb)[:,:n]
        if n==3: r=torch.cat([r,(pair[:,0,scan_index]-pins)[:,None]],-1)
        return r
    for _ in range(iterations):
        z=z.detach().requires_grad_(True)
        r=residual(z,ec,pin,True)
        jac=torch.stack([torch.autograd.grad(r[:,i].sum(),z,retain_graph=True)[0] for i in range(r.shape[1])],1)
        jt=jac.transpose(1,2)
        lhs=jt@jac+1e-5*torch.eye(z.shape[1],device=device,dtype=dtype)[None]
        delta=torch.linalg.solve(lhs,jt@r.detach().unsqueeze(-1)).squeeze(-1).clamp(-2,2)
        current=r.detach().square().sum(-1); best=z.detach(); best_score=current
        for damping in (1.0,0.3,0.1):
            trial=(z.detach()-damping*delta).clamp(-18,18)
            score=residual(trial,ec,pin,False).detach().square().sum(-1)
            better=score<best_score
            best=torch.where(better[:,None],trial,best); best_score=torch.minimum(best_score,score)
        z=best
        if float(best_score.max())<equilibrium_tol**2*0.1: break
    pairs=_unpack(z.detach(),n)
    _,_,ma,_=model.thermo(ec,_pad(pairs[:,0]),False)
    _,_,mb,_=model.thermo(ec,_pad(pairs[:,1]),False)
    rms=((ma[:,:n]-mb[:,:n])**2).mean(-1).sqrt().detach()
    tpa=(gg[None,:]-ma.detach()[:,:n]@grid.T).min(-1).values
    tpb=(gg[None,:]-mb.detach()[:,:n]@grid.T).min(-1).values
    separation=(pairs[:,0]-pairs[:,1]).abs().sum(-1)
    good=(rms<=equilibrium_tol)&(tpa>=-tpd_tol)&(tpb>=-tpd_tol)&(separation>0.005)
    records=[]
    for i in torch.where(good)[0].tolist():
        pair=pairs[i].detach().cpu().numpy()
        if tuple(pair[0])>tuple(pair[1]): pair=pair[::-1].copy()
        if any(np.linalg.norm(pair-np.array(r['endpoints']))<1e-4 for r in records): continue
        records.append({'endpoints':pair.tolist(),'equilibrium_rms':float(rms[i]),'minimum_grid_tpd':float(torch.minimum(tpa[i],tpb[i]))})
    if n==2 and records:
        records=[min(records,key=lambda r:r['equilibrium_rms'])]
    return {'status':'converged' if records else 'no_valid_coexistence','pairs':records,'attempts':len(unstable),'unstable_starts':k,'min_midpoint_tpd':float(mid_tpd.min()),'best_residual':float(rms.min()),'grid_resolution':grid_resolution}
