"""Differentiable, damped quasi-Newton endpoint refinement for LLE training.

The Jacobian is stopped (a first-order quasi-Newton derivative), while gradients
through residuals and every updated iterate are retained. This is not an exact
implicit derivative of a converged equilibrium. Final reporting uses solve_many.
"""
import torch
from .model import repeat_context, physics_loss
from .solver import _pack, _unpack, simplex_grid


def refine(model, ctx, proposals, steps=3):
    b,k=proposals.shape[:2]
    ec=repeat_context(ctx,k)
    z=_pack(proposals.reshape(-1,2,3)).requires_grad_(True)
    # Each proposal keeps its own largest phase-separation coordinate fixed.
    scan=(proposals[:,:,0]-proposals[:,:,1]).detach().abs().argmin(-1).flatten()
    pin=proposals[:,:,0].reshape(-1,3).gather(1,scan[:,None])
    for _ in range(steps):
        pairs=_unpack(z,3)
        ma=model.thermo(ec,pairs[:,0],True)[2]
        mb=model.thermo(ec,pairs[:,1],True)[2]
        r=torch.cat([ma-mb,pairs[:,0].gather(1,scan[:,None])-pin],-1)
        jac=torch.stack([torch.autograd.grad(r[:,i].sum(),z,retain_graph=True,create_graph=False)[0] for i in range(4)],1).detach()
        jt=jac.transpose(1,2)
        step=torch.linalg.solve(jt@jac+1e-3*torch.eye(4,device=z.device,dtype=z.dtype),jt@r.unsqueeze(-1)).squeeze(-1)
        z=(z-0.5*step.clamp(-2,2)).clamp(-18,18)
    return _unpack(z,3).reshape(b,k,2,3)


def surface_loss(model,ctx,target,valid,proposals, samples=4):
    """Cover multiple observed ties AND generated ties; test supporting planes.

    Labels enter training only. Deterministic simplex support replaces the old
    single random support point, to constrain curvature away from endpoints.
    """
    b=target.shape[0]
    choices=[]
    for row in valid:
        ids=torch.where(row)[0]
        choices.append(ids[torch.randperm(len(ids),device=ids.device)[:samples]] if len(ids)>=samples else ids[torch.arange(samples,device=ids.device)%len(ids)])
    ids=torch.stack(choices)
    observed=target[torch.arange(b,device=target.device)[:,None],ids]
    chosen=torch.linspace(0,proposals.shape[1]-1,samples,device=target.device).long()
    ties=torch.cat([observed,proposals[:,chosen]],1)
    k=ties.shape[1]; ec=repeat_context(ctx,k)
    pairs=ties.reshape(-1,2,3)
    local=physics_loss(model,ec,pairs[:,0],pairs[:,1]).reshape(b,k).mean(-1)
    grid=torch.tensor(simplex_grid(3,6),device=target.device,dtype=target.dtype)
    g=len(grid)
    gx=grid[None].expand(b,-1,-1).reshape(-1,3)
    gg=model.thermo(repeat_context(ctx,g),gx,True)[3].reshape(b,g)
    ma=model.thermo(ec,pairs[:,0],True)[2].reshape(b,k,3)
    mb=model.thermo(ec,pairs[:,1],True)[2].reshape(b,k,3)
    support_a=torch.relu(torch.einsum('bki,gi->bkg',ma,grid)-gg[:,None])
    support_b=torch.relu(torch.einsum('bki,gi->bkg',mb,grid)-gg[:,None])
    support=(support_a.square().mean((1,2))+support_b.square().mean((1,2)))*0.5
    return local+support
