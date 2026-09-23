"""Implicit local gradients at accepted final-solver roots (no label seeding)."""
import copy
import torch
from .model import repeat_context, select_context, endpoint_loss
from .solver import _pack, _unpack
from .solver_batch import solve_many


def attach_roots(model,ctx,roots):
    """Keep roots exact in forward; differentiate common-tangent equations.

    One alpha coordinate fixes the ternary branch gauge. Discrete root discovery
    and acceptance are not differentiable. A small ridge stabilizes singular
    near-critical Jacobians; hence the derivative there is regularized.
    """
    b,k=roots.shape[:2];ec=repeat_context(ctx,k)
    pairs=roots.reshape(-1,2,3).detach()
    scan=(pairs[:,0]-pairs[:,1]).abs().argmin(-1)
    pin=pairs[:,0].gather(1,scan[:,None])
    z=_pack(pairs).detach().requires_grad_(True)
    x=_unpack(z,3)
    ma=model.thermo(ec,x[:,0],True)[2];mb=model.thermo(ec,x[:,1],True)[2]
    r=torch.cat([ma-mb,x[:,0].gather(1,scan[:,None])-pin],-1)
    jac=torch.stack([torch.autograd.grad(r[:,i].sum(),z,retain_graph=True)[0] for i in range(4)],1).detach()
    jt=jac.transpose(1,2)
    delta=torch.linalg.solve(jt@jac+1e-8*torch.eye(4,device=z.device,dtype=z.dtype),jt@(r-r.detach()).unsqueeze(-1)).squeeze(-1)
    return _unpack(z-delta,3).reshape(b,k,2,3)


def final_root_loss(model,ctx,proposals,target,valid,iterations=28,grid_resolution=18):
    """Real final accepted sets; failed conditions use the other training losses."""
    numerical=copy.deepcopy(model).double().eval().requires_grad_(False)
    dc=tuple(v.detach().double() for v in ctx)
    solutions=solve_many(numerical,dc,proposals.detach().double(),iterations=iterations,grid_resolution=grid_resolution)
    del numerical
    losses=[]
    for i,result in enumerate(solutions):
        if not result['pairs']:continue
        roots=torch.tensor([p['endpoints'] for p in result['pairs']],device=proposals.device,dtype=proposals.dtype)[None]
        attached=attach_roots(model,select_context(ctx,slice(i,i+1)),roots)
        losses.append(endpoint_loss(attached,target[i:i+1],valid[i:i+1]).mean())
    loss=torch.stack(losses).sum()/len(solutions) if losses else proposals.sum()*0
    return loss,len(losses)
