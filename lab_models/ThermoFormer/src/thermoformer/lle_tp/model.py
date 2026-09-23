"""Reuse the VLE molecular views/Transformer and symmetric gE pair potential."""
import torch
from torch import nn
from ..models.thermoformer import ThermoFormer, ThermoFormerConfig

class TPLLE(nn.Module):
    def __init__(self,dimensions,hidden=96,layers=2):
        super().__init__()
        self.spec={'dimensions':dimensions,'hidden':hidden,'layers':layers}
        self.backbone=ThermoFormer(ThermoFormerConfig(feature_dim=sum(dimensions.values()),hidden_dim=hidden,layers=layers,heads=4,fusion_mode='naive',rdkit_feature_dim=dimensions['rdkit_2d'],unimol_feature_dim=dimensions['unimol_v2'],functional_group_feature_dim=dimensions['functional_groups'],dropout=0.0))
        self.proposal=nn.Sequential(nn.Linear(2*hidden+4,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,2))
        # The pure-vapor branch is unused by this liquid-only task.
        for p in self.backbone.vapor_pressure.parameters(): p.requires_grad_(False)

    def context(self,molecules,t,p,mask):
        tokens,views,_=self.backbone._encode_molecules(molecules)
        h,m,_,_=self.backbone._structural_context(tokens,views,t,p,mask/mask.sum(-1,keepdim=True),mask)
        return h,m,t,p,mask

    def propose(self,ctx,points=24):
        h,m,t,p,mask=ctx
        b,n,d=h.shape
        s=torch.linspace(0,1,points,device=h.device,dtype=h.dtype)[None,:,None,None].expand(b,-1,n,-1)
        s=torch.where((mask.sum(-1)==2)[:,None,None,None],torch.full_like(s,0.5),s)
        temp=((t-350)/150)[:,None,:,None].expand(-1,points,n,-1)
        press=torch.log(p.clamp_min(1e-8)/101.325)[:,None,:,None].expand(-1,points,n,-1)
        z=torch.cat([h[:,None].expand(-1,points,-1,-1),m[:,None].expand(-1,points,n,-1),temp,press,s,s*s],-1)
        logits=self.proposal(z).permute(0,1,3,2)
        result=logits.masked_fill(~mask[:,None,None].bool(),-1e9).softmax(-1)
        binary=mask.sum(-1)==2
        if binary.any():
            a=result[:,:, :,0].min(-1).values.clamp(1e-5,1-0.01001)
            b=torch.maximum(result[:,:,:,0].max(-1).values,a+0.01).clamp(max=1-1e-5)
            first=torch.stack([a,1-a,torch.zeros_like(a)],-1)
            second=torch.stack([b,1-b,torch.zeros_like(b)],-1)
            constrained=torch.stack([first,second],-2)
            result=torch.where(binary[:,None,None,None],constrained,result)
        return result

    def binary_coefficients(self,ctx):
        # Parameterize a cubic Redlich-Kister surface by its common-tangent
        # coordinates. These are neural latent coordinates, never observed inputs.
        initial=self.propose(ctx,points=1)[:,0,:,0].double()
        a=initial.min(-1).values.clamp(1e-5,1-0.01001)
        b=torch.maximum(initial.max(-1).values,a+0.01).clamp(max=1-1e-5)
        def basis(q):
            g0=q*(1-q); d0=1-2*q
            g1=g0*(2*q-1); d1=d0*(2*q-1)+2*g0
            return torch.stack([g0+(1-q)*d0,g1+(1-q)*d1],-1),torch.stack([g0-q*d0,g1-q*d1],-1)
        a1,a2=basis(a); b1,b2=basis(b)
        matrix=torch.stack([a1-b1,a2-b2],1)
        rhs=torch.stack([torch.log(b/a),torch.log((1-b)/(1-a))],-1)
        coeff=torch.linalg.solve(matrix,rhs.unsqueeze(-1)).squeeze(-1)
        return coeff.to(ctx[0].dtype)

    def thermo(self,ctx,x,create_graph=True):
        h,m,t,p,mask=ctx
        with torch.enable_grad():
            x=x if x.requires_grad else x.detach().requires_grad_(True)
            tokens,context=self.backbone._nonideality_tokens(h,m,t,p,x,mask)
            ge,_=self.backbone._excess_gibbs(tokens,context,x,mask)
            if bool((mask.sum(-1)==2).any()):
                coeff=self.binary_coefficients(ctx)
                q=x[:,0:1]
                binary_ge=q*(1-q)*(coeff[:,0:1]+coeff[:,1:2]*(2*q-1))
                ge=torch.where((mask.sum(-1)==2)[:,None],binary_ge,ge)
            grad=torch.autograd.grad(ge.sum(),x,create_graph=create_graph,retain_graph=True)[0]
            lng=(ge+grad-(x*grad*mask).sum(-1,keepdim=True))*mask
            mu=(x.clamp_min(1e-9).log()+lng)*mask
            gm=ge+(x*x.clamp_min(1e-9).log()*mask).sum(-1,keepdim=True)
        return ge,lng,mu,gm

def repeat_context(ctx,count):
    return tuple(v.repeat_interleave(count,dim=0) for v in ctx)

def select_context(ctx,index):
    return tuple(v[index] for v in ctx)

def pair_distance(pred,true):
    """Squared full-component distance, invariant to exchanging liquid phases."""
    direct=(pred[:,:,None]-true[:,None]).square().mean((-1,-2))
    swapped=(pred[:,:,None]-true[:,None].flip(-2)).square().mean((-1,-2))
    return torch.minimum(direct,swapped)

def endpoint_loss(pred,target,valid):
    d=pair_distance(pred,target)
    forward=d.masked_fill(~valid[:,None],float('inf')).min(-1).values.mean(-1)
    backward=(d.min(1).values*valid).sum(-1)/valid.sum(-1)
    return (forward+backward)*0.5

def physics_loss(model,ctx,alpha,beta):
    _,_,ma,ga=model.thermo(ctx,alpha)
    _,_,mb,gb=model.thermo(ctx,beta)
    mid=(alpha+beta)/2
    _,_,_,gm=model.thermo(ctx,mid)
    mask=ctx[-1]; separation=((alpha-beta)**2*mask).sum(-1,keepdim=True)
    # Equal chemical potentials alone admits a flat, unphysical gmix surface.
    barrier=torch.relu(0.02*separation-(gm-(ga+gb)/2)).square().squeeze(-1)
    # Reported zero compositions are censored, not exact log(0) activities.
    # Use a fixed numerical upper bound, declared in the protocol, for those entries.
    bound=1e-5
    za=alpha<=0; zb=beta<=0
    upper_a=ma+(torch.log(torch.tensor(bound,device=ma.device))-torch.log(torch.tensor(1e-9,device=ma.device)))*za
    upper_b=mb+(torch.log(torch.tensor(bound,device=mb.device))-torch.log(torch.tensor(1e-9,device=mb.device)))*zb
    residual=ma-mb
    residual=torch.where(za,torch.relu(mb-upper_a),residual)
    residual=torch.where(zb,torch.relu(ma-upper_b),residual)
    residual=torch.where(za&zb,torch.zeros_like(residual),residual)
    robust=torch.nn.functional.smooth_l1_loss(residual,torch.zeros_like(residual),reduction='none',beta=0.2)
    eq=(robust*mask).sum(-1)/mask.sum(-1)
    z=torch.rand_like(alpha).clamp_min(1e-5)*mask; z=z/z.sum(-1,keepdim=True)
    _,_,_,gz=model.thermo(ctx,z)
    support=torch.relu((z*ma*mask).sum(-1,keepdim=True)-gz).square().squeeze(-1)
    return eq+10*barrier+support
