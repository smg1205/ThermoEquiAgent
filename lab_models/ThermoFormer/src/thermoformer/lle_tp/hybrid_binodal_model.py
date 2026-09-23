"""Constructive ternary energy with a continuous parallel-tie binodal prior.

Let q=n.x, u=d.x, d.n=0, both orthogonal to (1,1,1). Two learned
curves u=a(q),b(q) define the binodal. A nonnegative well phi vanishes
with zero gradient on both curves. gmix=C(q)+phi*(2+sum x log x), up
to an affine pure-state reference, with convex C=q^2/2. Thus every
same-q endpoint pair has an analytically supporting common tangent.
This is a constrained LLE-only decoder, not a general existence classifier.
"""
import torch
from .model import TPLLE

class HybridBinodalLLE(TPLLE):
    def __init__(self,dimensions,hidden=96,layers=2):
        super().__init__(dimensions,hidden,layers)
        self.spec['decoder']='hybrid_binodal'
        self.supports_cached_binary=True

    @staticmethod
    def well(qgrid,a,b,q,u):
        query=torch.maximum(qgrid[:,:1],torch.minimum(q,qgrid[:,-1:]))
        index=torch.searchsorted(qgrid.contiguous(),query.contiguous(),right=True).clamp(1,qgrid.shape[1]-1)
        left=index-1
        ql=qgrid.gather(1,left);qr=qgrid.gather(1,index)
        fraction=(query-ql)/(qr-ql).clamp_min(1e-12)
        def interpolate(values):
            slopes=torch.cat([torch.zeros_like(values[:,:1]),(values[:,2:]-values[:,:-2])/(qgrid[:,2:]-qgrid[:,:-2]),torch.zeros_like(values[:,:1])],1)
            f=fraction
            return (2*f**3-3*f**2+1)*values.gather(1,left)+(f**3-2*f**2+f)*(qr-ql)*slopes.gather(1,left)+(-2*f**3+3*f**2)*values.gather(1,index)+(f**3-f**2)*(qr-ql)*slopes.gather(1,index)
        center=interpolate((a+b)/2)
        gap=interpolate((b-a).clamp_min(1e-12).log()).exp()
        aa=center-gap/2;bb=center+gap/2
        da=(u-aa).square();db=(u-bb).square()
        return da*db/(da+db+1e-20)+torch.relu(qgrid[:,:1]-q).square()+torch.relu(q-qgrid[:,-1:]).square()

    def context(self,molecules,t,p,mask):
        if not bool(((mask.sum(-1)==2)|(mask.sum(-1)==3)).all()):raise ValueError('Two or three components required')
        h,m,t,p,mask=super().context(molecules,t,p,mask)
        if bool((mask.sum(-1)==2).all()):
            coeff=super().binary_coefficients((h,m,t,p,mask))
            z=h.new_zeros(len(h),24);v=h.new_zeros(len(h),3)
            return h,m,t,p,z,z,z,v,v,v,coeff,mask
        raw=super().propose((h,m,t,p,mask),24)
        delta=raw[:,:,1]-raw[:,:,0]
        reference=delta[torch.arange(len(delta),device=delta.device),delta.square().sum(-1).argmax(-1)]
        aligned=delta*torch.where((delta*reference[:,None]).sum(-1,keepdim=True)>=0,1.,-1.)
        d=aligned.mean(1);d=d-d.mean(-1,keepdim=True)
        fallback=torch.tensor([1.,-1.,0.],device=d.device,dtype=d.dtype)[None]
        d=torch.where(d.norm(dim=-1,keepdim=True)>1e-8,d,fallback)
        d=d/d.norm(dim=-1,keepdim=True)
        n=torch.cross(d,torch.ones_like(d)/3**.5,dim=-1)
        n=n/n.norm(dim=-1,keepdim=True)
        qraw=(raw.mean(2)*n[:,None]).sum(-1)
        sort=qraw.argsort(-1)
        qlow=n.min(-1).values+1e-3;qhigh=n.max(-1).values-1e-3
        span=torch.minimum((qraw.max(-1).values-qraw.min(-1).values).clamp_min(1e-3),qhigh-qlow)
        middle=torch.maximum(qlow+span/2,torch.minimum(qraw.mean(-1),qhigh-span/2))
        s=torch.linspace(-.5,.5,24,device=d.device,dtype=d.dtype)
        q=middle[:,None]+span[:,None]*s
        rawu=(raw*d[:,None,None]).sum(-1)
        aa=rawu.min(-1).values.gather(1,sort);bb=rawu.max(-1).values.gather(1,sort)
        bound=(1e-7-1/3-q[:,:,None]*n[:,None])/torch.where(d[:,None].abs()>1e-10,d[:,None],torch.ones_like(d[:,None]))
        lower=bound.masked_fill(d[:,None]<=1e-10,-float('inf')).max(-1).values
        upper=bound.masked_fill(d[:,None]>=-1e-10,float('inf')).min(-1).values
        width=(upper-lower).clamp_min(1e-7);margin=.001*width
        gap=torch.minimum(torch.full_like(width,.005),width*.1)
        aa=torch.maximum(lower+margin,torch.minimum(aa,upper-margin-gap))
        bb=torch.maximum(aa+gap,torch.minimum(bb,upper-margin))
        pure=self.well(q,aa,bb,n,d).clamp_min(1e-20)
        coeff=super().binary_coefficients((h,m,t,p,mask)) if bool((mask.sum(-1)==2).any()) else torch.zeros(len(h),2,device=h.device,dtype=h.dtype)
        return h,m,t,p,q,aa,bb,d,n,pure,coeff,mask

    def propose(self,ctx,points=24):
        if len(ctx)==5:return super().propose(ctx,points)
        q,a,b,d,n=ctx[4:9]
        ids=torch.linspace(0,q.shape[1]-1,points,device=q.device).round().long()
        q=q[:,ids];a=a[:,ids];b=b[:,ids]
        alpha=1/3+q[:,:,None]*n[:,None]+a[:,:,None]*d[:,None]
        beta=1/3+q[:,:,None]*n[:,None]+b[:,:,None]*d[:,None]
        result=torch.stack([alpha,beta],2)
        binary=ctx[-1].sum(-1)==2
        if binary.any():result=torch.where(binary[:,None,None,None],super().propose((*ctx[:4],ctx[-1]),points),result)
        return result

    def thermo(self,ctx,x,create_graph=True):
        with torch.enable_grad():
            x=x if x.requires_grad else x.detach().requires_grad_(True)
            mask=ctx[-1];binary=mask.sum(-1)==2
            ideal=(x*x.clamp_min(1e-12).log()*mask).sum(-1,keepdim=True)
            coeff=ctx[-2];u0=x[:,0:1]
            binary_ge=u0*(1-u0)*(coeff[:,0:1]+coeff[:,1:2]*(2*u0-1))
            if bool(binary.all()):ge=binary_ge
            else:
                qgrid,a,b,d,n,_=ctx[4:10]
                q=(x*n).sum(-1,keepdim=True);u=(x*d).sum(-1,keepdim=True)
                well=self.well(qgrid,a,b,q,u)
                # This normalization equals one on the simplex boundary away
                # from a boundary binodal, and vanishes at interior endpoints.
                phi=well/(well+x.prod(-1,keepdim=True)).clamp_min(1e-20)
                affine=(x*(.5*n.square()+2)).sum(-1,keepdim=True)
                geo=.5*q.square()+phi*(2+ideal)-affine-ideal
                ge=torch.where(binary[:,None],binary_ge,geo)
            grad=torch.autograd.grad(ge.sum(),x,create_graph=create_graph,retain_graph=True)[0]
            lng=(ge+grad-(x*grad*mask).sum(-1,keepdim=True))*mask
            mu=(x.clamp_min(1e-12).log()+lng)*mask
            gm=ge+ideal
        return ge,lng,mu,gm
