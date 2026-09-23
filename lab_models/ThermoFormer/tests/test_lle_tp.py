"""Tests for label-free TP inference, split isolation and thermodynamics."""
import unittest
import numpy as np
import torch
from src.thermoformer.lle_tp.data import Condition,partition
from src.thermoformer.lle_tp.metrics import score
from src.thermoformer.lle_tp.model import TPLLE,endpoint_loss
from src.thermoformer.lle_tp.solver import solve

class RegularSolution:
    def __init__(self,a): self.a=a
    def thermo(self,ctx,x,create_graph=True):
        with torch.enable_grad():
            x=x if x.requires_grad else x.detach().requires_grad_(True)
            ge=self.a*x[:,0:1]*x[:,1:2]
            grad=torch.autograd.grad(ge.sum(),x,create_graph=create_graph,retain_graph=True)[0]
            lng=(ge+grad-(x*grad).sum(-1,keepdim=True))*ctx[-1]
            mu=(x.clamp_min(1e-9).log()+lng)*ctx[-1]
            gm=ge+(x*x.clamp_min(1e-9).log()*ctx[-1]).sum(-1,keepdim=True)
        return ge,lng,mu,gm

class Tests(unittest.TestCase):
    def test_splits_keep_conditions_together(self):
        c=[Condition(((str(i),str(i+1)),280.+j,101.3),()) for i in range(8) for j in range(8)]
        for mode in ('random','system'):
            a=partition(c,mode,3); b=partition(c,mode,3)
            self.assertEqual([x.key for x in a['test']],[x.key for x in b['test']])
            kk=[{x.key if mode=='random' else x.key[0] for x in a[name]} for name in a]
            self.assertFalse(kk[0]&kk[1] or kk[0]&kk[2] or kk[1]&kk[2])
    def test_phase_swap_and_failure_metrics(self):
        p=[[[.05,.95],[.95,.05]]]
        r={'observed':p,'predicted':[p[0][::-1]]}
        self.assertAlmostEqual(score([r])['matched_endpoint_r2_successful_only'],1)
        r['predicted']=[]
        self.assertEqual(score([r])['failure_penalized_set_distance'],1)
        self.assertFalse(score([r])['r2_target_met'])
    def test_regular_solution_and_ideal(self):
        dtype=torch.float64
        ctx=(torch.zeros(1,3,1,dtype=dtype),torch.zeros(1,1,1,dtype=dtype),torch.ones(1,1,dtype=dtype)*300,torch.ones(1,1,dtype=dtype)*101,torch.tensor([[1.,1.,0.]],dtype=dtype))
        seeds=torch.tensor([[[.03,.97,0],[.97,.03,0]]],dtype=dtype)
        ideal=solve(RegularSolution(0),ctx,seeds)
        self.assertEqual(ideal['status'],'no_instability_found')
        result=solve(RegularSolution(3),ctx,seeds,iterations=40)
        self.assertTrue(result['pairs'],result)
        x=result['pairs'][0]['endpoints'][0][0]
        self.assertAlmostEqual(x,0.07072,places=3)
    def test_batched_ternary_matches_scalar(self):
        from src.thermoformer.lle_tp.solver_batch import solve_many
        from src.thermoformer.lle_tp.model import repeat_context
        dtype=torch.float64
        ctx=(torch.zeros(1,3,1,dtype=dtype),torch.zeros(1,1,1,dtype=dtype),torch.ones(1,1,dtype=dtype)*300,torch.ones(1,1,dtype=dtype)*101,torch.ones(1,3,dtype=dtype))
        seeds=torch.tensor([[[.05,.75,.2],[.75,.05,.2]],[[.04,.86,.1],[.86,.04,.1]]],dtype=dtype)
        scalar=solve(RegularSolution(4),ctx,seeds,iterations=12,grid_resolution=8)
        self.assertTrue(scalar['pairs'],scalar)
        batched=solve_many(RegularSolution(4),repeat_context(ctx,2),seeds[None].repeat(2,1,1,1),iterations=12,grid_resolution=8)
        for b in batched:
            self.assertEqual(b['status'],scalar['status'])
            self.assertEqual(len(b['pairs']),len(scalar['pairs']))
            np.testing.assert_allclose([p['endpoints'] for p in b['pairs']],[p['endpoints'] for p in scalar['pairs']],atol=1e-7)

    def test_model_simplex_ge_pure_and_gradients(self):
        torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        model=TPLLE({'rdkit_2d':4,'unimol_v2':4,'functional_groups':2},hidden=16,layers=1)
        state=(torch.randn(2,3,10),torch.ones(2,1)*300,torch.ones(2,1)*101,torch.tensor([[1.,1.,0.],[1.,1.,1.]]))
        ctx=model.context(*state); proposal=model.propose(ctx,4)
        self.assertTrue(torch.allclose(proposal.sum(-1),torch.ones(2,4,2),atol=1e-6))
        pure=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
        ge,_,_,_=model.thermo(ctx,pure)
        self.assertTrue(torch.allclose(ge,torch.zeros_like(ge)))
        x=torch.tensor([[.3,.7,0],[.2,.3,.5]])
        _,lng,_,_=model.thermo(ctx,x)
        lng.square().sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.backbone.pair_potential.parameters()))

if __name__=='__main__': unittest.main()
