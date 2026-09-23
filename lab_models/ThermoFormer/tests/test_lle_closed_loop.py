import unittest
import torch
from tests.test_lle_tp import RegularSolution
from src.thermoformer.lle_tp.closed_loop import refine

class ClosedLoopTests(unittest.TestCase):
    def test_refinement_reduces_activity_error_and_backpropagates(self):
        torch.set_num_threads(2)
        dtype=torch.float64
        ctx=(torch.zeros(1,3,1,dtype=dtype),torch.zeros(1,1,1,dtype=dtype),torch.ones(1,1,dtype=dtype)*300,torch.ones(1,1,dtype=dtype)*101,torch.ones(1,3,dtype=dtype))
        strength=torch.tensor(4.,dtype=dtype,requires_grad=True)
        model=RegularSolution(strength)
        initial=torch.tensor([[[[.05,.75,.2],[.75,.05,.2]]]],dtype=dtype,requires_grad=True)
        refined=refine(model,ctx,initial,steps=4)
        def error(pair):
            ma=model.thermo(ctx,pair[:,0,0])[2]
            mb=model.thermo(ctx,pair[:,0,1])[2]
            return (ma-mb).square().sum()
        self.assertLess(error(refined).item(),error(initial).item())
        self.assertTrue(torch.allclose(refined.sum(-1),torch.ones_like(refined.sum(-1))))
        refined[...,0,0].sum().backward()
        self.assertTrue(torch.isfinite(strength.grad).all())
        self.assertGreater(strength.grad.abs().item(),1e-6)
        self.assertTrue(torch.isfinite(initial.grad).all())

    def test_implicit_root_gradient_matches_finite_difference(self):
        from scipy.optimize import brentq
        import math
        from src.thermoformer.lle_tp.implicit_roots import attach_roots
        dtype=torch.float64
        ctx=(torch.zeros(1,3,1,dtype=dtype),torch.zeros(1,1,1,dtype=dtype),torch.ones(1,1,dtype=dtype)*300,torch.ones(1,1,dtype=dtype)*101,torch.ones(1,3,dtype=dtype))
        def root(chi):return brentq(lambda a:math.log(a/(.8-a))+chi*(.8-2*a),1e-6,.399)
        a=root(4.)
        roots=torch.tensor([[[[a,.8-a,.2],[.8-a,a,.2]]]],dtype=dtype)
        chi=torch.tensor(4.,dtype=dtype,requires_grad=True)
        attached=attach_roots(RegularSolution(chi),ctx,roots)
        self.assertTrue(torch.allclose(attached,roots,atol=1e-12,rtol=0))
        attached[0,0,0,0].backward()
        finite=(root(4.0001)-root(3.9999))/.0002
        self.assertAlmostEqual(chi.grad.item(),finite,places=5)

if __name__=='__main__':unittest.main()
