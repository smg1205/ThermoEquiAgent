import unittest,torch
from src.thermoformer.lle_tp.binodal_model import BinodalLLE
from src.thermoformer.lle_tp.model import repeat_context
from src.thermoformer.lle_tp.solver import simplex_grid
class BinodalTests(unittest.TestCase):
 def test_common_tangent_global_grid_and_pure_limit(self):
  torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
  m=BinodalLLE({'rdkit_2d':4,'unimol_v2':4,'functional_groups':2},16,1).double()
  ctx=m.context(torch.randn(1,3,10,dtype=torch.float64),torch.ones(1,1,dtype=torch.float64)*300,torch.ones(1,1,dtype=torch.float64)*101,torch.ones(1,3,dtype=torch.float64))
  pairs=m.propose(ctx)[0];self.assertTrue(bool((pairs>0).all()));self.assertTrue(torch.allclose(pairs.sum(-1),torch.ones(24,2,dtype=torch.float64)))
  ec=repeat_context(ctx,24);ma=m.thermo(ec,pairs[:,0])[2];mb=m.thermo(ec,pairs[:,1])[2]
  self.assertLess(float((ma-mb).abs().max()),1e-7)
  grid=torch.tensor(simplex_grid(3,18),dtype=torch.float64);gm=m.thermo(repeat_context(ctx,len(grid)),grid)[3].flatten()
  self.assertGreaterEqual(float((gm[None]-ma@grid.T).min()),-1e-8)
  pure=m.thermo(repeat_context(ctx,3),torch.eye(3,dtype=torch.float64))[0]
  self.assertLess(float(pure.abs().max()),1e-10)
  pairs.square().sum().backward()
  self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
if __name__=='__main__':unittest.main()
