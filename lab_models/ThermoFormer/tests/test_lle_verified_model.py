import unittest,numpy as np,torch
from src.thermoformer.lle_tp.hybrid_binodal_model import HybridBinodalLLE
from src.thermoformer.lle_tp.model import TPLLE,repeat_context
from src.thermoformer.lle_tp.solver import simplex_grid
from src.thermoformer.lle_tp.solver_cached import solve_many as fast
from src.thermoformer.lle_tp.solver_batch import solve_many as slow
class VerifiedModelTests(unittest.TestCase):
 def test_cached_binary_matches_legacy_energy_and_solver(self):
  torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
  dims={'rdkit_2d':4,'unimol_v2':4,'functional_groups':2}
  old=TPLLE(dims,16,1).double();new=HybridBinodalLLE(dims,16,1).double();new.load_state_dict(old.state_dict())
  state=(torch.randn(2,3,10,dtype=torch.float64),torch.ones(2,1,dtype=torch.float64)*300,torch.ones(2,1,dtype=torch.float64)*101,torch.tensor([[1.,1.,0.]]*2,dtype=torch.float64))
  oc=old.context(*state);nc=new.context(*state);x=torch.tensor([[.2,.8,0],[.6,.4,0]],dtype=torch.float64)
  for a,b in zip(old.thermo(oc,x),new.thermo(nc,x)):self.assertTrue(torch.allclose(a,b,atol=1e-9,rtol=1e-9))
  proposal=new.propose(nc,1).detach();a=fast(new,nc,proposal);b=slow(new,nc,proposal)
  for aa,bb in zip(a,b):
   self.assertEqual(aa['status'],bb['status'])
   np.testing.assert_allclose([v['endpoints'] for v in aa['pairs']],[v['endpoints'] for v in bb['pairs']],atol=1e-9)
 def test_ternary_support_and_pure_limits(self):
  torch.backends.mha.set_fastpath_enabled(False)
  m=HybridBinodalLLE({'rdkit_2d':4,'unimol_v2':4,'functional_groups':2},16,1).double()
  c=m.context(torch.randn(1,3,10,dtype=torch.float64),torch.ones(1,1,dtype=torch.float64)*300,torch.ones(1,1,dtype=torch.float64)*101,torch.ones(1,3,dtype=torch.float64))
  x=m.propose(c)[0];ec=repeat_context(c,24);ma=m.thermo(ec,x[:,0])[2];mb=m.thermo(ec,x[:,1])[2]
  self.assertLess(float((ma-mb).abs().max()),1e-7)
  grid=torch.tensor(simplex_grid(3,18),dtype=torch.float64);gm=m.thermo(repeat_context(c,len(grid)),grid)[3].flatten()
  self.assertGreaterEqual(float((gm[None]-ma@grid.T).min()),-1e-8)
  self.assertLess(float(m.thermo(repeat_context(c,3),torch.eye(3,dtype=torch.float64))[0].abs().max()),1e-10)
 def test_certified_roots_are_also_roots_of_original_solver(self):
  from src.thermoformer.lle_tp.solver_certified import solve_many as certified
  m=HybridBinodalLLE({'rdkit_2d':4,'unimol_v2':4,'functional_groups':2},16,1).double()
  q=torch.linspace(-.1,.1,24,dtype=torch.float64)[None];a=torch.full_like(q,-.2);b=torch.full_like(q,.2)
  d=torch.tensor([[1.,-1.,0.]],dtype=torch.float64)/2**.5;n=torch.cross(d,torch.ones_like(d)/3**.5,dim=-1)
  c=(torch.zeros(1,3,16,dtype=torch.float64),torch.zeros(1,1,16,dtype=torch.float64),torch.ones(1,1,dtype=torch.float64)*300,torch.ones(1,1,dtype=torch.float64)*101,q,a,b,d,n,torch.ones(1,3,dtype=torch.float64),torch.zeros(1,2,dtype=torch.float64),torch.ones(1,3,dtype=torch.float64))
  proposal=m.propose(c,8);new=certified(m,c,proposal)[0];old=slow(m,c,proposal)[0]
  self.assertEqual(new['method'],'certified_sampled_roots');self.assertEqual(new['iterations_used'],0);self.assertEqual(len(new['pairs']),8)
  reference=np.array([v['endpoints'] for v in old['pairs']])
  for v in new['pairs']:
   x=np.array(v['endpoints']);distance=np.minimum(((reference-x)**2).sum((1,2)),((reference-x[::-1])**2).sum((1,2)))
   self.assertLess(distance.min(),1e-8)

if __name__=='__main__':unittest.main()
