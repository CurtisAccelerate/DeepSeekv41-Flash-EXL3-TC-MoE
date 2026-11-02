import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'overlay/usr/local/lib/python3.12/dist-packages'))
from hybrid.abc_contract import PROFILES,capture_policy,lane_rows,descriptor
class SelectedPlanTests(unittest.TestCase):
 def test_all_users_and_no_narrow_sampler_changes(self):
  for confidence in (False,True):
   p=capture_policy('selected',confidence)
   self.assertEqual(p['batches'],tuple(range(1,9)))
   self.assertEqual(p['widths'],(5,));self.assertEqual(p['samplers'],())
   self.assertEqual(p['reference_samplers'],tuple((b,False,False,False) for b in range(1,9)))
 def test_only_projection_differs(self):
  a,b=PROFILES['B'],PROFILES['B-NO-WOA']
  self.assertEqual([k for k in vars(a) if k!='name' and getattr(a,k)!=getattr(b,k)],['woa'])
 def test_retained_experts(self):
  self.assertEqual([b for b in range(1,9) if lane_rows('hybrid',b,5)],[1,7,8])
  self.assertEqual(capture_policy('selected',True)['families'],('B','B-NO-WOA'))
 def test_full_finite_bank_and_original_descriptor(self):
  p=capture_policy('selected',False)
  self.assertEqual(len(p['families'])*len(p['batches']),16)
  for b in range(1,9):self.assertEqual(descriptor(b,5),(6*b,b,True,False,0))
if __name__=='__main__':unittest.main()
