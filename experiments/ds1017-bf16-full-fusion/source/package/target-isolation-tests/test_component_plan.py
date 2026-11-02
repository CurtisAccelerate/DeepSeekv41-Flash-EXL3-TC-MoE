import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'overlay/usr/local/lib/python3.12/dist-packages'))
from hybrid.abc_contract import PROFILES,capture_policy,lane_rows

class IsolationContractTests(unittest.TestCase):
 def test_only_one_target_feature_changes_in_each_removal(self):
  base=PROFILES['B']
  for name,field in [('B-NO-DENSE','dense'),('B-NO-WOA','woa'),('B-OWN-EXPERTS','experts')]:
   other=PROFILES[name]
   self.assertEqual([k for k in vars(base) if k!='name' and getattr(base,k)!=getattr(other,k)],[field])
 def test_drafter_sampling_and_width_are_identical(self):
  for name in capture_policy('isolate',True)['families']:
   p=PROFILES[name]
   self.assertEqual((p.fp8_draft,p.fused_markov,p.gate,p.width),(False,False,False,5))
 def test_no_extra_heads_gates_or_narrow_banks(self):
  for has_confidence in (False,True):
   p=capture_policy('isolate',has_confidence)
   self.assertEqual(p['batches'],(1,2,8));self.assertEqual(p['widths'],(5,))
   self.assertEqual(p['samplers'],())
   self.assertEqual(p['reference_samplers'],((1,False,False,False),(2,False,False,False),(8,False,False,False)))
 def test_target_matrix_has_every_single_removal(self):
  p=capture_policy('isolate',True)
  self.assertEqual(set(p['families']),{'B','B-NO-DENSE','B-NO-WOA','B-OWN-EXPERTS'})
  self.assertEqual(len({PROFILES[n].family() for n in p['families']}),4)
 def test_original_experts_cannot_use_alternate_lane(self):
  for b in range(1,9):self.assertFalse(lane_rows(PROFILES['B-OWN-EXPERTS'].experts,b,5))
 def test_hybrid_keeps_our_multiuser_expert_shapes(self):
  self.assertEqual([b for b in range(1,9) if lane_rows('hybrid',b,5)],[1,7,8])
 def test_existing_complete_bank_contract_is_unchanged(self):
  p=capture_policy('abc',True)
  self.assertEqual(p['batches'],tuple(range(1,9)));self.assertEqual(p['families'],('B',))
  self.assertEqual(p['widths'],(5,));self.assertEqual(len(p['samplers']),8)
 def test_unknown_capture_scope_still_fails(self):
  with self.assertRaises(ValueError):capture_policy('isolate-unbounded',True)

if __name__=='__main__':unittest.main()
