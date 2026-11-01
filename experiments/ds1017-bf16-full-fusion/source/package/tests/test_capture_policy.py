import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'overlay/usr/local/lib/python3.12/dist-packages'))
from hybrid.abc_contract import capture_policy,PROFILES

class CapturePolicyTests(unittest.TestCase):
    def test_abc_supports_b_and_c_every_user_count_without_target_ablation_banks(self):
        plan=capture_policy('abc',True)
        self.assertEqual(plan['batches'],tuple(range(1,9)))
        self.assertEqual(plan['widths'],(5,))
        families={PROFILES[n].family() for n in plan['families']}
        self.assertEqual(len(families),1)
        keys=set(plan['samplers'])|set(plan['reference_samplers'])
        for name in ('B','C'):
            p=PROFILES[name];self.assertIn(p.family(),families)
            for b in range(1,9):self.assertIn((b,p.fp8_draft and b<=3,p.fused_markov,p.gate),keys)
        self.assertEqual(len(plan['samplers']),8)
        self.assertEqual(len(plan['reference_samplers']),8)
        self.assertTrue(all(not key[3] for key in keys))
        self.assertFalse(any(b>3 and fp8 for b,fp8,_,_ in plan['samplers']))
    def test_existing_scopes_and_gate_off_contract(self):
        self.assertEqual(capture_policy('b1',True)['batches'],(1,))
        self.assertTrue(all(not g for _,_,_,g in capture_policy('b1',True)['samplers']))
        self.assertEqual(capture_policy('all',True)['widths'],(5,3))
        self.assertTrue(all(not g for _,_,_,g in capture_policy('abc',False)['samplers']))
        with self.assertRaises(ValueError):capture_policy('unknown',True)
    def test_reference_aliases_preserve_original_objects_and_reject_missing(self):
        from types import SimpleNamespace as NS
        from hybrid.abc_runtime import reference_sampler
        states={b:NS(sample=lambda x:x) for b in range(1,9)}
        bank=NS(reference_sampler=states[1],r2=NS(greedy={b:s for b,s in states.items() if b>1}))
        for b in range(1,9):self.assertIs(reference_sampler(bank,b),states[b])
        bank.r2.greedy.pop(8)
        with self.assertRaises(ValueError):reference_sampler(bank,8)

if __name__=='__main__':unittest.main()
