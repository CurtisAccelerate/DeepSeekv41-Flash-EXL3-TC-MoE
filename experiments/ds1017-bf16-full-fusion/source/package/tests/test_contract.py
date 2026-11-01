import asyncio,importlib,json,os,sys,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'overlay/usr/local/lib/python3.12/dist-packages'))
from hybrid.abc_contract import *
from hybrid import abc_admission

class ContractTests(unittest.TestCase):
    def test_token_specific_and_prompt_logprobs_never_receive_fake_logits(self):
        from hybrid.abc_greedy import logits
        for md,prompt in [(SimpleNamespace(logprob_token_ids={'r':[42]}),{}),(SimpleNamespace(logprob_token_ids=None),{'r':3})]:
            runner=SimpleNamespace(input_batch=SimpleNamespace(sampling_metadata=md),num_prompt_logprobs=prompt)
            with patch('hybrid.abc_runtime.current',return_value=PROFILES['C']):
                self.assertIsNone(logits(runner,None,SimpleNamespace(has_structured_output_requests=False)))
    def test_target_ablations_retain_incumbent_collectives_and_48_row_tc(self):
        from hybrid.abc_startup import Plan
        for name in ('B-DENSE','B-WOA'):
            self.assertEqual(Plan(8,5,PROFILES[name]).experts,'tc')
            self.assertFalse(lane_rows(PROFILES[name].experts,8,5))
            self.assertEqual(Plan(7,5,PROFILES[name]).served_b,8)
            self.assertEqual(Plan(2,5,PROFILES[name]).collective,(1,1024,1))
            self.assertEqual(Plan(1,5,PROFILES[name]).collective,(1,512,1))
            self.assertEqual(Plan(8,5,PROFILES[name]).collective,'pair')
    def test_same_rows_are_different_families(self):
        self.assertNotEqual(descriptor(2,5),descriptor(3,3))
        self.assertEqual(descriptor(2,5)[0],descriptor(3,3)[0])
    def test_hybrid_preserves_our_multiuser_range(self):
        self.assertEqual([b for b in range(1,9) if lane_rows('hybrid',b,5)],[1,7,8])
        self.assertFalse(PROFILES['B'].fp8_draft);self.assertFalse(PROFILES['B'].gate)
    def test_prefill_mixed_and_nonuniform_are_ineligible(self):
        self.assertIsNone(eligible_layout({'r':6},{}))
        self.assertIsNone(eligible_layout({'a':6,'b':4},{'a':[0]*5,'b':[0]*3}))
        self.assertEqual(eligible_layout({'a':4,'b':4},{'a':[0]*3,'b':[0]*3}),(2,3))
    def test_selection_requires_current_identity_and_epoch(self):
        s=Selection('pin',['A','B'])
        for ident,epoch,profile in [('wrong',1,'B'),('pin',0,'B'),('pin',1,'C')]:
            with self.assertRaises(ValueError):s.prepare('t',profile,'session',ident,epoch)
        self.assertEqual(s.selected,'A');self.assertIsNone(s.pending)
    def test_prepare_does_not_select_commit_and_rollback_are_bound(self):
        s=Selection('pin',['A','B']);s.prepare('t','B','session','pin',1)
        self.assertEqual(s.selected,'A')
        with self.assertRaises(ValueError):s.commit('different')
        s.rollback('t');self.assertEqual(s.selected,'A')
        s.prepare('u','B','session','pin',1);s.commit('u');self.assertEqual(s.selected,'B')
        with self.assertRaises(ValueError):s.prepare('v','A','session','pin',1)
        s.end();self.assertIsNone(s.session);self.assertEqual(s.selected,'A')
    def test_no_donor_or_stale_costs(self):
        for value in [None,{},dict(schema='star.abc.costs.v1',identity='wrong')]:
            with self.assertRaises(ValueError):validate_costs(value,'pin')
        good=dict(schema='star.abc.costs.v1',identity='pin',hardware='STAR-140W',source='measured_full_cycle',evidence_sha256='a'*64,milliseconds={str(b):{'3':20.,'5':25.} for b in range(1,9)})
        self.assertEqual(validate_costs(good,'pin')['1']['3'],20.)
        good['milliseconds']['8']['3']=float('nan')
        with self.assertRaises(ValueError):validate_costs(good,'pin')
    def test_routes_require_all_rows_and_matched_identity(self):
        good=dict(schema='star.abc.routes.v1',identity='pin',hardware='STAR-140W',source='matched_star_wall_and_quality',evidence_sha256='a'*64,experts={str(b):'hybrid' for b in range(1,9)})
        self.assertEqual(validate_routes(good,'pin'),good)
        with self.assertRaises(ValueError):validate_routes(good,'different')
        good['experts'].pop('8')
        with self.assertRaises(ValueError):validate_routes(good,'pin')
    def test_invalid_geometry_rejects_bools(self):
        for b,k in [(True,5),(0,5),(9,5),(1,True),(1,4)]:
            with self.assertRaises(ValueError):descriptor(b,k)

class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def check_body(self,obj,valid):
        messages=[{'type':'http.request','body':json.dumps(obj).encode(),'more_body':False}];sent=[]
        async def receive():return messages.pop(0)
        async def send(item):sent.append(item)
        replay=await abc_admission.receive_for(SimpleNamespace(abc_narrow=True),{'method':'POST'},receive,send)
        self.assertEqual(replay is not None,valid)
        if valid:self.assertEqual(json.loads((await replay())['body']),obj)
        else:self.assertEqual(sent[0]['status'],400)
    async def test_sampled_and_implicit_defaults_blocked_for_narrow(self):
        await self.check_body({},False);await self.check_body({'temperature':1},False)
        await self.check_body({'temperature':0,'n':2},False)
        await self.check_body({'temperature':0,'n':1},True)
    async def test_wide_mode_does_not_consume_or_modify_body(self):
        async def receive():raise AssertionError('Wide mode consumed body')
        self.assertIs(await abc_admission.receive_for(SimpleNamespace(abc_narrow=False),{'method':'POST'},receive,None),receive)

class IntegrationShapeTests(unittest.TestCase):
    def test_all_source_parses_and_guards_are_present(self):
        import ast
        for p in (ROOT/'overlay').rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
        r=(ROOT/'overlay/usr/local/lib/python3.12/dist-packages/hybrid/abc_runtime.py').read_text()
        self.assertIn("_abc_layout",r)
        gate=(ROOT/'overlay/usr/local/lib/python3.12/dist-packages/hybrid/abc_gate.py').read_text()
        self.assertIn('num_draft_tokens!=[5]*b',gate)
        self.assertIn('MASK.fill_(1)',gate)

if __name__=='__main__':unittest.main()

