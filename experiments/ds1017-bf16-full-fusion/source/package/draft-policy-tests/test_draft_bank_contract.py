import ast,ctypes,re,sys,unittest
from pathlib import Path
PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'overlay/usr/local/lib/python3.12/dist-packages'))
from hybrid.abc_contract import PROFILES,capture_policy,lane_rows,validate_policy_costs
class DraftBankContract(unittest.TestCase):
 def test_only_one_target_family_per_draft_scope(self):
  for scope,prefix in [('draft-balanced','GN-BALANCED'),('draft-speed','GN-SPEED')]:
   p=capture_policy(scope,True)
   self.assertEqual(p['families'],(prefix+'-WIDE',))
   self.assertEqual(p['batches'],tuple(range(1,9)));self.assertEqual(p['widths'],(5,3))
 def test_all_sampler_and_width_variants_prepared(self):
  for scope,prefix in [('draft-balanced','GN-BALANCED'),('draft-speed','GN-SPEED')]:
   p=capture_policy(scope,True);keys=set(p['samplers'])|set(p['reference_samplers'])
   for suffix in ['WIDE','NARROW','AUTO','GATE','AUTO-GATE','REFERENCE-DRAFT','WIDE-ORACLE','NARROW-ORACLE']:
    pr=PROFILES[prefix+'-'+suffix]
    self.assertEqual(pr.family(),PROFILES[prefix+'-WIDE'].family())
    for b in range(1,9):self.assertIn((b,pr.fp8_draft and b<=3,pr.fused_markov,pr.gate),keys)
 def test_gate_absent_without_confidence_head(self):
  for scope in ('draft-balanced','draft-speed'):
   self.assertTrue(all(not g for b,fp8,fused,g in capture_policy(scope,False)['samplers']))
 def test_balanced_never_enables_fp8_projection(self):
  self.assertTrue(all(not p.woa for n,p in PROFILES.items() if n.startswith('GN-BALANCED-')))
  self.assertTrue(all(p.woa for n,p in PROFILES.items() if n.startswith('GN-SPEED-')))
 def test_width_routes_preserve_unique_shapes(self):
  self.assertEqual([b for b in range(1,9) if lane_rows('hybrid',b,5)],[1,7,8])
  self.assertEqual([b for b in range(1,9) if lane_rows('hybrid',b,3)],[1])
  self.assertEqual([4*b for b in range(1,9)],[4,8,12,16,20,24,28,32])
 def test_only_narrow_oracles_change_fixed_width(self):
  for prefix in ('GN-BALANCED','GN-SPEED'):
   for k,name in [(3,'NARROW'),(5,'WIDE')]:
    a=PROFILES[prefix+'-'+name];b=PROFILES[prefix+'-'+name+'-ORACLE']
    self.assertEqual(a.width,k);self.assertEqual(b.width,k)
    self.assertEqual([n for n in vars(a) if n!='name' and getattr(a,n)!=getattr(b,n)],['oracle'])
 def test_original_fold_module_is_unchanged(self):
  import hashlib
  p=PACKAGE/'host-overlay/integration/w9_exl_ab/fold_serve.py'
  self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'4fd2ece81de428012f30ddeaa01210260817b10121ef6bef96e29cde0be1fe60')
 def test_oracle_retains_plain_greedy_guard(self):
  s=(PACKAGE/'overlay/usr/local/lib/python3.12/dist-packages/hybrid/abc_fidelity.py').read_text(encoding='utf-8')
  self.assertIn('not sampling_metadata.all_greedy',s)
  self.assertIn('k=profile.width',s);self.assertIn('n not in (0,k)',s)

 def test_costs_bind_target_sampler_and_gate_before_admission(self):
  import copy
  identity='a'*64
  c=dict(schema='star.abc.costs.v1',identity=identity,hardware='STAR-140W',source='measured_full_cycle',evidence_sha256='b'*64,milliseconds={str(b):{'3':10.,'5':12.} for b in range(1,9)},context=dict(target_family=[True,False,'hybrid'],fp8_draft=True,fused_markov=True,gate=False,draft_tokens=5,calibration_prompt_sha256='c'*64))
  validate_policy_costs(c,identity,'GN-BALANCED-AUTO')
  for key,bad in [('target_family',[True,True,'hybrid']),('fp8_draft',False),('fused_markov',False),('gate',True),('draft_tokens',3),('calibration_prompt_sha256','')]:
   wrong=copy.deepcopy(c);wrong['context'][key]=bad
   with self.assertRaises(ValueError):validate_policy_costs(wrong,identity,'GN-BALANCED-AUTO')
  with self.assertRaises(ValueError):validate_policy_costs(c,identity,'GN-BALANCED-AUTO-GATE')
  c['context']['gate']=True;validate_policy_costs(c,identity,'GN-BALANCED-AUTO-GATE')
  text=(PACKAGE/'overlay/usr/local/lib/python3.12/dist-packages/hybrid/abc_control.py').read_text(encoding='utf-8')
  node=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.AsyncFunctionDef) and n.name=='select')
  checks=[n.lineno for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='validate_policy_costs']
  close=[n.lineno for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='close']
  self.assertLess(min(checks),min(close))

if __name__=='__main__':unittest.main()
