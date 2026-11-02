from pathlib import Path
from types import SimpleNamespace
import copy,dataclasses,hashlib,importlib.util,json,sys,torch
assert not torch.cuda.is_initialized()
old=Path('/opt/star-prefill/package/reference-evidence/BASE-dsv41_r1.py');new=Path('/usr/local/lib/python3.12/dist-packages/hybrid/dsv41_r1.py');data=old.read_bytes()
assert hashlib.sha256(data).hexdigest()=='e112b065f76d80e61e40b93d6a7128c4ead23954ec47c04b397eb943ca79d5f0'
assert hashlib.sha256(new.read_bytes()).hexdigest()=='0b5388a81b99a0bdc55fdf781107bcba1e02c7e2c3a8f2186875260bb9d8b0ce'
newline=b'\r\n' if b'\r\n' in data else b'\n'
reverse=new.read_bytes().decode('utf-8').replace('\r\n','\n')
for row in reversed(json.loads(Path('/opt/star-prefill/package/reference-evidence/GREEDY-PROPOSAL-PATCH.json').read_text())):reverse=reverse.replace(row['after'],row['before'])
assert reverse.replace('\n',newline.decode()).encode('utf-8')==data
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
a=load('old_draft_plan_cpu',old);b=load('new_draft_plan_cpu',new)
def norm(plan):return None if plan is None else dataclasses.asdict(plan)
def run(m,ids,sps,gens,mark=None):return m.build_draft_plan(ids,sps,gens,m.FlagCache(),glog_enabled=True,mark=mark,vocab_expected=129280)
checks=0;overrides=0
for batch in range(1,9):
 for temperature in (0.,1.,.7):
  for seeded in (False,True):
   for structured in (False,True):
    for generator in (False,True):
     for extra in ({},{'dsv41_draft_tq':.8},{'dsv41_glog':1},{'dsv41_xcheck':1},{'dsv41_proposal_policy':'unknown'},{'dsv41_proposal_policy':True}):
      sps=[SimpleNamespace(temperature=temperature,seed=1017 if seeded else None,structured_outputs={} if structured else None,guided_decoding=None,extra_args=copy.deepcopy(extra)) for _ in range(batch)]
      ids=[str(i) for i in range(batch)];gens=list(range(batch)) if generator else []
      original=copy.deepcopy(sps)
      assert norm(run(a,ids,sps,gens))==norm(run(b,ids,sps,gens));checks+=1
      for sp in sps:sp.extra_args['dsv41_proposal_policy']='greedy'
      p=run(b,ids,sps,gens);random_single=batch==1 and temperature>0 and not seeded and not structured and not generator
      if random_single:
       assert p is not None and p.tq==[0.] and not p.any_sample and not p.all_sample and not p.needs_r1_path or (p is not None and p.any_glog and not p.any_sample)
       overrides+=1
      else:
       # The override does not manufacture sampled proposals for any protected row.
       assert p is None or (not p.any_sample and all(t==0 for t in p.tq))
      for x,y in zip(sps,original):
       assert x.temperature==y.temperature and x.seed==y.seed and x.structured_outputs==y.structured_outputs
receipt=dict(passed=True,source_sha256=hashlib.sha256(data).hexdigest(),candidate_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),unchanged_split_cpu_cases=checks,explicit_random_single_overrides=overrides,reverse_patch_exact=True,target_sampling_fields_unchanged=True,default_policy_unchanged=True,invalid_override_keeps_existing_split=True,installed=False,gpu_qualified=False,scope='CPU draft-plan selection only. Does not qualify real proposal/rejection/q/RNG/distribution or performance. New image/native-source attestations and full GPU startup required before testing; narrow sampled admission remains blocked.')

assert not torch.cuda.is_initialized()
Path('/opt/star-prefill/proposal-policy-cpu.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
