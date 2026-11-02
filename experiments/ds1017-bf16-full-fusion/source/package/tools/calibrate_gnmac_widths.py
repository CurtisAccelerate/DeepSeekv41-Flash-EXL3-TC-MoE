"""Finite valid-request local width calibration; never loads or recaptures a model."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,importlib.util,json,shutil,subprocess,sys,threading,time,uuid

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--prefix',choices=['GN-BALANCED','GN-SPEED'],required=True);a=ap.parse_args()
 root=a.root.resolve();assert not a.out.exists();a.out.mkdir(parents=True)
 sys.path.insert(0,str(root/'package/tools'));from control_client import Client
 from stream_wall_probe import measure
 from system_snapshot import snapshot
 from run_matrix import coverage
 import bench_exact as exact
 startup=json.loads((root/'STARTUP-PROGRESS.json').read_text());assert startup['phase']=='qualified' and startup['extra_fold_cases_per_rank']==96
 launch=json.loads((root/'launch.json').read_text());reload=json.loads((root/'RELOAD-RECEIPT.json').read_text())
 owner=json.loads(subprocess.check_output(['docker','-H','unix:///run/dsv41-private/docker.sock','inspect',launch['name']]))[0]
 assert owner['Id']==reload['container'] and owner['Image']==launch['image'] and owner['State']['Running']
 assert launch['env'].get('STAR_ABC_POLICY_OBSERVE')=='1' and launch['env']['STAR_ABC_SCOPE']==('draft-balanced' if a.prefix=='GN-BALANCED' else 'draft-speed')
 assert launch['cmd'][launch['cmd'].index('--kv-cache-dtype')+1]=='bfloat16'
 spec=importlib.util.spec_from_file_location('full_prefill','/dev/shm/prefill_matched_components_v6.py');full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)
 c=Client(root);session=uuid.uuid4().hex;salts=set();doc=dict(schema='star.gnmac.width-calibration.v1',started=time.time(),root=str(root),image=launch['image'],owner=owner['Id'],prefix=a.prefix,cells=[],source_hashes={},context=dict(prompt='Original Morrow hash-map prose',prompt_sha256=hashlib.sha256(exact.PROMPTS['prose'].encode()).hexdigest(),cap=400,temperature=0,sampler='retained FP8-draft+fused greedy sampler, BF16 target head; original sampled path untouched',gate=False,kv='bfloat16',power_w=140),complete=False)
 def save():(a.out/'results.json').write_text(json.dumps(doc,indent=2)+'\n')
 def status():return c.call(dict(op='abc_status'))
 def native():return full.compact_prefill(c.call(dict(op='prefill_status')))
 def batch(b,label):
  barrier=threading.Barrier(b)
  def one(slot):
   salt=uuid.uuid4().hex;salts.add(salt)
   body=dict(model='deepseek-v4.1-flash-1017',messages=[dict(role='user',content=exact.PROMPTS['prose'])],max_tokens=400,temperature=0,top_p=1,stream=True,stream_options=dict(include_usage=True),chat_template_kwargs=exact.KW,cache_salt=salt)
   barrier.wait();r=measure(c.base_url+'/v1/chat/completions',body,label+'-'+str(slot),extra_headers=c.headers(session))
   assert r['transport_complete'] and r['http_status']==200 and r.get('response_id')
   assert r['usage']['prompt_tokens_details']['cached_tokens']==0 and r['usage']['completion_tokens']==400
   r['matched_protocol_sha256']=hashlib.sha256(canonical({k:v for k,v in body.items() if k not in ('model','cache_salt')})).hexdigest()
   return r
  t=time.monotonic()
  with ThreadPoolExecutor(max_workers=b) as pool:rs=list(pool.map(one,range(b)))
  return dict(requests=rs,external_batch_wall_s=time.monotonic()-t,scope='Observer calibration wall only; headline timing uses the separately pinned original benchmark.')
 begin=False;err=None
 try:
  begin=True;doc['begin']=c.call(dict(op='abc_begin',session=session));save()
  for b in range(1,9):
   for k in ((5,3) if b%2 else (3,5)):
    profile=a.prefix+('-WIDE' if k==5 else '-NARROW');selection=c.call(dict(op='abc_select',session=session,profile=profile,costs=None))
    pf=c.call(dict(op='prefill_select',session=session,profile='MOE_RAGGED'));assert pf['prefix_cache_reset']
    before=native();exact.BASE=c.base_url;exact.MODEL='deepseek-v4.1-flash-1017';exact.HEADERS=c.headers(session)
    primer=exact.stream(full.frozen_cases(full.BASELINE,quick=True)[0]['text'],1,900)
    proof=full.hit_coverage(before,native(),'MOE_RAGGED');assert proof['passed']
    row=dict(users=b,width=k,profile=profile,selection=selection,primer=primer,primer_coverage=proof,system=snapshot(),first_use_batch=batch(b,'warm-'+profile+'-'+str(b)),batches=[]);doc['cells'].append(row);save()
    ab=status()
    for rep in range(3):row['batches'].append(batch(b,profile+'-'+str(b)+'-'+str(rep)));save()
    row['coverage']=coverage(ab,status(),profile,b);assert row['coverage']['requested_B_observed_on_all_ranks']
    row['after_native']=native();save()
  assert len(doc['cells'])==16 and len(salts)==288
 except BaseException as exc:err=exc;doc['error']=repr(exc)
 finally:
  if begin:
   try:doc['end']=c.call(dict(op='abc_end',session=session))
   except Exception as exc:doc['end_error']=repr(exc)
  save()
 if err is not None:raise err
 assert doc.get('end') and not doc.get('end_error')
 # No new GPU work: drain the existing background sinks by waiting for stable
 # closed-owner files; analyzer also rejects missing/dropped/incomplete records.
 files=list((root/'round-logs').glob('round-*.jsonl'))+list((root/'state').glob('gnmac-policy-events-pid*.jsonl'))
 assert len([p for p in files if p.name.startswith('round-')])==4 and any(p.name.startswith('gnmac-policy-events') for p in files)
 previous=None
 for _ in range(20):
  sizes=[p.stat().st_size for p in files]
  if sizes==previous:break
  previous=sizes;time.sleep(.25)
 else:raise RuntimeError('Observer logs did not settle after calibration')
 rounds=a.out/'logs';rounds.mkdir();doc['logs']=[]
 for p in files:
  dst=rounds/p.name;shutil.copyfile(p,dst);doc['logs'].append(dict(path='logs/'+p.name,sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),bytes=dst.stat().st_size))
 doc.update(complete=True,finished=time.time(),calibration_requests=216,first_use_requests=72);save();print(json.dumps(dict(complete=True,cells=16,out=str(a.out))))
if __name__=='__main__':main()
