"""CPU verifier for actual-request steady full-cycle width costs. Not a benchmark runner."""
from pathlib import Path
import argparse,hashlib,json,math,statistics

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def analyze(raw,out):
 d=read(raw);assert d['schema']=='star.gnmac.width-calibration.v1' and d['complete'] and not d.get('error') and not d.get('end_error') and d.get('end')
 assert len(d['cells'])==16 and d['calibration_requests']==216 and d['first_use_requests']==72
 logs={};observer=[];logpins=[]
 for row in d['logs']:
  p=raw.parent/row['path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes'];logpins.append(row)
  rows=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
  assert not any(x.get('sink_dropped',0) for x in rows),'Observer dropped records'
  if p.name.startswith('round-'):
   assert rows[0]['type']=='header';rank=rows[0]['rank'];assert rank not in logs
   assert rows[0]['boot']['async_scheduling'] and rows[0]['boot']['num_speculative_tokens']==5
   records=[x for x in rows[1:] if x['type'] in ('round','attest')];by={x['it']:x for x in records};assert len(by)==len(records)
   logs[rank]=by
  else:observer.extend(rows)
 assert set(logs)=={0,1,2,3} and observer
 identities={c['selection']['identity'] for c in d['cells']};assert len(identities)==1;identity=next(iter(identities))
 seen=set();global_protocol=None;rows=[];salts=set()
 for cell in d['cells']:
  b,k,p=cell['users'],cell['width'],cell['profile'];assert (b,k) not in seen and b in range(1,9) and k in (3,5);seen.add((b,k))
  assert p==d['prefix']+('-WIDE' if k==5 else '-NARROW') and cell['selection']['prefix_cache_reset']
  assert cell['primer_coverage']['passed'] and cell['coverage']['requested_B_observed_on_all_ranks']
  assert len(cell['primer_coverage']['rows'])==4 and all(x['full_target_layer_coverage'] for x in cell['primer_coverage']['rows'])
  assert len(cell['system']['gpus'])==4 and all(float(g['power_w'])==140 and float(g['temperature_c'])<80 for g in cell['system']['gpus'])
  tag={key:cell['selection'][key] for key in ('identity','epoch','session')}
  def tagged(r):return all(r.get('abc',{}).get(key)==v for key,v in tag.items()) and r.get('abc',{}).get('profile')==p
  current=[r for r in logs[0].values() if tagged(r)];assert current
  available_ids={rid for r in current for rid in r.get('nd',{})}
  rep_rows=[]
  assert len(cell['batches'])==3
  for rep,batch in enumerate(cell['batches']):
   assert len(batch['requests'])==b;reqids=set()
   for request in batch['requests']:
    assert request['transport_complete'] and request['usage']['completion_tokens']==400 and request['usage']['prompt_tokens_details']['cached_tokens']==0
    body=request['request_body'];salt=body['cache_salt'];assert salt not in salts;salts.add(salt)
    assert body['temperature']==0 and body['top_p']==1 and body['max_tokens']==400
    protocol=hashlib.sha256(json.dumps({key:v for key,v in body.items() if key not in ('model','cache_salt')},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert protocol==request['matched_protocol_sha256']
    if global_protocol is None:global_protocol=protocol
    assert protocol==global_protocol
    rid=request['response_id'];matches=[x for x in available_ids if x==rid or x.startswith(rid+'-')]
    assert len(matches)==1,'Response-to-worker request mapping missing or ambiguous';reqids.add(matches[0])
   assert len(reqids)==b
   shape=[];key=f'target:{p}:B{b}:k{k}'
   for r in current:
    if set(r.get('nd',{}))!=reqids or r.get('decode')!=b or r.get('prefill')!=0 or r.get('B')!=b or r.get('tokens')!=(k+1)*b or set(r['nd'].values())!={k}:continue
    assert r['complete'] and r['graph']['dispatch']=='FULL' and r['graph']['path']=='abc_private_bank'
    assert r['graph']['abc_target_hits'].get(key)==1
    for rank in range(4):
     a=logs[rank].get(r['it']);assert a is not None and a['complete'] and tagged(a)
     assert a['abc']['hits'].get(key)==1,'Expected exact selected graph missing on rank'
    assert r['T']['greedy']==b and r['T']['random']==0
    shape.append(r)
   shape.sort(key=lambda x:x['it']);intervals=[]
   for left,right in zip(shape,shape[1:]):
    if right['it']!=left['it']+1:continue
    ms=(right['t_mono_ns']-left['t_mono_ns'])/1e6;assert math.isfinite(ms) and ms>0
    intervals.append(dict(iteration=left['it'],next_iteration=right['it'],milliseconds=ms))
   assert len(intervals)>=20,'Too few real steady full-batch iteration intervals'
   obs=[e for e in observer if e.get('identity')==identity and e.get('epoch')==tag['epoch'] and e.get('session')==tag['session'] and e.get('profile')==p]
   choices=[e for e in obs if e['type']=='choose' and set(e['request_ids'])==reqids]
   accepted=[e for e in obs if e['type']=='observe' and e['request_id'] in reqids]
   assert len(choices)>=len(intervals) and all(e['mode']=='fixed' and e['width']==k for e in choices) and len(accepted)>=len(intervals)*b
   assert all(e['accepted_raw']==e['accepted'] and 0<=e['accepted']<=e['verified'] and e['verified']==k for e in accepted),'Acceptance callback geometry/values invalid'
   rep_rows.append(dict(rep=rep,intervals=intervals,mean_full_cycle_ms=statistics.mean(x['milliseconds'] for x in intervals),observed_choices=len(choices),acceptance_callbacks=len(accepted),accepted_drafts=sum(e['accepted'] for e in accepted),verified_drafts=sum(e['verified'] for e in accepted),raw_external_wall_s=batch['external_batch_wall_s']))
  value=statistics.median(x['mean_full_cycle_ms'] for x in rep_rows)
  rows.append(dict(users=b,width=k,profile=p,milliseconds=value,repetitions=rep_rows))
 assert seen=={(b,k) for b in range(1,9) for k in (3,5)} and len(salts)==216
 out.mkdir(parents=True,exist_ok=False)
 evidence=dict(passed=True,schema='star.gnmac.full-cycle-evidence.v1',identity=identity,raw_sha256=sha(raw),logs=logpins,context=d['context'],rows=rows,method='Median of3 per-request-batch MEAN worker start-to-next-start intervals. Same-requestset, sameB/k, pure decode, consecutive iterations, actual requested graph on all4ranks. Includes asynchronous pacing/CPU/GPU/collective/draft/verify; NOT isolated kernel latency. No within-repetition stall trimming. Separate full first-use batches excluded and retained in raw.',scope='Measured on the declared Morrow prose workload at140W and BF16KV. Local policy-price evidence, not general throughput or quality prediction.')
 ep=out/'COST-EVIDENCE.json';ep.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
 costs=dict(schema='star.abc.costs.v1',identity=identity,hardware='STAR-140W',source='measured_full_cycle',evidence_sha256=sha(ep),context=dict(target_family=[True,d['prefix']=='GN-SPEED','hybrid'],fp8_draft=True,fused_markov=True,gate=False,draft_tokens=5,calibration_prompt_sha256=d['context']['prompt_sha256']),milliseconds={str(b):{str(k):next(r['milliseconds'] for r in rows if (r['users'],r['width'])==(b,k)) for k in (3,5)} for b in range(1,9)})
 (out/'LOCAL-COSTS.json').write_text(json.dumps(costs,indent=2)+'\n',encoding='utf-8');return costs
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();c=analyze(a.results,a.out);print(json.dumps(dict(passed=True,points=16,identity=c['identity'],evidence_sha256=c['evidence_sha256'])))
