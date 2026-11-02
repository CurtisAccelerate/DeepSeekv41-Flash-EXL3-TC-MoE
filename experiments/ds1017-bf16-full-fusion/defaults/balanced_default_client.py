"""Use the measured balanced recipe through the existing private owner protocol.

Host tool; never starts/stops a model, changes public admission or repoints clients.
The root must be an already-qualified dedicated stage supported by control_client.
"""
from pathlib import Path
import argparse,concurrent.futures,copy,json,sys,uuid

def prepare_request(body,greedy_proposals=False):
 result=copy.deepcopy(body)
 if result.get('stream') is not True:raise ValueError('Supply stream=true for recorded wall/TTFT/gaps')
 if greedy_proposals:
  extras=dict(result.get('vllm_xargs') or {});extras['dsv41_proposal_policy']='greedy';result['vllm_xargs']=extras
 # Cache salts only isolate measurement; caller must request this deliberately.
 return result

def execute(client,bodies,measure,greedy_proposals=False):
 if client.spec.get('kv_cache_dtype')!='bfloat16':raise ValueError('Selected recipe requires the qualified BF16 cache boot')
 if not 1<=len(bodies)<=8:raise ValueError('Qualified concurrency is one through eight')
 requests=[prepare_request(b,greedy_proposals) for b in bodies];session=uuid.uuid4().hex
 report=dict(recipe='balanced-bf16-fixed-five',proposal_policy='greedy-opt-in' if greedy_proposals else 'existing-serving-split',requests=[],model_quality_verified=False)
 client.call(dict(op='abc_begin',session=session))
 try:
  report['selection']=client.call(dict(op='abc_select',session=session,profile='GN-BALANCED-WIDE',costs=None))
  if report['selection']['identity']!='c3fb3c600a8f6e8add3d5d33f580c70b4cae04d89e00ad7b0be0432dac7ac6f4':raise RuntimeError('Source/native/cache bank identity differs from selected release')
  report['prefill_selection']=client.call(dict(op='prefill_select',session=session,profile='MOE_RAGGED'))
  if not report['prefill_selection']['prefix_cache_reset']:raise RuntimeError('Selection did not reset prefix cache')
  def one(body):return measure(client.base_url+'/v1/chat/completions',body,extra_headers=client.headers(session))
  with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as pool:report['requests']=list(pool.map(one,requests))
  if not all(x['transport_complete'] and x['http_status']==200 for x in report['requests']):raise RuntimeError('Incomplete API request; inspect saved exception/owner logs')
 finally:report['end']=client.call(dict(op='abc_end',session=session))
 return report

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--request',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--greedy-proposals',action='store_true',help='Explicit experimental T1 policy; mixed workload gains and seeded-output variation are documented')
 a=ap.parse_args();sys.path.insert(0,str(a.root/'package/tools'));from control_client import Client
 from stream_wall_probe import measure
 body=json.loads(a.request.read_text(encoding='utf-8'));bodies=body if isinstance(body,list) else [body]
 result=execute(Client(a.root),bodies,measure,a.greedy_proposals);a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
