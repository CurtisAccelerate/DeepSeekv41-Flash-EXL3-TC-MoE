"""Explicit qualified private lane selection; speed is the default."""
from pathlib import Path
import argparse,concurrent.futures,copy,json,os,sys,uuid
from lanes import config
def validate_ready(spec,rows,receipts,lane):
 c=config(lane);scope=spec.get('env',{}).get('STAR_ABC_SCOPE')
 if spec.get('kv_cache_dtype')!='bfloat16':raise ValueError('All release lanes require the BF16 KV boot')
 if scope!=c['capture_scope'] and not (lane=='quality' and scope in ('draft-speed','draft-balanced')):raise ValueError('Projection lane differs from captured bank; configure a fresh cold load for '+lane)
 if len(rows)!=4 or len(receipts)!=4 or sorted(x['rank'] for x in rows)!=list(range(4)):raise ValueError('All four qualified ranks required')
 identities={x['identity'] for x in rows}
 if len(identities)!=1:raise ValueError('Rank identities disagree')
 for row,r in zip(sorted(rows,key=lambda x:x['rank']),receipts):
  if not row.get('ok') or row.get('pending') is not None or c['profile'] not in row['available']:raise ValueError('Selected profile is not ready on every rank')
  if not r['ready'] or r['identity']!=row['identity'] or r['capture_scope']!=scope or r['kv_cache_dtype']!='bfloat16':raise ValueError('Durable startup receipt does not bind the live bank')
  if r['candidate_batch_sizes']!=list(range(1,9)) or 5 not in r['candidate_widths']:raise ValueError('All requested fixedfive shapes must be qualified')
  if r['memory']['free']<1073741824 or r['memory']['growth']>1152*1024**2:raise ValueError('Startup capture/floor qualification failed')
  if not r['extra_fold_checks']['passed'] or r['extra_fold_checks']['cases']!=96:raise ValueError('Actual-weight row qualification absent')
  if not r['sampler_interleave']['passed'] or r['sampler_interleave']['cases']!=32:raise ValueError('Sampler ownership qualification absent')
  if not r['prefill_preparation'].get('ready'):raise ValueError('Healthy reference fallback is not a qualified candidate')
  if c['profile'] not in r['available']:raise ValueError('Profile absent from durable startup receipt')
 return next(iter(identities))
def execute(client,bodies,measure,lane='speed',greedy_proposals=False):
 c=config(lane)
 if not 1<=len(bodies)<=8:raise ValueError('Use one through eight request bodies')
 requests=copy.deepcopy(bodies)
 for b in requests:
  if b.get('stream') is not True:raise ValueError('Supply stream=true to record wall/TTFT/gaps')
  if greedy_proposals:b.setdefault('vllm_xargs',{})['dsv41_proposal_policy']='greedy'
 before=client.call(dict(op='abc_status'))
 receipts=[json.loads((client.root/'round-logs'/f'abc-preparation-rank{rank}.json').read_text()) for rank in range(4)]
 identity=validate_ready(client.spec,before['abc'],receipts,lane);session=uuid.uuid4().hex
 report=dict(lane=c,proposal_policy='greedy-opt-in' if greedy_proposals else 'existing-serving-split',requests=[],model_quality_verified=False)
 client.call(dict(op='abc_begin',session=session))
 try:
  report['selection']=client.call(dict(op='abc_select',session=session,profile=c['profile'],costs=None))
  if report['selection']['identity']!=identity:raise RuntimeError('Bank changed during selection')
  report['prefill_selection']=client.call(dict(op='prefill_select',session=session,profile=c['prefill']))
  if not report['prefill_selection']['prefix_cache_reset']:raise RuntimeError('Prefix-cache reset missing')
  def one(body):return measure(client.base_url+'/v1/chat/completions',body,extra_headers=client.headers(session))
  with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as pool:report['requests']=list(pool.map(one,requests))
  if not all(x['transport_complete'] and x['http_status']==200 for x in report['requests']):raise RuntimeError('Incomplete request; inspect saved owner logs')
 finally:report['end']=client.call(dict(op='abc_end',session=session))
 return report
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--request',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--lane',choices=('speed','bf16','quality'),default=os.environ.get('DS1017_LANE','speed'));ap.add_argument('--greedy-proposals',action='store_true');a=ap.parse_args()
 sys.path.insert(0,str(a.root/'package/tools'));from control_client import Client
 from stream_wall_probe import measure
 body=json.loads(a.request.read_text(encoding='utf-8'));result=execute(Client(a.root),body if isinstance(body,list) else [body],measure,a.lane,a.greedy_proposals)
 a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
