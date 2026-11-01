"""Post-load mixed modes, joins/finishes, cancellation and constrained-output checks."""
import argparse,json,uuid,time,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request,urlopen
from control_client import Client
from precision_common import parse_response,compare_ids
from system_snapshot import snapshot

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--profiles',default='A,B,C,A');a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);client=Client(a.root);session=uuid.uuid4().hex
    doc=dict(schema='star.abc.stability.v1',profiles=[],model_quality_verified=False)
    def save():(a.out/'results.json').write_text(json.dumps(doc,indent=2))
    def body():return dict(model='deepseek-v4.1-flash-1017',messages=[dict(role='user',content='Return only JSON {"ok":true,"n":3}.')],temperature=0,top_p=1,max_tokens=64,stream=False,return_token_ids=True,chat_template_kwargs={'thinking':False,'enable_thinking':False},cache_salt=uuid.uuid4().hex)
    def call(obj,barrier=None,cancel=False):
        if barrier:barrier.wait()
        start=time.perf_counter()
        with urlopen(Request(client.base_url+'/v1/chat/completions',json.dumps(obj).encode(),client.headers(session)),timeout=180) as resp:
            if cancel:
                raw=resp.read1(128)
                return dict(client_cancelled=True,read_bytes=len(raw),wall_s=time.perf_counter()-start,request=obj)
            raw=json.loads(resp.read())
        ch=raw['choices'][0];text=ch['message'].get('content');parsed=None
        try:parsed=json.loads(text)
        except (ValueError,TypeError):pass
        return dict(request=obj,raw=raw,wall_s=time.perf_counter()-start,json_pass=parsed==dict(ok=True,n=3),**parse_response(raw))
    client.call(dict(op='abc_begin',session=session))
    try:
        for profile in a.profiles.split(','):
            selection=client.call(dict(op='abc_select',session=session,profile=profile,costs=None))
            rec=dict(profile=profile,selection=selection,system_before=snapshot(),before=client.call(dict(op='abc_status')),cases=[]);doc['profiles'].append(rec);save()
            initial=call(body());rec['cases'].append(dict(kind='initial_B1',result=initial))
            q_before=client.call(dict(op='abc_status'));req=body();req['temperature']=1.;sampled=call(req);q_after=client.call(dict(op='abc_status'))
            previous={r['rank']:r['hits'] for r in q_before['abc']}
            q_hits={r['rank']:r['hits'].get('sampled:'+profile+':B1',0)-previous[r['rank']].get('sampled:'+profile+':B1',0) for r in q_after['abc']}
            rec['cases'].append(dict(kind='unseeded_T1_B1',result=sampled,sampled_bank_hits=q_hits))
            if profile!='A' and not all(v>0 for v in q_hits.values()):raise RuntimeError('Candidate BF16 sampled B1 bank was not observed on every rank')
            for scenario in ('mixed_sampling','constrained','join_leave_cancel'):
                barrier=threading.Barrier(8)
                def one(slot):
                    req=body();cancel=scenario=='join_leave_cancel' and slot in (0,3)
                    if scenario=='mixed_sampling' and slot%2:req['temperature']=1.;req['seed']=1017+slot
                    if scenario=='constrained':req['response_format']={'type':'json_schema','json_schema':{'name':'answer','schema':{'type':'object','properties':{'ok':{'type':'boolean'},'n':{'type':'integer'}},'required':['ok','n'],'additionalProperties':False}}}
                    if cancel:
                        req.update(stream=True,max_tokens=2000);req['messages']=[dict(role='user',content='Count from 1 to 10000, separated by spaces.')]
                    return call(req,barrier,cancel)
                with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(one,range(8)))
                rec['cases'].append(dict(kind=scenario,results=results));save()
            final=call(body());rec['cases'].append(dict(kind='final_B1',result=final));rec['B1_after_burst_exact']=compare_ids(initial,final)
            rec['after']=client.call(dict(op='abc_status'));rec['system_after']=snapshot()
            rec['bounded_screen_pass']=initial['json_pass'] and sampled['json_pass'] and final['json_pass'] and rec['B1_after_burst_exact']['exact'] and all(r.get('client_cancelled') or (r['json_pass'] and r['finish']!='length') for c in rec['cases'] if 'results' in c for r in c['results'])
            save()
    finally:
        try:doc['end']=client.call(dict(op='abc_end',session=session))
        except Exception as exc:doc['end_error']=repr(exc)
        save()
    print(json.dumps(dict(output=str(a.out),model_quality_verified=False)))

if __name__=='__main__':main()
