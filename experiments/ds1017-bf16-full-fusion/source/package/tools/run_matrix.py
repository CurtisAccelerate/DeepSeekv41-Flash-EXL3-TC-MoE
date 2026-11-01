"""Post-load paired speed/wall/quality screen. Never loads or restarts a model."""
import argparse,json,uuid,time,hashlib,statistics,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from control_client import Client
from stream_wall_probe import measure
from system_snapshot import snapshot
import bench_exact as exact

CASES=[
 dict(id='facts',prompt='Source record: Maya owns task R7. R7 is blocked waiting for a bearing. Task R8 belongs to Omar and is complete. Reply only JSON with keys blocked_task, owner, blocker, completed_task.',expected={'blocked_task':'R7','owner':'Maya','blocker':'bearing','completed_task':'R8'}),
 dict(id='arithmetic',prompt='Reply only JSON: subtotal is 37 items at 19 dollars each, less an 88 dollar discount. Keys subtotal,discount,total must contain integer dollar amounts.',expected={'subtotal':703,'discount':88,'total':615}),
 dict(id='code',prompt='Reply only JSON with a code string. Implement Python function moving_average(xs, window). Return the arithmetic mean of every complete trailing window. Empty input returns []. Require a positive integer window (bool is invalid); otherwise raise ValueError. Window larger than input returns []. No external packages. Output the function and no calls.',oracle='moving_average'),
 dict(id='binary_search',prompt='Reply only JSON with a code string. Implement Python binary_search(nums, target): return any matching index in a sorted ascending list or -1 if absent. Empty input returns -1. Include no calls and no external packages.',oracle='binary_search'),
]

def grade(case,text,root):
    try:obj=json.loads(text)
    except ValueError:return dict(pass_screen=False,reason='not JSON')
    if 'expected' in case:return dict(pass_screen=obj==case['expected'],actual=obj)
    from code_oracle import check
    return check(case['oracle'],obj.get('code'),root)

def coverage(before,after,profile,b):
    old={r['rank']:r['hits'] for r in before['abc']};rows=[]
    for r in after['abc']:
        delta={k:v-old[r['rank']].get(k,0) for k,v in r['hits'].items() if v!=old[r['rank']].get(k,0)}
        eligible=sum(v for k,v in delta.items() if k.startswith('target:'+profile+':B'+str(b)+':k'))
        rows.append(dict(rank=r['rank'],hits=delta,requested_B_target_hits=eligible))
    return dict(requested_B_observed_on_all_ranks=len(rows)==4 and all(r['requested_B_target_hits']>0 for r in rows),rows=rows)

def finish_quality(doc):
    baseline=next((p for p in doc['profiles'] if p['profile']=='B'),None)
    def key(c):return (c['users'],c['rep'],c['case'],c['slot'])
    if baseline:
        base={key(c):c['response']['content'] for c in baseline['cells']}
        doc['draft_isolation']={str(p['index'])+':'+p['profile']:all(key(c) in base and c['response']['content']==base[key(c)] for c in p['cells']) for p in doc['profiles'] if p['profile'] in ['C','C-FP8','C-FUSED','C-GATE','C-ALL','C3','C-AUTO']}
    doc['fast_screen_pass']=bool(doc['profiles']) and all(p['cells'] and all(c.get('screen',{}).get('pass_screen') is True and c['completed_without_cap'] and c.get('coverage',{}).get('requested_B_observed_on_all_ranks') is True for c in p['cells']) for p in doc['profiles']) and all(doc.get('draft_isolation',{}).values())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path)
    ap.add_argument('--profiles',default='A,B,C,A');ap.add_argument('--phase',choices=['speed','quality','wall'],default='quality')
    ap.add_argument('--levels',default='1,2,3,4,5,6,7,8');ap.add_argument('--reps',type=int,default=3)
    ap.add_argument('--quality-reps',type=int,default=1)
    ap.add_argument('--costs',type=Path,help='Locally measured C/C3 full-cycle table, only for C-AUTO')
    ap.add_argument('--routes',type=Path,help='Measured per-B expert map, installed while A is selected')
    ap.add_argument('--frozen-requests',type=Path,help='JSON list of exact request bodies; no automatic tool execution')
    ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
    profiles=a.profiles.split(',');levels=list(map(int,a.levels.split(',')))
    if not levels or any(b not in range(1,9) for b in levels) or a.reps<1 or a.quality_reps<1:ap.error('B1..8 and positive repeat counts required')
    costs=json.loads(a.costs.read_text()) if a.costs else None
    if 'C-AUTO' in profiles and costs is None:ap.error('C-AUTO requires --costs; donor constants are not accepted')
    if any(p.endswith('-ROUTED') for p in profiles) and a.routes is None:ap.error('Routed arms require --routes')
    client=Client(a.root);session=uuid.uuid4().hex;a.out.mkdir(parents=True,exist_ok=False)
    doc=dict(schema='star.abc.measurement.v1',phase=a.phase,profiles=[],started=time.time(),model_quality_verified=False,production_approved=False)
    spec=json.loads((a.root/'launch.json').read_text())
    doc['boot']=dict(image=spec['image'],kv_cache_dtype=spec.get('kv_cache_dtype','fp8_ds_mla'),comparison_contract_sha256=spec.get('comparison_contract_sha256'),root=str(a.root),launch_sha256=hashlib.sha256((a.root/'launch.json').read_bytes()).hexdigest())
    def save():(a.out/'results.json').write_text(json.dumps(doc,indent=2,ensure_ascii=False))
    client.call(dict(op='abc_status'));client.call(dict(op='abc_begin',session=session))
    exact.BASE=client.base_url;exact.MODEL='deepseek-v4.1-flash-1017';exact.HEADERS=client.headers(session)
    try:
        if a.routes:doc['routing']=client.call(dict(op='abc_route',session=session,routing=json.loads(a.routes.read_text())))
        for index,profile in enumerate(profiles):
            receipt=client.call(dict(op='abc_select',session=session,profile=profile,costs=costs if profile=='C-AUTO' else None))
            before=client.call(dict(op='abc_status'))
            record=dict(index=index,profile=profile,selection=receipt,cells=[],before=before,system_before=snapshot());doc['profiles'].append(record);save()
            if a.phase=='speed':
                for name in exact.PROMPTS:
                    exact.stream(exact.WARMUPS[name],32,900)
                    for b in levels:
                        cell_before=client.call(dict(op='abc_status'))
                        batches=[exact.morrow_batch(name,b) for _ in range(a.reps)]
                        cell_after=client.call(dict(op='abc_status'))
                        record['cells'].append(dict(prompt=name,users=b,batches=batches,coverage=coverage(cell_before,cell_after,profile,b),S=statistics.median(x['batch_mean_s'] for x in batches),D=statistics.median(x['d_tok_s'] for x in batches),E=statistics.median(x['e_tok_s'] for x in batches),wall_s=statistics.median(x['wall_s'] for x in batches)));save()
            else:
                cases=CASES if a.frozen_requests is None else [{'id':str(i),'body':b} for i,b in enumerate(json.loads(a.frozen_requests.read_text()))]
                # Frozen real requests are replayed verbatim at B1; built-in screens exercise each B.
                for b in ([1] if a.frozen_requests is not None else levels):
                    for rep in range(a.quality_reps):
                        for case in cases:
                            cell_before=client.call(dict(op='abc_status'))
                            barrier=threading.Barrier(b)
                            def one(slot):
                                body=case.get('body') or dict(model=exact.MODEL,messages=[dict(role='user',content=case['prompt'])],temperature=0,top_p=1,seed=1017,max_tokens=1500,stream=True,stream_options={'include_usage':True},chat_template_kwargs={'thinking':False,'enable_thinking':False},cache_salt=uuid.uuid4().hex)
                                if body.get('stream') is not True:raise ValueError('Frozen requests must explicitly request streaming')
                                barrier.wait()
                                response=measure(exact.BASE+'/v1/chat/completions',body,label=profile+':'+case['id'],extra_headers=client.headers(session))
                                complete=response['transport_complete'] and bool(response['finish_reasons']) and not any(r=='length' for r in response['finish_reasons'])
                                return dict(case=case['id'],users=b,rep=rep,slot=slot,response=response,completed_without_cap=complete)
                            with ThreadPoolExecutor(max_workers=b) as pool:results=list(pool.map(one,range(b)))
                            cell_coverage=coverage(cell_before,client.call(dict(op='abc_status')),profile,b)
                            for result in results:
                                result['coverage']=cell_coverage
                                if a.phase=='quality' and 'body' not in case:result['screen']=grade(case,result['response']['content'],a.root)
                                record['cells'].append(result)
                            save()
            record['after']=client.call(dict(op='abc_status'));record['system_after']=snapshot();save()
        if a.phase=='quality' and a.frozen_requests is None:
            finish_quality(doc)
        doc['finished']=time.time();save()
    finally:
        try:doc['end']=client.call(dict(op='abc_end',session=session))
        except Exception as exc:doc['end_error']=repr(exc)
        save()
    print(json.dumps(dict(output=str(a.out),fast_screen_pass=doc.get('fast_screen_pass'),model_quality_verified=False,production_approved=False)))

if __name__=='__main__':main()

