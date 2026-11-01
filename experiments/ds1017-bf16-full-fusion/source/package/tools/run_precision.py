"""Post-load numeric-token, same-prefix decode NLL/KL, and verifier controls."""
import argparse,json,time,uuid,statistics
from pathlib import Path
from urllib.request import Request,urlopen
from control_client import Client
from precision_common import PROMPTS,parse_response,compare_ids,compare_teacher,ids_hash
from run_matrix import coverage
from system_snapshot import snapshot

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--mode',choices=('tokens','teacher','fidelity'),default='tokens');ap.add_argument('--profiles')
    ap.add_argument('--teacher-data',type=Path,help='Frozen JSON cases with body, prompt_token_ids and token_ids; may be held-out reference tokens')
    ap.add_argument('--full-vectors',action='store_true',help='Optional first 128 FP32 full-vocab vectors per case/profile; NOT a timing test')
    ap.add_argument('--historical',type=Path,help='Original GNmac baseline-live.json, decoded token strings only')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);client=Client(a.root);session=uuid.uuid4().hex
    defaults=dict(tokens='A,B-DENSE,B-WOA,B-EXPERTS,B-NO-DENSE,B-NO-WOA,B,B-LANE,C-FP8,C-FUSED,C,A',teacher='A,B-DENSE,B-WOA,B-EXPERTS,B-NO-DENSE,B-NO-WOA,B,B-LANE,A',fidelity='A,A-ORACLE,B,B-ORACLE,C-FP8,C-FUSED,C,B-LANE,B-LANE-ORACLE,B-ORACLE')
    profiles=(a.profiles or defaults[a.mode]).split(',')
    doc=dict(schema='star.abc.precision.v1',mode=a.mode,profiles=[],started=time.time(),diagnostic_not_speed=True,model_quality_verified=False)
    spec=json.loads((a.root/'launch.json').read_text())
    doc['boot']=dict(image=spec['image'],kv_cache_dtype=spec.get('kv_cache_dtype','fp8_ds_mla'),comparison_contract_sha256=spec.get('comparison_contract_sha256'),root=str(a.root),launch_sha256=__import__('hashlib').sha256((a.root/'launch.json').read_bytes()).hexdigest())
    def save():(a.out/'results.json').write_text(json.dumps(doc,indent=2,ensure_ascii=False))
    def post(body):
        start=time.perf_counter()
        with urlopen(Request(client.base_url+'/v1/chat/completions',json.dumps(body).encode(),client.headers(session)),timeout=900) as r:raw=json.loads(r.read())
        return dict(wall_s=time.perf_counter()-start,request=body,raw=raw,**parse_response(raw))
    def body(prompt,cap):return dict(model='deepseek-v4.1-flash-1017',messages=[dict(role='user',content=prompt)],temperature=0,top_p=1,max_tokens=cap,stream=False,logprobs=True,top_logprobs=5,return_token_ids=True,chat_template_kwargs={'thinking':False,'enable_thinking':False},cache_salt=uuid.uuid4().hex)
    client.call(dict(op='abc_begin',session=session))
    try:
        corpus=None
        if a.mode=='teacher':
            if a.teacher_data:corpus=json.loads(a.teacher_data.read_text())
            else:
                corpus=[]
                for label,prompt,cap in PROMPTS:
                    request=body(prompt,cap);got=post(request)
                    corpus.append(dict(label=label,body=request,prompt_token_ids=got['prompt_token_ids'],token_ids=got['token_ids'][:512],source='A greedy self-generated diagnostic continuation; NOT held-out quality data'))
            (a.out/'teacher-data.json').write_text(json.dumps(corpus,indent=2))
        for i,profile in enumerate(profiles):
            selection=client.call(dict(op='abc_select',session=session,profile=profile,costs=None))
            rec=dict(profile=profile,index=i,selection=selection,cases=[],system_before=snapshot());doc['profiles'].append(rec);save()
            cases=corpus if corpus is not None else [dict(label=l,body=body(p,c)) for l,p,c in PROMPTS]
            for case in cases:
                request=json.loads(json.dumps(case['body']));label=case['label'];before=client.call(dict(op='abc_status'))
                if a.mode=='fidelity':request['logprobs']=False;request.pop('top_logprobs',None)
                if a.mode=='teacher':
                    probe_label=session[:12]+'-'+label
                    request.update(max_tokens=len(case['token_ids']),ignore_eos=True,return_token_ids=True,stream=False,logprobs=True,top_logprobs=5,cache_salt=uuid.uuid4().hex)
                    request['vllm_xargs']=dict(abc_teacher_ids=case['token_ids'],abc_teacher_prompt_sha256=ids_hash(case['prompt_token_ids']),abc_probe_label=probe_label,abc_probe_full=a.full_vectors)
                result=post(request);result.update(label=label,coverage=coverage(before,client.call(dict(op='abc_status')),profile,1))
                if profile.endswith('-ORACLE') and not all(r['hits'].get('oracle_forced_rejection:'+profile,0)>0 for r in result['coverage']['rows']):raise RuntimeError('Forced-rejection path not observed on every rank')
                if a.mode=='teacher':
                    result['teacher_reference']=dict(source=case.get('source'),critical_field=case.get('critical_field'))
                    path=a.root/'round-logs'/('abc-teacher-'+probe_label+'.jsonl')
                    rows=[json.loads(line) for line in path.read_text().splitlines()]
                    rows=[r for r in rows if r['epoch']==selection['epoch'] and r['profile']==profile]
                    if len({r['index'] for r in rows})!=len(case['token_ids']) or len(rows)!=len(case['token_ids']):raise RuntimeError('Missing/duplicate teacher score rows')
                    if result['token_ids']!=case['token_ids']:raise RuntimeError('Teacher forcing failed to reproduce frozen numeric token IDs')
                    result['teacher']=rows
                rec['cases'].append(result);save()
            rec['system_after']=snapshot();save()
        controls={}
        for p in doc['profiles']:
            if p['profile']=='A':
                if not controls:controls={c['label']:c for c in p['cases']}
                else:doc['repeat_A']={c['label']:compare_ids(controls[c['label']],c) for c in p['cases']}
        if controls:doc['vs_A']={str(p['index'])+':'+p['profile']:{c['label']:compare_ids(controls[c['label']],c) for c in p['cases']} for p in doc['profiles']}
        if a.mode=='teacher':
            baseline=doc['profiles'][0];doc['teacher_comparisons']={}
            for p in doc['profiles'][1:]:
                groups={}
                for left,right in zip(baseline['cases'],p['cases']):
                    pairs=[compare_teacher(x,y,a.root/'round-logs') for x,y in zip(left['teacher'],right['teacher'])]
                    # Exclude the prefill-shaped first score. Require actual decode path coverage.
                    decode=[r for r in pairs if left['teacher'][r['index']]['scheduled_tokens'] in (4,6) and right['teacher'][r['index']]['scheduled_tokens'] in (4,6)]
                    if not decode or not right['coverage']['requested_B_observed_on_all_ranks']:raise RuntimeError('Actual small-row target bank coverage missing')
                    groups[left['label']]=dict(rows=pairs,decode_rows=len(decode),mean_decode_delta_nll=statistics.mean(r['delta_nll'] for r in decode),decode_winner_changes=sum(r['winner_changed'] for r in decode))
                doc['teacher_comparisons'][str(p['index'])+':'+p['profile']]=groups
        if a.mode=='fidelity':
            by={p['profile']:p for p in doc['profiles']};doc['verifier_comparisons']={}
            for spec,oracle in [('A','A-ORACLE'),('B','B-ORACLE'),('C-FP8','B-ORACLE'),('C-FUSED','B-ORACLE'),('C','B-ORACLE'),('B-LANE','B-LANE-ORACLE'),('C-GATE','B-ORACLE'),('C-ALL','B-ORACLE'),('B-LANE-GATE','B-LANE-ORACLE')]:
                if spec in by and oracle in by:
                    doc['verifier_comparisons'][spec+' vs '+oracle]={x['label']:compare_ids(x,y) for x,y in zip(by[spec]['cases'],by[oracle]['cases'])}
        if a.historical:
            old={r['label']:r for r in json.loads(a.historical.read_text())['rows']}
            doc['historical_string_comparison']={str(p['index'])+':'+p['profile']:{c['label']:dict(decoded_token_strings_equal=c['token_strings']==old[c['label']]['tokens'],text_equal=c['content']==old[c['label']]['content'],limitation='Historical file has decoded strings, not numeric IDs or pinned full runtime; not a same-prefix numeric proof') for c in p['cases']} for p in doc['profiles']}
        doc['finished']=time.time();save()
    finally:
        try:doc['end']=client.call(dict(op='abc_end',session=session))
        except Exception as exc:doc['end_error']=repr(exc)
        save()
    print(json.dumps(dict(output=str(a.out),mode=a.mode,model_quality_verified=False)))

if __name__=='__main__':main()
