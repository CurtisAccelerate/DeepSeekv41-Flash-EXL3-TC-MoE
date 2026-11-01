"""Post-load populated-context retrieval/decode coverage and ingestion wall check."""
import argparse,json,time,uuid
from pathlib import Path
from urllib.request import Request,urlopen
from control_client import Client
from stream_wall_probe import measure
from run_matrix import coverage
from system_snapshot import snapshot
import bench_exact as exact

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--profiles',default='A,B,C,A');ap.add_argument('--contexts',default='16384,65536,131072');a=ap.parse_args()
    contexts=list(map(int,a.contexts.split(',')))
    if not contexts or any(c<1024 or c>261888 for c in contexts):ap.error('Use 1024..261888 prompt tokens; reserve generation space')
    a.out.mkdir(parents=True,exist_ok=False);client=Client(a.root);session=uuid.uuid4().hex
    exact.BASE=client.base_url;exact.MODEL='deepseek-v4.1-flash-1017';exact.HEADERS=client.headers(session)
    doc=dict(schema='star.abc.capacity.v1',profiles=[],model_quality_verified=False,limitation='One request per context, not eight simultaneous maximum-context requests or a broad long-context evaluation')
    def save():(a.out/'results.json').write_text(json.dumps(doc,indent=2))
    client.call(dict(op='abc_begin',session=session))
    try:
        prompts=[]
        for target in contexts:
            key=uuid.uuid4().hex;needle='Archived source record: validation_code='+key+'.\n'
            lo,hi=1,target
            best=None
            for _ in range(22):
                words=(lo+hi)//2
                text=needle+exact.filler(words,target)+'\nQuestion: Return only JSON with validation_code copied exactly from the archived source record.'
                count=exact.token_count(text)
                if best is None or abs(count-target)<abs(best[0]-target):best=(count,text)
                if abs(count-target)<=16:break
                if count<target:lo=words+1
                else:hi=words-1
            if abs(best[0]-target)>32:raise ValueError('Context calibration failed')
            prompts.append(dict(target=target,prompt_tokens=best[0],prompt=best[1],answer=key))
        for profile in a.profiles.split(','):
            selection=client.call(dict(op='abc_select',session=session,profile=profile,costs=None));rec=dict(profile=profile,selection=selection,system=snapshot(),cells=[]);doc['profiles'].append(rec)
            for case in prompts:
                body=dict(model=exact.MODEL,messages=[dict(role='user',content=case['prompt'])],temperature=0,top_p=1,max_tokens=256,stream=True,stream_options={'include_usage':True},chat_template_kwargs=exact.KW,cache_salt=uuid.uuid4().hex)
                before=client.call(dict(op='abc_status'));response=measure(exact.BASE+'/v1/chat/completions',body,profile+':'+str(case['target']),extra_headers=client.headers(session));after=client.call(dict(op='abc_status'))
                answer=None
                try:answer=json.loads(response['content'])
                except ValueError:pass
                rec['cells'].append(dict(target_prompt_tokens=case['target'],calibrated_tokens=case['prompt_tokens'],response=response,coverage=coverage(before,after,profile,1),retrieval_pass=answer=={'validation_code':case['answer']},completed_without_cap=response['transport_complete'] and bool(response['finish_reasons']) and 'length' not in response['finish_reasons']));save()
    finally:
        try:doc['end']=client.call(dict(op='abc_end',session=session))
        except Exception as exc:doc['end_error']=repr(exc)
        save()
    print(json.dumps(dict(output=str(a.out),model_quality_verified=False)))

if __name__=='__main__':main()
