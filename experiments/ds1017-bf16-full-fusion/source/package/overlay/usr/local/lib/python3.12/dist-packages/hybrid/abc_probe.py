"""Bounded teacher forcing on real decode graphs. DIAGNOSTIC, never a speed run.

Records unmodified target logits, then forces the frozen target/draft token
sequence through ordinary engine scheduling/KV/rejection bookkeeping.
"""
import hashlib,json,re
from pathlib import Path

def ids_hash(ids):return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()

def validate(extra,prompt_ids):
    if not isinstance(extra,dict) or 'abc_teacher_ids' not in extra:return None
    ids=extra['abc_teacher_ids'];label=extra.get('abc_probe_label')
    if not isinstance(ids,list) or not 1<=len(ids)<=512 or any(type(i) is not int or not 0<=i<129280 for i in ids):raise ValueError('Teacher IDs must be 1..512 real vocabulary integers')
    if not isinstance(label,str) or re.fullmatch('[A-Za-z0-9_-]{1,64}',label) is None:raise ValueError('Invalid bounded probe label')
    if extra.get('abc_teacher_prompt_sha256')!=ids_hash(prompt_ids):raise ValueError('Teacher prompt tokens differ from the frozen reference')
    # The OpenAI request's vllm_xargs numeric union transports JSON booleans as
    # integer0/1. Accept only that lossless representation and canonicalize it.
    full=extra.get('abc_probe_full',False)
    if type(full) not in (bool,int) or full not in (0,1):raise ValueError('abc_probe_full must be bool or transport integer0/1')
    return dict(ids=ids,label=label,prompt_length=len(prompt_ids),prompt_sha256=ids_hash(prompt_ids),teacher_sha256=ids_hash(ids),full=bool(full))

def expected(plan,input_position):
    i=int(input_position)+1-plan['prompt_length']
    return i,plan['ids'][i] if 0<=i<len(plan['ids']) else 0

def before(runner,so):
    from . import abc_runtime as R
    runner._abc_teacher_plans={};runner._abc_probe_desc=None
    profile=R.current()
    if profile is None:return
    plans={}
    for rid in runner.input_batch.req_ids[:runner.input_batch.num_reqs]:
        state=runner.requests[rid];sp=state.sampling_params
        extra=getattr(sp,'extra_args',None)
        if not isinstance(extra,dict) or 'abc_teacher_ids' not in extra:continue
        if profile.gate or profile.oracle:raise ValueError('Teacher diagnostic requires ungated ordinary verification')
        from .dsv41_r1 import p_is_approx
        if float(sp.temperature)!=0 or p_is_approx(sp) or getattr(sp,'structured_outputs',None) is not None:raise ValueError('Teacher diagnostic requires plain greedy, unprocessed target logits')
        plan=validate(extra,state.prompt_token_ids)
        if sp.max_tokens!=len(plan['ids']) or not sp.ignore_eos:raise ValueError('Teacher diagnostic requires max_tokens=len(script), ignore_eos=true')
        plans[rid]=plan
    if plans and len(plans)!=runner.input_batch.num_reqs:raise ValueError('Teacher and ordinary traffic cannot share a diagnostic batch')
    runner._abc_teacher_plans=plans

def observe(runner,so,logits,indices,metadata,desc):
    plans=getattr(runner,'_abc_teacher_plans',None)
    if not plans:return
    import torch
    from . import abc_runtime as R
    if logits is None or hasattr(logits,'_abc_ids') or logits.ndim!=2 or logits.shape[1]!=129280:raise RuntimeError('Teacher probe did not receive full real target logits')
    if not runner.input_batch.sampling_metadata.all_greedy:raise RuntimeError('Sampled teacher batch refused')
    counts=[1]*len(plans) if metadata is None else [n+1 for n in metadata.num_draft_tokens]
    reqs=runner.input_batch.req_ids[:runner.input_batch.num_reqs]
    if len(reqs)!=len(counts) or sum(counts)!=logits.shape[0]:raise RuntimeError('Teacher logit-to-request map differs')
    positions=runner._get_positions(indices)
    if positions.ndim!=1:raise RuntimeError('Only text scalar positions are qualified for teacher diagnostics')
    positions=positions.long().cpu().tolist();inputs=runner.input_ids.gpu[indices].long().cpu().tolist()
    if len(positions)!=sum(counts):raise RuntimeError('Teacher index geometry differs')
    forced=[];row=0;records=[];paths=[]
    vals=logits.float()
    if not bool(torch.isfinite(vals).all()):raise RuntimeError('Nonfinite target logits in teacher diagnostic')
    normalizer=torch.logsumexp(vals,dim=-1);top,topids=torch.topk(vals,5,dim=-1)
    winners=torch.argmax(vals,dim=-1);tags=R.BANK.selection.record()
    for rid,count in zip(reqs,counts):
        plan=plans[rid]
        for _ in range(count):
            pos=positions[row];index,token=expected(plan,pos);forced.append(token)
            input_index=pos-plan['prompt_length']
            if input_index>=0:
                should=plan['ids'][input_index] if input_index<len(plan['ids']) else 0
                if inputs[row]!=should:raise RuntimeError('Teacher prefix inputs do not match the frozen script')
            if 0<=index<len(plan['ids']):
                tops=top[row].tolist();tids=topids[row].tolist();ln=float(normalizer[row])
                record=dict(schema='star.abc.teacher.v1',label=plan['label'],request_id=rid,identity=tags['identity'],profile=tags['selected'],epoch=tags['epoch'],
                    prompt_sha256=plan['prompt_sha256'],teacher_sha256=plan['teacher_sha256'],index=index,input_position=pos,input_token_id=inputs[row],teacher_token_id=token,
                    graph=[desc.num_tokens,desc.num_reqs,desc.uniform,desc.has_lora,desc.num_active_loras],actual_B=len(reqs),scheduled_tokens=so.num_scheduled_tokens[rid],
                    logits_dtype=str(logits.dtype),nll=float(normalizer[row]-vals[row,token]),winner=int(winners[row]),margin=tops[0]-tops[1],tie=tops[0]==tops[1],
                    top=[dict(id=i,logit=v,logprob=v-ln) for i,v in zip(tids,tops)],diagnostic_forcing=True,speed_measurement=False)
                if plan['full'] and index<128 and int(R.BANK.worker.rank)==0:
                    import numpy as np
                    name='abc-vector-'+plan['label']+'-'+tags['selected']+'-'+str(tags['epoch'])+'-'+str(index)+'.npy'
                    path=Path('/r1out')/name
                    if path.exists():raise RuntimeError('Refusing to overwrite a frozen diagnostic vector')
                    np.save(path,vals[row].detach().cpu().numpy(),allow_pickle=False);record['full_vector']=name
                records.append(record);paths.append(Path('/r1out')/('abc-teacher-'+plan['label']+'.jsonl'))
            row+=1
    if int(R.BANK.worker.rank)==0:
        for path,record in zip(paths,records):
            with path.open('a') as f:f.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
    # The unmodified scores are durably recorded above. Forcing is solely to
    # keep all arms on the identical teacher prefix; these are not quality replies.
    logits.fill_(-float('inf'))
    logits.scatter_(1,torch.tensor(forced,device=logits.device,dtype=torch.long).view(-1,1),0.)
    R.BANK.hits['teacher_target_rows']+=len(records)

def drafts(p,b,hidden):
    runner=getattr(p,'_dsv41_runner',None);plans=getattr(runner,'_abc_teacher_plans',None)
    if not plans:return None
    import torch
    from . import abc_runtime as R
    if b!=len(plans):raise RuntimeError('Teacher draft batch changed')
    positions=p.positions[p._anchor_idx[:b]].long().cpu().tolist();result=[]
    for rid,pos in zip(runner.input_batch.req_ids[:b],positions):
        plan=plans[rid];result.append([expected(plan,pos+j)[1] for j in range(5)])
    p._last_draft_probs=None;R.BANK.hits['teacher_draft_rows']+=b
    return torch.tensor(result,dtype=torch.long,device=hidden.device)
