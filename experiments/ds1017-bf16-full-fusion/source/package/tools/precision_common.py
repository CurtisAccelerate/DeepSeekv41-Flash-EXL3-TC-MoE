"""Token identity and paired teacher statistics. No model calls on import."""
import json,math,hashlib
PROMPTS=[
 ('count','Count from 1 to 40. Output only the numbers, separated by spaces. No other text.',80),
 ('code','def binary_search(nums, target) -> int: index of target in a sorted list, or -1.\nOutput only Python source. No comments, no docstrings, no markdown fences.',160),
 ('prose','Explain a hash map collision in two sentences.',80),
 ('tool','Return one JSON object with keys ok and n. ok is true. n is 3. No other text.',40),
]
def ids_hash(ids):return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()
def parse_response(obj):
    ch=obj['choices'][0];pieces=(ch.get('logprobs') or {}).get('content') or []
    ids=ch.get('token_ids');prompt=obj.get('prompt_token_ids')
    if not isinstance(ids,list) or not ids or any(type(v) is not int or not 0<=v<129280 for v in ids):raise ValueError('Server did not return numeric completion token IDs')
    if not isinstance(prompt,list) or not prompt or any(type(v) is not int for v in prompt):raise ValueError('Server did not return frozen prompt token IDs')
    return dict(token_ids=ids,prompt_token_ids=prompt,prompt_sha256=ids_hash(prompt),token_strings=[p.get('token') for p in pieces],content=ch['message'].get('content'),finish=ch.get('finish_reason'),logprobs=pieces)
def compare_ids(a,b):
    if a.get('prompt_sha256')!=b.get('prompt_sha256'):raise ValueError('Prompt token identities differ')
    x,y=a['token_ids'],b['token_ids'];first=next((i for i,(v,w) in enumerate(zip(x,y)) if v!=w),None)
    if first is None and len(x)!=len(y):first=min(len(x),len(y))
    return dict(exact=x==y,first_difference=first,left_length=len(x),right_length=len(y),
        at_first=None if first is None else dict(left=x[first] if first<len(x) else None,right=y[first] if first<len(y) else None),
        note='Only the shared prefix through the first difference supports a causal token-choice comparison; later positions can follow different contexts')
def compare_teacher(a,b,root,other_root=None):
    if any(a[k]!=b[k] for k in ('prompt_sha256','teacher_sha256','index','input_position','teacher_token_id')):raise ValueError('Unmatched teacher prefix')
    r=dict(index=a['index'],delta_nll=b['nll']-a['nll'],winner_changed=a['winner']!=b['winner'],left_winner=a['winner'],right_winner=b['winner'],left_margin=a['margin'],right_margin=b['margin'],graph_match=a['graph']==b['graph'],left_graph=a['graph'],right_graph=b['graph'])
    if a.get('full_vector') and b.get('full_vector'):
        import numpy as np
        x=np.load(root/a['full_vector'],allow_pickle=False).astype('float64');y=np.load((other_root or root)/b['full_vector'],allow_pickle=False).astype('float64')
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid logit vectors')
        def logsoftmax(v):v=v-v.max();return v-np.log(np.exp(v).sum())
        p,q=logsoftmax(x),logsoftmax(y);delta=float(np.max(np.abs(p-q)))
        r.update(kl_left_to_right=float(np.sum(np.exp(p)*(p-q))),max_logprob_delta=delta,max_logit_delta=float(np.max(np.abs(x-y))),
                 argmax_stability_certificate_for_this_pair=bool(a['margin']>2*delta),certificate_scope='Only this exact fixed teacher prefix and these two measured full vectors')
    return r
