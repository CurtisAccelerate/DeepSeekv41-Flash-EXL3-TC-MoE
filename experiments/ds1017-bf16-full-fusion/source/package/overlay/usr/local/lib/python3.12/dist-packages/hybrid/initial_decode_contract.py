"""Initial-only causal metadata guard. No serving activation or CUDA operations."""
from contextlib import contextmanager
from .attention_qualification import QualificationRejected

ACTIVE=None
class CompletedMetadataRejection(QualificationRejected):pass

@contextmanager
def initial_decode(runner,consensus,records,b=1,query_len=6):  # STAR_ABC integration
    global ACTIVE
    if ACTIVE is not None:raise RuntimeError('Nested initial metadata context')
    if query_len not in (4,6):raise RuntimeError("Unsupported initial query length")
    state=dict(runner=runner,consensus=consensus,records=records,calls=0,b=b,query_len=query_len)
    ACTIVE=state
    try:
        yield
        if state['calls']!=1:raise RuntimeError('Initial metadata guard call count differs')
    finally:ACTIVE=None

def validate_common(runner,cm,*,num_tokens,num_reqs,num_tokens_padded,num_reqs_padded,ubatch_slices):
    """Called once after cm_base construction, before ANY builder runs.

    Outside private initial target dummy this is an immediate no-op. Every rank
    votes before entering builder/model operations; communication tickets are
    advanced normally and are never journal-restored.
    """
    state=ACTIVE
    if state is None:return
    error=None;row={}
    try:
        if runner is not state['runner']:raise ValueError('Initial runner identity differs')
        row=dict(num_tokens=num_tokens,num_reqs=num_reqs,num_tokens_padded=num_tokens_padded,
            num_reqs_padded=num_reqs_padded,causal=cm.causal,max_query_len=cm.max_query_len,
            actual_tokens=cm.num_actual_tokens,common_reqs=cm.num_reqs,
            query_start=cm.query_start_loc_cpu.tolist(),mm_ranges_empty=not bool(cm.mm_req_doc_ranges),
            no_ubatches=ubatch_slices is None,never_served=getattr(runner,'_w26_serving_seen',True) is False,
            empty_requests=runner.input_batch.num_reqs==0 and len(runner.requests)==0)
        b=state['b']
        q=state['query_len']
        expected=dict(num_tokens=q*b,num_reqs=b,num_tokens_padded=q*b,num_reqs_padded=b,
            causal=True,max_query_len=q,actual_tokens=q*b,common_reqs=b,query_start=list(range(0,q*b+1,q)),
            mm_ranges_empty=True,no_ubatches=True,never_served=True,empty_requests=True)
        if row!=expected:raise ValueError('Private initial target must be causal T6/B1, without prefill or multimodal ranges')
        if state['calls']!=0:raise ValueError('Repeated metadata construction in target-only dummy')
    except Exception as exc:error=repr(exc)
    state['calls']+=1
    vote=dict(phase='w26.initial-target-metadata.v1',ok=error is None,metadata=row,error=error,call=state['calls'])
    votes=state['consensus'](vote)
    state['records'].append(votes)
    if len(votes)!=4 or not all(isinstance(v,dict) and v.get('phase')==vote['phase'] and v.get('call')==state['calls'] for v in votes):
        raise RuntimeError('Metadata collective phase/schema mismatch: '+str(votes))
    if not all(v.get('ok') is True and v.get('call')==1 and v.get('metadata')==row for v in votes):
        raise CompletedMetadataRejection('All-rank initial causal metadata guard rejected: '+str(votes))
