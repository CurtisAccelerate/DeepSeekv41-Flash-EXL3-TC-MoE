"""GNmac acceptance controller integrated with our private scheduler policy.
GNmac-lab MIT; adaptive-depth foundation Morrowmake Apache-2.0.
Offline candidate only. Local measured costs required; defaults remain fixed five."""
import json,os,math,time
from pathlib import Path
from .abc_contract import validate_costs,validate_policy_costs

_CACHE=None
_STAMP=None
_REACH={}
_ACC={}
_LAST=None
_EVENT_SINK=None

def event(kind,**fields):
    # Optional CPU diagnostic using the existing bounded background sink.
    # No device synchronization; disabled by default.
    global _EVENT_SINK
    if os.environ.get('STAR_ABC_POLICY_OBSERVE','0')!='1' or not _CACHE or not _CACHE.get('armed'):return
    if _EVENT_SINK is None:
        from .dsv41_r1_trace import JsonlSink
        root=Path(os.environ.get('W9_RESIDENT_API_CONTROL','/w9/state'))
        _EVENT_SINK=JsonlSink(str(root/('gnmac-policy-events-pid'+str(os.getpid())+'.jsonl')))
    _EVENT_SINK.put(dict(schema='star.gnmac.policy-event.v1',type=kind,t_mono_ns=time.monotonic_ns(),
                         profile=_CACHE['profile'],identity=_CACHE['identity'],epoch=_CACHE['epoch'],session=_CACHE.get('session'),**fields))

def policy():
    global _CACHE,_STAMP,_LAST
    p=Path(os.environ.get('W9_RESIDENT_API_CONTROL','/w9/state'))/'abc-policy.json'
    try:stamp=(p.stat().st_mtime_ns,p.stat().st_size)
    except FileNotFoundError:return None
    if stamp!=_STAMP:
        obj=json.loads(p.read_text())
        if obj.get('schema')!='star.abc.policy.v1':raise RuntimeError('Bad ABC scheduler policy schema')
        if obj.get('armed') and obj.get('width')=='auto':validate_policy_costs(obj.get('costs'),obj.get('identity'),obj.get('profile'))
        if obj.get('width') not in (3,5,'auto'):raise RuntimeError('Bad ABC width')
        if _CACHE is None or (obj.get('identity'),obj.get('epoch'))!=(_CACHE.get('identity'),_CACHE.get('epoch')):
            _REACH.clear();_ACC.clear();_LAST=None
        _CACHE=obj;_STAMP=stamp
    return _CACHE if _CACHE.get('armed') else None
def enabled():return policy() is not None
def pad_k(sched):
    # AsyncScheduler publishes the chosen NEXT width to this placeholder list
    # after each schedule. New decode padding must match running requests,
    # including across controller epoch changes; use the actual running width.
    p=policy()
    if p is None:return sched.num_spec_tokens
    return len(getattr(sched,'_spec_token_placeholders',())) or sched.num_spec_tokens

def expected(r,k):
    reach,acc=(_REACH.get(r),_ACC.get(r))
    e,run,c=(1.0,1.0,0.9)
    for j in range(k):
        if reach is not None:c=(acc[j]+2.0*c)/(reach[j]+2.0)
        run*=c;e+=run
    return e

def choose(sched,scheduled):
    global _LAST
    p=policy()
    if p is None:return sched.num_spec_tokens
    if p['width'] in (3,5):
        event('choose',mode='fixed',users=len(scheduled),width=p['width'],request_ids=list(scheduled))
        return p['width']
    b=len(scheduled)
    if not 1<=b<=8:return sched.num_spec_tokens
    row=validate_costs(p['costs'],p['identity'])[str(b)]
    candidates=[k for k in (3,5) if k<=sched.num_spec_tokens and str(k) in row]
    if not candidates:return sched.num_spec_tokens
    expected_tokens={k:sum(expected(r,k) for r in scheduled) for k in candidates}
    score={k:expected_tokens[k]/row[str(k)] for k in candidates}
    best=max(score,key=lambda k:(score[k],k))
    if _LAST in score and _LAST!=best and score[best]<=score[_LAST]*1.03:best=_LAST
    _LAST=best
    event('choose',mode='auto',users=b,width=best,request_ids=list(scheduled),expected_tokens=expected_tokens,costs_ms={k:row[str(k)] for k in candidates})
    return best

def observe(sched,r,k,accepted):
    if k<=0 or not enabled():return
    a=min(max(accepted,0),k)
    event('observe',request_id=r,verified=k,accepted=a,accepted_raw=accepted)
    reach=_REACH.setdefault(r,[0.]*5);acc=_ACC.setdefault(r,[0.]*5)
    for j in range(5):reach[j]*=.9;acc[j]*=.9
    for j in range(min(a+1,k,5)):
        reach[j]+=1.
        if j<a:acc[j]+=1.

def forget(r):
    event('forget',request_id=r)
    _REACH.pop(r,None);_ACC.pop(r,None)

