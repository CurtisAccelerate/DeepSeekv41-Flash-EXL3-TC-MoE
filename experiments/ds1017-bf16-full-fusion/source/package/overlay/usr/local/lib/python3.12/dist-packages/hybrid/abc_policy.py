"""Scheduler-side width control. No donor cost constants; defaults remain fixed five."""
import json,os,math
from pathlib import Path
from .abc_contract import validate_costs

_CACHE=None
_STAMP=None
_REACH={}
_ACC={}
_LAST=5
def policy():
    global _CACHE,_STAMP,_LAST
    p=Path(os.environ.get('W9_RESIDENT_API_CONTROL','/w9/state'))/'abc-policy.json'
    try:stamp=(p.stat().st_mtime_ns,p.stat().st_size)
    except FileNotFoundError:return None
    if stamp!=_STAMP:
        obj=json.loads(p.read_text())
        if obj.get('schema')!='star.abc.policy.v1':raise RuntimeError('Bad ABC scheduler policy schema')
        if obj.get('armed') and obj.get('width')=='auto':validate_costs(obj.get('costs'),obj.get('identity'))
        if obj.get('width') not in (3,5,'auto'):raise RuntimeError('Bad ABC width')
        if _CACHE is None or (obj.get('identity'),obj.get('epoch'))!=(_CACHE.get('identity'),_CACHE.get('epoch')):
            _REACH.clear();_ACC.clear();_LAST=5
        _CACHE=obj;_STAMP=stamp
    return _CACHE if _CACHE.get('armed') else None
def enabled():return policy() is not None
def pad_k(sched):
    p=policy()
    if p is None:return sched.num_spec_tokens
    if p['width'] in (3,5):return p['width']
    return _LAST
def expected(r,k):
    reach=_REACH.get(r,[0.]*5);acc=_ACC.get(r,[0.]*5);run=1.;out=1.
    for i in range(k):
        c=(acc[i]+4*.75)/(reach[i]+4);run*=c;out+=run
    return out
def choose(sched,scheduled):
    global _LAST
    p=policy()
    if p is None:return sched.num_spec_tokens
    if p['width'] in (3,5):return p['width']
    b=len(scheduled)
    if not 1<=b<=8:return 5
    costs=validate_costs(p['costs'],p['identity'])[str(b)]
    score={k:sum(expected(r,k) for r in scheduled)/costs[str(k)] for k in (3,5)}
    best=max(score,key=lambda k:(score[k],k))
    if score[best]<score.get(_LAST,0)*1.03:best=_LAST
    _LAST=best;return best
def observe(sched,r,k,accepted):
    if not enabled():return
    reach=_REACH.setdefault(r,[0.]*5);acc=_ACC.setdefault(r,[0.]*5)
    for i in range(min(k,5)):
        if i<=accepted:reach[i]=reach[i]*.98+1;acc[i]=acc[i]*.98+int(i<accepted)
def forget(r):_REACH.pop(r,None);_ACC.pop(r,None)

