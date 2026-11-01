"""One drained test session; finite precaptured profiles, all-rank atomic selection."""
import json,os,uuid
from pathlib import Path
from .abc_contract import PROFILES,validate_costs,validate_routes,digest

def worker_invoke(worker,method,*args):
    import torch
    bank=getattr(worker,'_abc_bank',None)
    if bank is None or bank.selection is None:raise ValueError('ABC was not prepared at startup')
    if method=='status':return bank.status()
    selection=bank.selection
    if method=='route':
        if selection.selected!='A' or selection.session is None or selection.pending is not None:raise ValueError('Route update requires armed A and no transaction')
        bank.routes=json.loads(json.dumps(validate_routes(args[0],selection.identity)))
        return dict(rank=int(worker.rank),ok=True,routes_sha256=digest(bank.routes))
    if method=='prepare':
        request=args[0]
        if set(request)!={'token','profile','session','identity','epoch'}:raise ValueError('Unexpected ABC transaction fields')
        worker._w26_attention_adapter.current_identity()
        selection.prepare(**request)
    elif method=='commit':
        torch.cuda.synchronize();selection.commit(args[0])
        from . import abc_gate
        abc_gate.reset()
    elif method=='rollback':selection.rollback(args[0])
    elif method=='end':
        torch.cuda.synchronize();selection.end()
        from . import abc_gate
        abc_gate.reset()
    else:raise ValueError('Unknown ABC worker operation')
    return dict(rank=int(worker.rank),ok=True,**selection.record())

class Owner:
    def __init__(self,control):
        self.c=control;self.session=None;self.identity=None;self.epoch=0
        self.c.root.mkdir(parents=True,exist_ok=True)
        self.policy('A')  # A restart cannot inherit an armed policy from an old test session.
    @staticmethod
    def validate(q):
        allowed={'abc_status':{'op'},'abc_begin':{'op','session'},'abc_select':{'op','session','profile','costs'},'abc_end':{'op','session'},'abc_route':{'op','session','routing'}}
        if not isinstance(q,dict) or q.get('op') not in allowed or set(q)!=allowed[q['op']]:raise ValueError('Invalid ABC envelope')
        if q['op']!='abc_status' and (not isinstance(q['session'],str) or not q['session']):raise ValueError('Missing test session')
        if q['op']=='abc_select':
            if q['profile'] not in PROFILES:raise ValueError('Unknown profile')
            if q['costs'] is not None and not isinstance(q['costs'],dict):raise ValueError('Costs must be an object or null')
            if PROFILES[q['profile']].width!='auto' and q['costs'] is not None:raise ValueError('Costs only apply to auto verification')
    async def rows(self,method,args=()):
        rows=await self.c.rpc('wabc_'+method,args,timeout=120)
        if len(rows)!=4 or sorted(r.get('rank') for r in rows)!=list(range(4)) or not all(r.get('ok') is True for r in rows):raise RuntimeError('Missing/failed ABC rank receipt')
        return rows
    async def close(self):
        g=self.c.admission
        g.open=False;g.attention_guarded=True;g.attention_session=None
        await g.wait_drained();await self.c.pause();await self.c.qualify()
    def policy(self,profile,costs=None):
        p=PROFILES[profile];record=dict(schema='star.abc.policy.v1',armed=self.session is not None,session=self.session,identity=self.identity,epoch=self.epoch,profile=profile,width=p.width,costs=costs)
        self.c.save('abc-policy.json',record)
    async def select(self,name,costs=None):
        p=PROFILES[name]
        if p.width=='auto':validate_costs(costs,self.identity)
        elif costs is not None:raise ValueError('Costs only apply to auto width')
        await self.close()
        token=uuid.uuid4().hex;request=dict(token=token,profile=name,session=self.session,identity=self.identity,epoch=self.epoch+1)
        prepared=await self.rows('prepare',(request,))
        if any(r.get('pending')!=(token,name,self.session,self.epoch+1) and r.get('pending')!=[token,name,self.session,self.epoch+1] for r in prepared):raise RuntimeError('ABC prepare identity mismatch')
        committed=await self.rows('commit',(token,))
        if not all((r['selected'],r['session'],r['identity'],r['epoch'])==(name,self.session,self.identity,self.epoch+1) for r in committed):raise RuntimeError('ABC commit mismatch')
        self.epoch+=1;self.policy(name,costs)
        # KV produced by another target arithmetic path must never contaminate the comparison.
        reset=await self.c.engine.reset_prefix_cache(reset_running_requests=False)
        if reset is not True:raise RuntimeError('ABC prefix-cache reset did not succeed')
        self.c.admission.abc_narrow=p.width!=5 or p.oracle
        self.c.admission.attention_session=self.session
        await self.c.engine.resume_generation()
        if await self.c.engine.is_paused():raise RuntimeError('ABC resume was not confirmed')
        self.c.phase='benchmark_serving';self.c.persist()
        return dict(profile=name,identity=self.identity,epoch=self.epoch,rows=committed,session=self.session,ordinary_admission=False,prefix_cache_reset=True,requires_headers=['X-W9-Benchmark-Token','X-W26-Benchmark-Session'],model_quality_verified=False)
    async def handle_locked(self,q):
        self.validate(q)
        if q['op']=='abc_status':return dict(abc=await self.rows('status'),session=self.session)
        # Existing public production selection is never silently disabled or re-proven.
        g=self.c.admission
        if g.production_active or g.production_transition:raise ValueError('ABC begins in the dedicated test launch, before public production enable')
        if q['op']=='abc_begin':
            if self.session is not None or self.c.attention_bank.session is not None:raise ValueError('A bank session is already active')
            status=await self.rows('status')
            identities={r['identity'] for r in status}
            if len(identities)!=1 or not all('A' in r['available'] and r['pending'] is None and r['session'] is None for r in status):raise ValueError('ABC reference readiness differs')
            self.identity=next(iter(identities));self.epoch=status[0]['epoch'];self.session=q['session']
            try:return await self.select('A')
            except BaseException as exc:
                await self.c.quarantine('ABC begin failed: '+repr(exc),poison=True);raise
        if self.session is None or q['session']!=self.session:raise ValueError('Wrong ABC session')
        if q['op']=='abc_route':
            route=validate_routes(q['routing'],self.identity)
            status=await self.rows('status')
            if not all(r['selected']=='A' for r in status):raise ValueError('Select A before changing per-B routes')
            await self.close()
            try:
                rows=await self.rows('route',(route,))
                if not all(r.get('routes_sha256')==digest(route) for r in rows):raise RuntimeError('Route receipt mismatch')
                return await self.select('A')
            except BaseException as exc:
                await self.c.quarantine('ABC route update failed: '+repr(exc),poison=True);raise
        if q['op']=='abc_select':
            status=await self.rows('status')
            if not all(q['profile'] in r['available'] for r in status):raise ValueError('Profile is not ready on every rank')
            if PROFILES[q['profile']].experts=='routed' and not all(r.get('routes') is not None for r in status):raise ValueError('Measured per-B routing has not been installed')
            if PROFILES[q['profile']].width=='auto':validate_costs(q['costs'],self.identity)
            try:return await self.select(q['profile'],q['costs'])
            except BaseException as exc:
                # Partial/unknown outcomes never resume a guessed baseline.
                await self.c.quarantine('ABC transaction failed: '+repr(exc),poison=True);raise
        await self.close()
        try:
            rows=await self.rows('end')
            if not all(r['selected']=='A' and r['session'] is None for r in rows):raise RuntimeError('ABC end disagrees')
        except BaseException as exc:
            await self.c.quarantine('ABC end failed: '+repr(exc),poison=True);raise
        self.session=None;self.policy('A');g.attention_guarded=False;g.attention_session=None;g.abc_narrow=False
        self.c.phase='paused';self.c.persist()
        return dict(ended=True,reference_retained=True,scheduler_paused=True,public_admission=False,rows=rows)

