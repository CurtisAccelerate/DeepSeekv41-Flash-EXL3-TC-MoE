"""Finite offline/worker contract. No torch imports or GPU side effects."""
from dataclasses import dataclass, asdict
import hashlib, json, math

SCHEMA = 'star.abc.v1'
@dataclass(frozen=True)
class Profile:
    name: str
    dense: bool = False
    woa: bool = False
    experts: str = 'ours'
    fp8_draft: bool = False
    fused_markov: bool = False
    gate: bool = False
    width: object = 5
    oracle: bool = False
    def family(self): return (self.dense, self.woa, self.experts)
    @property
    def reference(self):return self.family()==(False,False,'ours')

PROFILES = {p.name:p for p in (
    Profile('GN-BALANCED-WIDE',True,False,'hybrid',True,True),
    Profile('GN-BALANCED-NARROW',True,False,'hybrid',True,True,width=3),
    Profile('GN-BALANCED-AUTO',True,False,'hybrid',True,True,width='auto'),
    Profile('GN-BALANCED-GATE',True,False,'hybrid',True,True,gate=True),
    Profile('GN-BALANCED-NARROW-GATE',True,False,'hybrid',True,True,gate=True,width=3),
    Profile('GN-BALANCED-AUTO-GATE',True,False,'hybrid',True,True,gate=True,width='auto'),
    Profile('GN-BALANCED-REFERENCE-DRAFT',True,False,'hybrid'),
    Profile('GN-BALANCED-WIDE-ORACLE',True,False,'hybrid',True,True,oracle=True),
    Profile('GN-BALANCED-NARROW-ORACLE',True,False,'hybrid',True,True,width=3,oracle=True),
    Profile('GN-SPEED-WIDE',True,True,'hybrid',True,True),
    Profile('GN-SPEED-NARROW',True,True,'hybrid',True,True,width=3),
    Profile('GN-SPEED-AUTO',True,True,'hybrid',True,True,width='auto'),
    Profile('GN-SPEED-GATE',True,True,'hybrid',True,True,gate=True),
    Profile('GN-SPEED-NARROW-GATE',True,True,'hybrid',True,True,gate=True,width=3),
    Profile('GN-SPEED-AUTO-GATE',True,True,'hybrid',True,True,gate=True,width='auto'),
    Profile('GN-SPEED-REFERENCE-DRAFT',True,True,'hybrid'),
    Profile('GN-SPEED-WIDE-ORACLE',True,True,'hybrid',True,True,oracle=True),
    Profile('GN-SPEED-NARROW-ORACLE',True,True,'hybrid',True,True,width=3,oracle=True),
    Profile('A'),
    Profile('B', True, True, 'hybrid'),
    Profile('C', True, True, 'hybrid', True, True),
    Profile('B-DENSE', True), Profile('B-WOA', woa=True),
    Profile('B-EXPERTS', experts='hybrid'), Profile('B-LANE', True, True, 'lane'),
    Profile('C-FP8', True, True, 'hybrid', True),
    Profile('C-FUSED', True, True, 'hybrid', fused_markov=True),
    Profile('C-GATE', True, True, 'hybrid', gate=True),
    Profile('C-ALL', True, True, 'hybrid', True, True, True),
    Profile('C3', True, True, 'hybrid', True, True, width=3),
    Profile('C-AUTO', True, True, 'hybrid', True, True, width='auto'),
    Profile('B-ROUTED',True,True,'routed'),Profile('C-ROUTED',True,True,'routed',True,True),
    Profile('B-NO-DENSE',False,True,'hybrid'),Profile('B-NO-WOA',True,False,'hybrid'),
    Profile('B-OWN-EXPERTS',True,True,'ours'),
    Profile('C-LANE',True,True,'lane',True,True),Profile('B-LANE-GATE',True,True,'lane',gate=True),
    Profile('A-ORACLE',oracle=True),Profile('B-ORACLE',True,True,'hybrid',oracle=True),
    Profile('C-ORACLE',True,True,'hybrid',True,True,oracle=True),
    Profile('B-LANE-ORACLE',True,True,'lane',oracle=True),
)}

def capture_policy(scope,confidence):
    """Finite declared bank geometry; ABC keeps the full reference, not extra target ablations."""
    if scope in ('draft-balanced','draft-speed'):
        batches=tuple(range(1,9));prefix='GN-BALANCED' if scope=='draft-balanced' else 'GN-SPEED'
        gates=(False,True) if confidence else (False,)
        return dict(batches=batches,widths=(5,3),families=(prefix+'-WIDE',),
                    samplers=tuple((b,b<=3,True,g) for b in batches for g in gates),
                    reference_samplers=tuple((b,False,False,False) for b in batches))
    if scope=='selected':
        batches=tuple(range(1,9))
        return dict(batches=batches,widths=(5,),families=('B','B-NO-WOA'),samplers=(),
                    reference_samplers=tuple((b,False,False,False) for b in batches))
    if scope=='isolate':
        batches=(1,2,8)
        return dict(batches=batches,widths=(5,),families=('B','B-NO-DENSE','B-NO-WOA','B-OWN-EXPERTS'),
                    samplers=(),reference_samplers=tuple((b,False,False,False) for b in batches))
    if scope not in ('all','b1','abc'):raise ValueError('Unknown capture scope')
    batches=(1,) if scope=='b1' else tuple(range(1,9))
    widths=(5,3) if scope=='all' else (5,)
    names=('B',) if scope=='abc' else ('B','B-DENSE','B-WOA','B-EXPERTS','B-LANE','B-NO-DENSE','B-NO-WOA')
    gates=(False,True) if confidence and scope!='b1' else (False,)
    samplers=tuple((b,fp8,fused,gate) for b in batches for fp8 in ((False,True) if b<=3 else (False,)) for fused in (False,True) for gate in gates)
    aliases=()
    if scope=='abc':
        samplers=tuple((b,b<=3,True,False) for b in batches)
        aliases=tuple((b,False,False,False) for b in batches)
    return dict(batches=batches,widths=widths,families=names,samplers=samplers,reference_samplers=aliases)

def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def descriptor(b,k=5):
    if type(b) is not int or b not in range(1,9) or type(k) is not int or k not in (3,5): raise ValueError('Unsupported B/k')
    return ((k+1)*b,b,True,False,0)
def family_key(family,b,k): return (*family,b,k)
def lane_rows(experts,b,k): return experts=='lane' or (experts=='hybrid' and (b==1 or (k==5 and b>=7)))
def eligible_layout(scheduled,spec):
    if not 1<=len(scheduled)<=8: return None
    ks={len(spec.get(r,())) for r in scheduled}
    if len(ks)!=1: return None
    k=next(iter(ks))
    if k not in (3,5) or any(n!=k+1 for n in scheduled.values()): return None
    return len(scheduled),k
def validate_costs(costs,identity):
    if not isinstance(costs,dict) or costs.get('schema')!='star.abc.costs.v1' or costs.get('identity')!=identity: raise ValueError('Wrong local cost identity')
    if costs.get('hardware')!='STAR-140W' or costs.get('source')!='measured_full_cycle': raise ValueError('Measured STAR full-cycle costs required')
    pin=costs.get('evidence_sha256','')
    if not isinstance(pin,str) or len(pin)!=64 or set(pin)-set('0123456789abcdef'):raise ValueError('Missing local cost evidence hash')
    rows=costs.get('milliseconds',{})
    for b in range(1,9):
        for k in (3,5):
            v=rows.get(str(b),{}).get(str(k))
            if type(v) not in (int,float) or not math.isfinite(v) or v<=0: raise ValueError('Missing/invalid B/k cost')
    return rows


def validate_policy_costs(costs,identity,profile):
    rows=validate_costs(costs,identity)
    if profile not in PROFILES:raise ValueError('Unknown measured-cost profile')
    p=PROFILES[profile]
    if profile.startswith('GN-'):
        c=costs.get('context')
        if not isinstance(c,dict) or c.get('target_family')!=list(p.family()):raise ValueError('Measured target family differs')
        if any(type(c.get(k)) is not bool or c[k]!=getattr(p,k) for k in ('fp8_draft','fused_markov','gate')):raise ValueError('Measured sampler/gate context differs')
        if c.get('draft_tokens')!=5:raise ValueError('Measured full proposal width differs')
        pin=c.get('calibration_prompt_sha256','')
        if len(pin)!=64 or set(pin)-set('0123456789abcdef'):raise ValueError('Missing measured prompt identity')
    return rows

def validate_routes(r,identity):
    if not isinstance(r,dict) or r.get('schema')!='star.abc.routes.v1' or r.get('identity')!=identity:raise ValueError('Wrong route identity')
    if r.get('source')!='matched_star_wall_and_quality' or r.get('hardware')!='STAR-140W':raise ValueError('Matched STAR results required')
    pin=r.get('evidence_sha256','')
    if not isinstance(pin,str) or len(pin)!=64 or set(pin)-set('0123456789abcdef'):raise ValueError('Missing route evidence hash')
    if set(r.get('experts',{}))!={str(b) for b in range(1,9)} or any(v not in ('hybrid','lane') for v in r['experts'].values()):raise ValueError('All B1..8 routes required')
    return r

class Selection:
    """Transactional selection; commit never captures or implicitly qualifies anything."""
    def __init__(self, identity, available):
        self.identity=identity;self.available=frozenset(available)
        if 'A' not in self.available: raise ValueError('Qualified optimized reference A required')
        self.selected='A';self.session=None;self.pending=None;self.epoch=0
    def prepare(self,token,profile,session,identity,epoch):
        if self.pending is not None: raise ValueError('Transaction pending')
        if identity!=self.identity or profile not in self.available or epoch!=self.epoch+1: raise ValueError('Unqualified/stale selection')
        if not all(isinstance(v,str) and v for v in (token,session)): raise ValueError('Missing transaction/session')
        self.pending=(token,profile,session,epoch)
    def commit(self,token):
        if self.pending is None or self.pending[0]!=token: raise ValueError('Unprepared commit')
        _,self.selected,self.session,self.epoch=self.pending;self.pending=None
    def rollback(self,token):
        if self.pending is not None and self.pending[0]!=token: raise ValueError('Wrong rollback token')
        self.pending=None
    def end(self): self.pending=None;self.session=None;self.selected='A'
    def record(self): return dict(identity=self.identity,available=sorted(self.available),selected=self.selected,session=self.session,epoch=self.epoch,pending=self.pending)

