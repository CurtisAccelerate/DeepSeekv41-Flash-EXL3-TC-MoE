"""Finite seven-arm future protocol. Selection never prepares or captures."""
from dataclasses import dataclass,asdict
import hashlib,json
from types import MappingProxyType
from .attention_profiles import CURRENT,Profile
from .native_mixed_private import PIN,PROOF_PIN,VERSION
SCHEMA='w26.cumulative-bank.v1'
BASELINE='baseline';ATTENTION='attention-only';SAMPLER='sampler-only';COMBINED='attention-sampler'
NEXT='next-attention';NEXT_COMBINED='next-attention-sampler'
MIXED='next-attention-sampler-native-mixed'
ARMS={BASELINE:(None,False),ATTENTION:('current',False),SAMPLER:(None,True),COMBINED:('current',True),NEXT:('next',False),NEXT_COMBINED:('next',True),MIXED:('next_mixed',True)}
DESCRIPTOR=(6,1,True,False,0)
MIXED_COMPONENT=dict(version=VERSION,cubin_sha256=PIN,qualification_sha256=PROOF_PIN,descriptor=DESCRIPTOR)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def family_attention_receipts(rows):
    """Rank-local state hashes differ legitimately; bind all four in one identity."""
    if len(rows)!=4 or any(not isinstance(row,dict) for row in rows):raise ValueError('Four attention receipt maps required')
    if any(set(row)!=set(rows[0]) for row in rows):raise ValueError('Attention readiness differs across ranks')
    result={}
    for key in rows[0]:
        if key not in ('current','next','next_mixed','r2'):raise ValueError('Unexpected attention profile key')
        common={k:v for k,v in rows[0][key].items() if k!='state_receipt_sha256'}
        if key=='next_mixed' and common.get('native')!=MIXED_COMPONENT:raise ValueError('Mixed native component proof differs')
        if any({k:v for k,v in row[key].items() if k!='state_receipt_sha256'}!=common for row in rows):
            raise ValueError('Attention source/profile proof differs across ranks')
        pins=[row[key]['state_receipt_sha256'] for row in rows]
        if any(not isinstance(pin,str) or len(pin)!=64 or any(c not in '0123456789abcdef' for c in pin) for pin in pins):
            raise ValueError('Invalid rank state receipt hash')
        result[key]=dict(common,state_receipt_sha256_by_rank=pins)
    return result
@dataclass(frozen=True)
class FamilyIdentity:
    baseline:dict
    sampler_source_sha256:str
    draft_contract_sha256:str
    attention_banks:dict|None=None
    def digest(self):return digest(asdict(self))
    def arm_identity(self,arm):
        attention,sampler=ARMS[arm]
        identity=dict(family_sha256=self.digest(),profile=arm,
            attention=('compressed-t6-split16-v1' if attention=='current' else
                (self.attention_banks or {}).get(attention,{}).get('version','unprepared-next')) if attention else 'incumbent',
            sampler='captured-local-nccl-b1k5-v4' if sampler else 'incumbent',
            target_sampling='unchanged',descriptor=DESCRIPTOR,
            native=(self.attention_banks or {}).get('next_mixed',{}).get('native',{'version':'unprepared-mixed'}) if attention=='next_mixed' else {'version':'boot'},
            other_descriptors_native='boot')
        r2=(self.attention_banks or {}).get('r2')
        if r2 is not None and attention=='next_mixed':
            identity['r2']={k:v for k,v in r2.items() if k!='state_receipt_sha256_by_rank'}
        return identity
    def arm_digest(self,arm):return digest(self.arm_identity(arm))

class ArmControl:
    def __init__(self,identity,attention_entry=None,attention_qualification=None,sampler_state=None,sampler_receipt=None,*,next_entry=None,next_qualification=None,next_profile=None,mixed_bank=None):
        self.identity=identity;self.entry=attention_entry;self.attention_qualification=attention_qualification
        self._identity_digest=identity.digest()
        if next_entry is not None:
            if type(next_profile) is not Profile or next_profile==CURRENT:raise ValueError('Unsupported next bank profile')
            if attention_entry is None or next_entry is attention_entry or next_entry['graph'] is attention_entry['graph'] or next_entry['pool']==attention_entry['pool'] or next_entry['attention_state'] is attention_entry['attention_state']:
                raise ValueError('Attention graph/pool/workspace separation failed')
        elif next_qualification is not None or next_profile is not None:raise ValueError('Partial next bank publication')
        mixed=mixed_bank or {}
        mixed_entry=mixed.get('entry')
        if mixed_entry is not None:
            if mixed.get('component')!=MIXED_COMPONENT or (identity.attention_banks or {}).get('next_mixed',{}).get('native')!=MIXED_COMPONENT:raise ValueError('Mixed publication differs from family component proof')
            if next_entry is None or mixed.get('profile')!=next_profile or not mixed.get('qualification') or not mixed['qualification'].benchmark_ready():raise ValueError('Mixed requires qualified next and independent proof')
            for prior in (attention_entry,next_entry):
                if mixed_entry is prior or mixed_entry['graph'] is prior['graph'] or mixed_entry['pool']==prior['pool'] or mixed_entry['attention_state'] is prior['attention_state']:raise ValueError('Mixed graph/pool/workspace separation failed')
            mixed['native'].assert_intact()
        elif mixed:raise ValueError('Partial mixed bank publication')
        self.mixed_native=mixed.get('native');self._mixed_native=self.mixed_native
        self.entries=MappingProxyType({'current':attention_entry,'next':next_entry,'next_mixed':mixed_entry})
        self.qualifications=MappingProxyType({'current':attention_qualification,'next':next_qualification,'next_mixed':mixed.get('qualification')})
        self.profiles=MappingProxyType({'current':CURRENT,'next':next_profile,'next_mixed':mixed.get('profile')})
        self._bindings={k:self.entry_signature(v) for k,v in self.entries.items()}
        self.sampler_state=sampler_state;self.sampler_receipt=sampler_receipt or {}
        self.selected=BASELINE;self.pending=None;self.session_token=None;self.reason=None
        self.production_active=False;self.production_pending=None;self.production_approval=None
        self.coverage={'attention_hits':0,'sampler_hits':0,'sampler_fallbacks':{}}
    @staticmethod
    def entry_signature(entry):
        if entry is None:return None
        return (id(entry),id(entry['graph']),id(entry['output']),entry['pool'],id(entry['attention_state']),
            id(entry['addresses']),entry['attention_calls'],id(entry.get('native_roots')))
    def assert_banks_intact(self):
        if self.entry is not self.entries['current'] or self.attention_qualification is not self.qualifications['current']:
            raise RuntimeError('Current attention aliases changed')
        if self.mixed_native is not self._mixed_native:raise RuntimeError('Mixed native root changed')
        for key,entry in self.entries.items():
            if self.entry_signature(entry)!=self._bindings[key]:raise RuntimeError('Captured attention bank binding changed')
            if entry is not None and entry['attention_state'].profile!=self.profiles[key]:raise RuntimeError('Captured attention profile changed')
    def availability(self):
        self.assert_banks_intact()
        ready={k:(entry is not None and self.qualifications[k] is not None and self.qualifications[k].benchmark_ready()) for k,entry in self.entries.items()}
        s=(self.sampler_state is not None and self.sampler_state.ready and self.sampler_receipt.get('ready') is True and self.sampler_receipt.get('whole_call_exact') is True and self.sampler_receipt.get('all_rank_ready') is True)
        return {arm:bool((profile is None or ready[profile]) and (not sampler or s)) for arm,(profile,sampler) in ARMS.items()}
    def disable(self,reason):
        # Hold all graph/workspace/weight roots; rollback is a selector change.
        self.selected=BASELINE;self.pending=None;self.session_token=None;self.reason=str(reason)
        self.production_active=False;self.production_pending=None;self.production_approval=None
    def prepare(self,r,*,paused,drained,admission_closed,identity):
        if self.production_active or self.production_pending is not None:raise ValueError('Production mode blocks benchmark selection')
        if self.identity.digest()!=self._identity_digest:raise RuntimeError('Family identity contents changed')
        if r.get('schema')!=SCHEMA or r.get('mode')!='benchmark_only':raise ValueError('Unsupported cumulative protocol')
        if not(paused and drained and admission_closed):raise ValueError('Select requires paused/drained/closed owner')
        arm=r.get('bank_id');token=r.get('transaction_token');session=r.get('benchmark_session_token')
        if arm not in ARMS or not isinstance(token,str) or not token or not isinstance(session,str) or not session:raise ValueError('Invalid arm/token/session')
        if identity!=self.identity or r.get('identity_sha256')!=identity.arm_digest(arm):raise ValueError('Family or requested arm identity mismatch')
        if self.pending is not None:raise ValueError('Transaction already pending')
        if self.session_token is not None:raise ValueError('End the prior private session before select')
        if not self.availability()[arm]:raise ValueError('Requested arm unavailable; no downgrade')
        self.pending=(token,arm,session)
        return dict(prepared=True,transaction_token=token,bank_id=arm,identity_sha256=identity.arm_digest(arm),components=identity.arm_identity(arm))
    def commit(self,token,*,all_rank_prepared):
        if self.production_active or self.production_pending is not None:raise ValueError('Production mode blocks benchmark commit')
        if not all_rank_prepared or self.pending is None or self.pending[0]!=token or not self.availability()[self.pending[1]]:
            self.disable('Unanimous prepare/token missing');return False
        _,self.selected,self.session_token=self.pending;self.pending=None;return True
    def check_session(self,*,armed,identity,session):
        if self.production_pending is not None:raise RuntimeError('Production transaction pending')
        if self.production_active:
            if armed or session is not None or self.session_token is not None or self.pending is not None or self.selected!=MIXED or identity!=self.identity:raise RuntimeError('Invalid production execution binding')
            if not self.availability()[MIXED]:raise RuntimeError('Production bank lost readiness')
            return
        if self.pending is not None:raise RuntimeError('Replay during transaction')
        if identity!=self.identity:raise RuntimeError('Loaded family changed')
        if not armed or session!=self.session_token:raise RuntimeError('Unarmed or different private session')
        if not self.availability()[self.selected]:raise RuntimeError('Selected component lost readiness')
    def lookup(self,*,runtime_mode,descriptor,armed,identity,session):
        if self.production_pending is not None:raise RuntimeError('Production transaction pending')
        if self.pending is not None:raise RuntimeError('Replay during transaction')
        if self.selected==BASELINE:return None
        self.check_session(armed=armed,identity=identity,session=session)
        if not ARMS[self.selected][0] or runtime_mode!='FULL' or tuple(descriptor)!=DESCRIPTOR:return None
        self.coverage['attention_hits']+=1;return self.entries[ARMS[self.selected][0]]
    def r2_allowed(self,*,runtime_mode,armed,identity,session):
        """R2 graphs serve only the mixed arm, under the same session/production binding."""
        if self.production_pending is not None:raise RuntimeError('Production transaction pending')
        if self.pending is not None:raise RuntimeError('Replay during transaction')
        if self.selected!=MIXED or runtime_mode!='FULL':return False
        self.check_session(armed=armed,identity=identity,session=session)
        return True
    def sampler(self,*,armed,identity,session):
        if self.production_pending is not None:raise RuntimeError('Production transaction pending')
        if self.pending is not None:raise RuntimeError('Sampling during transaction')
        if self.selected==BASELINE:return None
        self.check_session(armed=armed,identity=identity,session=session)
        return self.sampler_state if ARMS[self.selected][1] else None
    def status(self):
        if self.identity.digest()!=self._identity_digest:raise RuntimeError('Family identity contents changed')
        return dict(schema=SCHEMA,selected_bank=self.selected,default_bank=BASELINE,baseline_retained=True,
            selected_components=self.identity.arm_identity(self.selected),identity_sha256=self.identity.arm_digest(self.selected),
            family_identity=asdict(self.identity),family_sha256=self.identity.digest(),
            available_arms={arm:dict(ready=ready,identity_sha256=self.identity.arm_digest(arm),components=self.identity.arm_identity(arm)) for arm,ready in self.availability().items()},
            sampler_qualification=self.sampler_receipt,attention_qualification=None if self.attention_qualification is None else asdict(self.attention_qualification),
            attention_bank_qualifications={k:None if q is None else asdict(q) for k,q in self.qualifications.items()},
            benchmark_ready=any(v for k,v in self.availability().items() if k!=BASELINE),production_enabled=self.production_active,production_pending=self.production_pending is not None,production_approval=self.production_approval,pending=self.pending is not None,reason=self.reason,coverage=self.coverage)
