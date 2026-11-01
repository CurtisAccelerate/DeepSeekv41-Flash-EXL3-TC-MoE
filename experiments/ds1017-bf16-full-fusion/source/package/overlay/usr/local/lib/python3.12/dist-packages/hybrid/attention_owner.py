"""Future owner-side vertical adapter. Not installed in the retained API.

Requires matching worker RPCs and session-aware admission middleware. Missing
capabilities fail before resume; optional candidate rejection preserves baseline.
"""
from .cumulative_contract import SCHEMA, BASELINE, ARMS

class AttentionOwner:
    def __init__(self, control):
        self.control=control
        self.session=None
        self.selected=BASELINE
        self.identity_sha256=None

    @staticmethod
    def validate(q):
        fields={'attention_bank_status':{'op'},
                'attention_bank_select':{'op','request'},
                'attention_benchmark_begin':{'op','benchmark_session_token'},
                'attention_benchmark_end':{'op','benchmark_session_token'}}
        if not isinstance(q,dict) or q.get('op') not in fields or set(q)!=fields[q['op']]:
            raise ValueError('Invalid attention control envelope')
        if q['op']=='attention_bank_select':
            r=q['request'];required={'schema','mode','bank_id','transaction_token','benchmark_session_token','identity_sha256'}
            if not isinstance(r,dict) or set(r)!=required or r['schema']!=SCHEMA or r['mode']!='benchmark_only' or r['bank_id'] not in ARMS:
                raise ValueError('Invalid attention bank request')
            if any(not isinstance(r[k],str) or not r[k] for k in required):
                raise ValueError('Attention request fields must be nonempty strings')
        elif q['op']!='attention_bank_status' and (not isinstance(q['benchmark_session_token'],str) or not q['benchmark_session_token']):
            raise ValueError('Missing benchmark session')
        return q['op']

    async def rows(self,name,args=()):
        rows=await self.control.rpc(name,args,timeout=60)
        if len(rows)!=4 or sorted(r.get('rank') for r in rows)!=list(range(4)):
            raise RuntimeError('Missing/duplicate attention rank receipts')
        return rows

    async def abort(self,reason):
        c=self.control
        if c.admission:
            c.admission.open=False
            c.admission.attention_guarded=True
            c.admission.attention_session=None
            # Accepted handlers may not yet have enqueued into the scheduler.
            # Drain them before pausing, while their engine work can complete.
            await c.admission.wait_drained()
        await c.pause()
        rows=await self.rows('w26_attention_bank_abort',(str(reason),))
        if not all(r.get('ok') and r.get('selected_bank')==BASELINE and r.get('baseline_retained') for r in rows):
            await c.quarantine('Uncertain attention rollback: '+str(reason),poison=True)
            raise RuntimeError('Could not prove all-rank baseline selection')
        self.selected=BASELINE;self.session=None
        c.admission.attention_guarded=False
        return rows

    def guard_legacy_operation(self,op):
        if (getattr(self.control.admission,'production_active',False) or getattr(self.control.admission,'production_transition',False)) and op!='status':raise ValueError('Disable production bank before legacy operations')
        if (self.session is not None or getattr(self.control.admission,'attention_guarded',False)) and op!='status':
            raise ValueError('End the private benchmark before changing native/profile/admission state')

    async def handle_locked(self,q):
        """Caller holds existing Control.lock; validates before any state change."""
        op=self.validate(q);c=self.control
        if getattr(c.admission,'production_active',False) or getattr(c.admission,'production_transition',False):raise ValueError('Production mode blocks private benchmark operations')
        if op=='attention_bank_status':
            return dict(attention_ranks=await self.rows('w26_attention_bank_status'),status=c.snapshot(),scheduler_paused=await c.engine.is_paused())
        if c.poisoned:raise RuntimeError('Owner poisoned')
        if not getattr(c.admission,'attention_protocol_ready',False):
            raise ValueError('Session-aware admission middleware is not installed')
        if op=='attention_benchmark_end':
            if q['benchmark_session_token']!=self.session:raise ValueError('Wrong active benchmark session')
            rows=await self.abort('benchmark_complete')
            await c.qualify()
            return dict(abort=rows,status=c.snapshot(),scheduler_paused=True)
        if op=='attention_benchmark_begin':
            if q['benchmark_session_token']!=self.session:raise ValueError('No selected matching benchmark session')
            if c.admission.open or not await c.engine.is_paused():raise ValueError('Benchmark begin requires closed and paused owner')
            await c.qualify()
            rows=await self.rows('w26_attention_benchmark_arm',(self.session,self.identity_sha256))
            if not all(r.get('ok') and r.get('session_armed') and r.get('identity_sha256')==self.identity_sha256 for r in rows):
                return dict(candidate_disabled=True,abort=await self.abort('session readiness rejected'),status=c.snapshot())
            c.admission.attention_session=self.session
            await c.engine.resume_generation()
            if await c.engine.is_paused():
                await self.abort('benchmark resume not confirmed')
                raise RuntimeError('Benchmark resume not confirmed')
            c.phase='benchmark_serving';c.persist()
            return dict(session_armed=True,selected_bank=self.selected,commit=rows,status=c.snapshot())
        # Selection is serialized and is never a capture or model load.
        if self.session is not None:raise ValueError('End the existing private benchmark before selecting another bank')
        r=q['request'];c.admission.open=False;c.admission.attention_guarded=True;c.admission.attention_session=None
        await c.admission.wait_drained()
        await c.pause();await c.qualify()
        rows=await self.rows('w26_attention_bank_prepare',(r,))
        expected=(r['transaction_token'],r['bank_id'],r['identity_sha256'])
        ok=all(x.get('ok') and x.get('prepared') and (x.get('transaction_token'),x.get('bank_id'),x.get('identity_sha256'))==expected for x in rows)
        if not ok:
            return dict(candidate_disabled=True,prepare=rows,abort=await self.abort('prepare rejected'),status=c.snapshot())
        committed=await self.rows('w26_attention_bank_commit',(r['transaction_token'],))
        if not all(x.get('ok') and x.get('committed') and (x.get('transaction_token'),x.get('bank_id'),x.get('identity_sha256'))==expected for x in committed):
            return dict(candidate_disabled=True,prepare=rows,commit=committed,abort=await self.abort('commit rejected'),status=c.snapshot())
        self.selected=r['bank_id'];self.session=r['benchmark_session_token'];self.identity_sha256=r['identity_sha256']
        await c.qualify();c.persist()
        return dict(prepare=rows,commit=committed,selected_bank=self.selected,identity_sha256=self.identity_sha256,status=c.snapshot(),scheduler_paused=True)
