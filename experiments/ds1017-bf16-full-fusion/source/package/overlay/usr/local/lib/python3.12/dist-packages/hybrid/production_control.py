"""Default-off ordinary serving selection. No capture, module injection or load.

Approval is an operator-owned pinned JSON receipt, not a new signing system.
Unknown/partial failures stay closed and quarantined; recovery needs an operator.
"""
import hashlib,json,uuid
from pathlib import Path
from .cumulative_contract import MIXED,BASELINE

SCOPE={'descriptor':[6,1,True,False,0],'attention':'next-8-8-copy','sampler':'existing-qualified-b1-k5','unsupported_descriptors_native':'boot','target_sampling':'unchanged','performance_scope':'quality-qualified-C1; unsupported concurrency not assumed faster'}
MIXED_SHA='edc623dbeea0931aa37507ed8b2b78f12df054a3756d64507479fcab6c8529e3'
MIXED_MANIFEST='38910deb255139ced1775aa768027a68692d92a42d98203c751c972f60d58efd'
COUNTS={'rot':200,'a2':25,'a3':175,'b2':25,'b3':175}
def pin(value):return isinstance(value,str) and len(value)==64 and not(set(value)-set('0123456789abcdef'))
def rows4(rows):
    if not isinstance(rows,list) or len(rows)!=4 or sorted(r.get('rank',-1) for r in rows)!=list(range(4)) or not all(r.get('ok') is True for r in rows):raise RuntimeError('Four unique successful production receipts required')
    return sorted(rows,key=lambda r:r['rank'])
def runtime_identity(c):
    s=c.snapshot()
    return dict(owner_pid=s['owner_pid'],workers=[dict(rank=r['rank'],pid=r['pid'],capture_generation=r['capture_generation'],graph_sets=r['graph_sets']) for r in sorted(s['ranks'],key=lambda x:x['rank'])])

def worker_invoke(worker,method,*args):
    from .cumulative_worker import WorkerCumulative
    a=getattr(worker,'_w26_attention_adapter',None)
    if type(a) is not WorkerCumulative:raise RuntimeError('Production requires initialized cumulative worker')
    c=a.controller;identity=a.current_identity()
    if method=='status':return a.status()
    if method=='abort':
        result=a.abort(args[0]);a.current_identity()
        return dict(result,production_enabled=False,production_pending=False,session_armed=False)
    if method=='prepare':
        r=args[0]
        if set(r)!={'token','bank','identity_sha256','approval_sha256'} or r['bank']!=MIXED or not pin(r['approval_sha256']) or not isinstance(r['token'],str) or not r['token']:raise ValueError('Invalid production prepare')
        if c.production_active or c.production_pending is not None or c.pending is not None or c.session_token is not None or c.selected!=BASELINE or a.session_armed or a._production_binding is not None:raise ValueError('Production prepare requires idle unarmed baseline')
        if not c.availability()[MIXED] or identity.arm_digest(MIXED)!=r['identity_sha256']:raise ValueError('Unqualified production arm/identity')
        c.production_pending=(r['token'],r['identity_sha256'],r['approval_sha256'])
        return dict(rank=int(worker.rank),ok=True,prepared=True,**r)
    if method=='commit':
        token=args[0];pending=c.production_pending
        if pending is None or pending[0]!=token or not c.availability()[MIXED] or identity.arm_digest(MIXED)!=pending[1]:raise ValueError('Production commit is not prepared')
        c.selected=MIXED;c.session_token=None;c.production_approval=pending[2];c.production_active=True;c.production_pending=None;a.session_armed=False
        a._production_binding=(id(a._family),id(c.entries['next_mixed']),id(c.mixed_native),c.production_approval)
        a.current_identity();a.light_bindings()
        return dict(rank=int(worker.rank),ok=True,committed=True,token=token,bank=MIXED,identity_sha256=pending[1],approval_sha256=pending[2],production_enabled=True,session_armed=False)
    raise ValueError('Unknown production worker method')

class ProductionOwner:
    def __init__(self,control):self.control=control;self.active=False;self.approval=None
    @staticmethod
    def validate(q):
        fields={'production_bank_enable':{'op','approval_sha256'},'production_bank_disable':{'op'},'production_bank_status':{'op'}}
        if not isinstance(q,dict) or q.get('op') not in fields or set(q)!=fields[q['op']]:raise ValueError('Invalid production envelope')
        if q['op']=='production_bank_enable' and not pin(q['approval_sha256']):raise ValueError('Invalid approval pin')
        return q['op']
    def approval_record(self,sha,status):
        c=self.control
        def proof(expected):
            if not pin(expected):raise ValueError('Invalid proof hash')
            path=c.root/'production-proofs'/(expected+'.json');raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Production proof bytes differ')
            return json.loads(raw)
        approval=proof(sha)
        fields={'schema','approved','winner','family_sha256','identity_sha256','owner_receipt_sha256','source_manifest_sha256','comparison_sha256','quality_sha256','scope','code_quality_pass','prose_quality_pass','comparison_pass'}
        if set(approval)!=fields or approval['schema']!='w47.production-approval.v1' or approval['approved'] is not True or approval['winner']!=MIXED or approval['scope']!=SCOPE or any(approval[k] is not True for k in ('code_quality_pass','prose_quality_pass','comparison_pass')):raise ValueError('Production approval absent or scope differs')
        proof(approval['comparison_sha256']);proof(approval['quality_sha256'])
        owner=proof(approval['owner_receipt_sha256'])
        if owner.get('runtime')!=runtime_identity(c):raise ValueError('Approval belongs to another runtime owner')
        physical=owner.get('physical',{})
        required={'container_id','image','start_time','config_sha256','api_pid','gpu_pids','worker_pids','capture_generations','graph_sets','source_sha256'}
        if set(physical)!=required or not pin(physical['container_id']) or not physical['image'].startswith('sha256:') or not pin(physical['image'][7:]) or not physical['start_time'] or not pin(physical['config_sha256']):raise ValueError('Root physical owner receipt incomplete')
        workers=owner['runtime']['workers']
        if physical['worker_pids']!=[x['pid'] for x in workers] or physical['capture_generations']!=[x['capture_generation'] for x in workers] or physical['graph_sets']!=[x['graph_sets'] for x in workers] or len(physical['gpu_pids'])!=4:raise ValueError('Physical/namespace owner receipt differs')
        from .attention_startup import source_attestation
        source_attestation()
        manifest=Path(__file__).with_name('attention-imports.json')
        if hashlib.sha256(manifest.read_bytes()).hexdigest()!=approval['source_manifest_sha256']:raise ValueError('Production approval source manifest differs')
        if physical['source_sha256'].get('/usr/local/lib/python3.12/dist-packages/hybrid/attention-imports.json')!=approval['source_manifest_sha256']:raise ValueError('Physical source receipt differs from approval')
        for r in status:
            if r['family_sha256']!=approval['family_sha256'] or r['available_arms'][MIXED]['identity_sha256']!=approval['identity_sha256']:raise ValueError('Production approval family/arm differs')
        return approval
    async def rpc(self,method,args=()):return rows4(await self.control.rpc('w47_production_'+method,args,timeout=60))
    async def freeze(self):
        c=self.control;g=c.admission
        g.open=False;g.production_transition=True;g.attention_guarded=True;g.attention_session=None
        await g.wait_drained();await c.pause();await c.qualify()
    async def healthy(self,active,approval=None):
        rows=await self.rpc('status')
        for r in rows:
            if r['production_enabled'] is not active or r['production_pending'] or r['pending'] or r['session_armed'] or r['selected_bank']!=(MIXED if active else BASELINE) or not r['baseline_retained']:raise RuntimeError('Production all-rank mode differs')
            if active and (r['production_approval']!=approval or not r['available_arms'][MIXED]['ready']):raise RuntimeError('Production proof/readiness differs')
        return rows
    async def handle_locked(self,q):
        op=self.validate(q);c=self.control;g=c.admission
        if op=='production_bank_status':
            try:
                await c.qualify()
                # After disable, global native is mixed and private status must
                # not run. Legacy status is the correct inactive health path.
                ranks=await self.healthy(True,self.approval) if self.active else None
                return dict(production_active=self.active,approval_sha256=self.approval,production_ranks=ranks,status=c.snapshot(),scheduler_paused=await c.engine.is_paused())
            except BaseException as exc:
                g.open=False;g.production_transition=True
                await c.quarantine('Production health failed: '+repr(exc),poison=True)
                raise
        if c.poisoned:raise RuntimeError('Poisoned owner cannot change production')
        if op=='production_bank_enable' and (self.active or c.attention_bank.session is not None or c.attention_bank.selected!=BASELINE or g.attention_guarded or g.production_transition):raise ValueError('Production enable requires finished benchmark and idle baseline')
        if op=='production_bank_disable' and not self.active:raise ValueError('Production bank is not active')
        try:
            await self.freeze()
            if op=='production_bank_enable':
                initial=await self.healthy(False);approval=self.approval_record(q['approval_sha256'],initial)
                token=uuid.uuid4().hex;r=dict(token=token,bank=MIXED,identity_sha256=approval['identity_sha256'],approval_sha256=q['approval_sha256'])
                prepared=await self.rpc('prepare',(r,))
                if any(x.get('prepared') is not True or any(x.get(k)!=v for k,v in r.items()) for x in prepared):raise RuntimeError('Production prepare differs')
                committed=await self.rpc('commit',(token,))
                if any(x.get('committed') is not True or any(x.get(k)!=v for k,v in r.items()) or x.get('production_enabled') is not True or x.get('session_armed') is not False for x in committed):raise RuntimeError('Production commit differs')
                await self.healthy(True,q['approval_sha256']);await c.qualify()
                await c.engine.resume_generation()
                if await c.engine.is_paused():raise RuntimeError('Production resume not confirmed')
                # Single event-loop owner commits ordinary admission LAST.
                self.active=True;self.approval=q['approval_sha256'];g.production_active=True;g.attention_guarded=False;g.production_transition=False;g.open=True;c.phase='production_bank_serving';c.persist()
                result=dict(production_active=True,ordinary_admission=True,approval_sha256=self.approval,scope=SCOPE,prepare=prepared,commit=committed,status=c.snapshot())
            else:
                aborted=await self.rpc('abort',('production_disabled',))
                if any(x.get('production_enabled') is not False or x.get('selected_bank')!=BASELINE for x in aborted):raise RuntimeError('Production baseline rollback unproved')
                await self.healthy(False);await c.qualify()
                transaction=await c.version(dict(arm='transpose',version='w25-native-mixed',manifest='w25-native-mixed.json',manifest_sha256=MIXED_MANIFEST))
                if transaction.get('outcome')!='committed':raise RuntimeError('Current production native restore rejected')
                for row in rows4(transaction['steps'][1]['rows']):
                    value=row['result']
                    if value.get('node_counts')!=COUNTS or value.get('recaptures')!=0 or value.get('graph_handles_unchanged') is not True:raise RuntimeError('Legacy600-node restore proof differs')
                for row in rows4(transaction['commit']):
                    value=row['result']
                    if (value.get('version'),value.get('sha256'),value.get('phase'))!=('w25-native-mixed',MIXED_SHA,'committed'):raise RuntimeError('Current production native commit differs')
                await c.qualify()
                if c.collective_profile!='six' or c.active_versions.get('transpose')!='w25-native-mixed' or c.active_hashes.get('transpose')!=MIXED_SHA:raise RuntimeError('Current production selection differs')
                self.active=False;self.approval=None;g.production_active=False;g.production_transition=False;g.attention_guarded=False;c.phase='paused';c.persist()
                result=dict(production_active=False,current_production_restored=True,ordinary_admission=False,abort=aborted,version_transaction=transaction,status=c.snapshot(),scheduler_paused=True)
            c.save('production-bank.json',result);return result
        except BaseException as exc:
            # Even expected source/quality rejection remains closed. A partial
            # commit/unknown CUDA outcome never gets an optimistic rollback.
            g.open=False;g.production_transition=True
            await c.quarantine('Production transition failed: '+repr(exc),poison=True)
            raise
