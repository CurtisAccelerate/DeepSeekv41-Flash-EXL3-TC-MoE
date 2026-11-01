"""Future named worker RPC helpers. No preparation/capture is exposed by RPC."""
from .cumulative_contract import BASELINE, SCHEMA

def adapter(worker):
    value=getattr(worker,'_w26_attention_adapter',None)
    if value is None:raise ValueError('Attention bank was not prepared at initial startup')
    return value

class WorkerAttention:
    def __init__(self,worker,runtime,controller,current_identity):
        self.worker=worker;self.runtime=runtime;self.controller=controller
        self.current_identity=current_identity;self.session_armed=False
        runtime.attention_adapter=self

    def status(self):
        return dict(rank=int(self.worker.rank),ok=True,session_armed=self.session_armed,**self.controller.status())

    def prepare(self,request):
        self.session_armed=False
        # Owner's named RPC sequence is the pause/drain proof. The worker
        # cannot infer scheduler state from a CUDA synchronization alone.
        result=self.controller.prepare(request,paused=True,drained=True,admission_closed=True,identity=self.current_identity())
        return dict(rank=int(self.worker.rank),ok=True,**result)

    def commit(self,token):
        # Owner sends commit only after all four prepare receipts match.
        ok=self.controller.commit(token,all_rank_prepared=True)
        return dict(rank=int(self.worker.rank),ok=ok,committed=ok,transaction_token=token,bank_id=self.controller.selected,identity_sha256=self.current_identity().digest())

    def abort(self,reason):
        self.session_armed=False;self.controller.disable(reason)
        return dict(rank=int(self.worker.rank),ok=True,selected_bank=BASELINE,baseline_retained=True)

    def arm(self,session,identity_sha256):
        c=self.controller
        if c.pending is not None or session!=c.session_token or identity_sha256!=self.current_identity().digest():
            raise ValueError('Benchmark session/identity is not selected on this rank')
        self.session_armed=True
        return dict(rank=int(self.worker.rank),ok=True,session_armed=True,identity_sha256=identity_sha256)

    def lookup(self,mode,descriptor):
        c=self.controller
        if c.selected==BASELINE:return None
        if not self.session_armed:raise RuntimeError('Candidate replay outside armed benchmark session')
        return c.lookup(runtime_mode=mode,descriptor=descriptor,benchmark_session_token=c.session_token,admission_closed=True,identity=self.current_identity())

def invoke(worker,method,*args):
    a=getattr(worker,'_w26_attention_adapter',None)
    if a is None and method=='status':
        initial=getattr(worker,'_w26_attention_initial_receipt',{})
        return dict(rank=int(worker.rank),ok=True,schema=SCHEMA,selected_bank=BASELINE,default_bank=BASELINE,baseline_retained=True,benchmark_ready=False,production_enabled=False,reason=initial.get('reason','not_prepared'),initial_preparation=initial,session_armed=False)
    if a is None and method=='abort':
        return dict(rank=int(worker.rank),ok=True,selected_bank=BASELINE,baseline_retained=True)
    a=adapter(worker)
    if method not in ('status','prepare','commit','abort','arm'):raise ValueError('Unknown attention worker operation')
    try:return getattr(a,method)(*args)
    except ValueError as exc:
        # Only explicit protocol/precondition rejects become normal receipts.
        # Driver/communicator errors propagate to owner's uncertainty handling.
        return dict(rank=int(worker.rank),ok=False,error=str(exc),baseline_retained=True)
