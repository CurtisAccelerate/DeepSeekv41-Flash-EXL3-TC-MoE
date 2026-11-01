"""Atomic attention/sampler selector using the existing owner drain/session RPCs."""
from .cumulative_contract import BASELINE
class WorkerCumulative:
    def __init__(self,worker,runtime,controller,current_identity,proposer):
        self.worker=worker;self.runtime=runtime;self.controller=controller;self.current_identity=current_identity;self.session_armed=False
        self._production_binding=None
        self.proposer=proposer;self._family=controller.identity;self._state=controller.sampler_state
        self.r2=None
        self._entries=controller.entries;self._qualifications=controller.qualifications;self._profiles=controller.profiles
        if runtime is not None:runtime.attention_adapter=self
        proposer._w26_sampler_selector=self
        proposer._w26_sampler_published_selector=self
        worker._w26_attention_adapter=self
    def status(self):
        self.current_identity()
        return dict(rank=int(self.worker.rank),ok=True,session_armed=self.session_armed,**self.controller.status())
    def prepare(self,request):
        self.session_armed=False
        return dict(rank=int(self.worker.rank),ok=True,**self.controller.prepare(request,paused=True,drained=True,admission_closed=True,identity=self.current_identity()))
    def commit(self,token):
        c=self.controller;ok=c.commit(token,all_rank_prepared=True)
        return dict(rank=int(self.worker.rank),ok=ok,committed=ok,transaction_token=token,bank_id=c.selected,identity_sha256=self.current_identity().arm_digest(c.selected))
    def abort(self,reason):
        self.session_armed=False;self.controller.disable(reason)
        self._production_binding=None
        return dict(rank=int(self.worker.rank),ok=True,selected_bank=BASELINE,baseline_retained=True)
    def arm(self,session,identity_sha256):
        c=self.controller
        if c.production_active or c.production_pending is not None:raise ValueError('Production mode blocks benchmark arm')
        if c.pending is not None or session!=c.session_token or identity_sha256!=self.current_identity().arm_digest(c.selected):raise ValueError('Benchmark arm/session identity mismatch')
        self.session_armed=True
        return dict(rank=int(self.worker.rank),ok=True,session_armed=True,identity_sha256=identity_sha256)
    def lookup(self,mode,descriptor):
        c=self.controller
        if c.pending is not None:raise RuntimeError('Replay during transaction')
        self.light_bindings()
        if c.selected==BASELINE:return None
        return c.lookup(runtime_mode=mode,descriptor=descriptor,armed=self.session_armed,identity=self._family,session=c.session_token)
    def r2_allowed(self,mode):
        c=self.controller
        if c.pending is not None:raise RuntimeError('Replay during transaction')
        self.light_bindings()
        return c.r2_allowed(runtime_mode=mode,armed=self.session_armed,identity=self._family,session=c.session_token)
    def sampler(self):
        c=self.controller
        if c.pending is not None:raise RuntimeError('Sampling during transaction')
        self.light_bindings()
        if c.selected==BASELINE:return None
        return c.sampler(armed=self.session_armed,identity=self._family,session=c.session_token)
    def light_bindings(self):
        # Fixed O(1) object bindings only. Full graph/source/native inventory is
        # attested at status/prepare/commit/arm, never inside model-step dispatch.
        p=self.proposer
        c=self.controller
        if c.production_pending is not None:raise RuntimeError('Production transaction pending')
        if c.production_active and (c.selected!='next-attention-sampler-native-mixed' or c.session_token is not None or self.session_armed):raise RuntimeError('Production execution binding changed')
        if c.production_active!=(self._production_binding is not None):raise RuntimeError('Production mode/root binding changed')
        if self._production_binding is not None and self._production_binding!=(id(self._family),id(c.entries['next_mixed']),id(c.mixed_native),c.production_approval):raise RuntimeError('Production graph/approval binding changed')
        if (p._w26_sampler_selector is not self or p._w26_sampler_published_selector is not self
            or self.worker._w26_attention_adapter is not self or self.controller.identity is not self._family
            or self.controller.entries is not self._entries or self.controller.qualifications is not self._qualifications or self.controller.profiles is not self._profiles
            or self.controller.sampler_state is not self._state or getattr(p,'_w26_sampler_bank_state',None) is not self._state
            or (self.runtime is not None and self.runtime.attention_adapter is not self)):
            raise RuntimeError('Published cumulative binding changed')
