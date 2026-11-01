"""Future V1 FULL-only private graph bank. SOURCE STAGED, NOT QUALIFIED.

No lazy serving capture. No calls to capture_model/begin_capture/clear_all_graphs.
The native_contract must patch and qualify the private graph to the SAME native
functions/profile represented by its Bundle. Boot launch functions are not
assumed equivalent to later graph-only native replacements.
"""
import contextlib,os,sys
from hybrid.bank_protocol import Bank
from hybrid import w26_mhc_overlap as overlap

MODEL_DUAL_GRAPH_QUALIFIED=False
RUNTIMES={}
ACTIVE_RUNTIME=None
CAP_BYTES=256*1024**2
FLOOR_BYTES=1024**3

def active_candidate_context():
    """Source seam for the optional attention helper; never active in serving."""
    return ACTIVE_RUNTIME

def require_drafter_baseline(runner):
    """Fail closed unless the existing DSpark graph bank is already built.

    Read-only object inspection for the FUTURE initial preparation hook. This
    does not warm up, capture, replace, or register any drafter graph.
    """
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    drafter=getattr(runner,'drafter',None)
    if type(drafter).__name__!='DSparkProposer':raise ValueError('Expected established DSpark drafter')
    model=getattr(drafter,'model',None)
    if type(model) is not BreakableCUDAGraphWrapper:raise ValueError('Unsupported drafter wrapper')
    entries=getattr(model,'entries',None)
    if not entries or not any(getattr(e,'capture',None) is not None for e in entries.values()):raise ValueError('Baseline drafter capture not established')
    if not getattr(drafter,'_draft_layer_gid',None) or not getattr(drafter,'_ctx_slot',None) or not getattr(drafter,'_q_slot',None):raise ValueError('Baseline drafter metadata not initialized')
    return (id(model),tuple(sorted((repr(k),id(v),id(getattr(v,'capture',None))) for k,v in entries.items())))

def descriptor(d):
    return (d.num_tokens,d.num_reqs,d.uniform,d.has_lora,d.num_active_loras)

class Runtime:
    def __init__(self,runner,kernels,native_contract,before_free):
        import torch
        self.torch=torch;self.runner=runner;self.wrapper=runner.model
        self.kernels=kernels;self.native=native_contract;self.bank=Bank()
        self.mode='baseline';self.entry=None;self.pending_graphs=[];self.pool=torch.cuda.graph_pool_handle()
        self.attention_state=None;self.attention_calls=0
        assert self.pool!=self.wrapper.graph_pool,'Candidate must have a private allocator pool'
        self.baseline_entries={key:id(value) for key,value in self.wrapper.entries.items()}
        self.before_free=before_free
    def guard(self,reserve=0):
        self.torch.cuda.synchronize();free=self.torch.cuda.mem_get_info(self.runner.device)[0]
        growth=max(0,self.before_free-free)
        if growth+reserve>CAP_BYTES or free-reserve<FLOOR_BYTES:raise ValueError('Optional graph memory budget exceeded')
        assert self.baseline_entries=={key:id(value) for key,value in self.wrapper.entries.items()},'Incumbent entries changed'
        return dict(physical_growth=growth,free=free,budget=CAP_BYTES,floor=FLOOR_BYTES,
            allocated=self.torch.cuda.memory_allocated(self.runner.device),
            reserved=self.torch.cuda.memory_reserved(self.runner.device))
    @contextlib.contextmanager
    def candidate_math(self):
        global ACTIVE_RUNTIME
        assert overlap.ACTIVE is None and ACTIVE_RUNTIME is None
        overlap.ACTIVE=self.kernels;ACTIVE_RUNTIME=self
        try:yield
        finally:
            overlap.ACTIVE=None;ACTIVE_RUNTIME=None
    def capture_at_call(self,args,kwargs):
        from vllm.model_executor.offloader.base import get_offloader
        from vllm.distributed.device_communicators import pynccl_allocator
        # Use a single FULL graph. Attention break decorators run inline under
        # FULL; no Breakable TLS entry is needed and no candidate is registered
        # with the W9 baseline/native graph registry.
        saved_pool=pynccl_allocator._graph_pool_id
        saved_stream=self.torch.cuda.current_stream()
        graph=self.torch.cuda.CUDAGraph(keep_graph=True)
        self.pending_graphs.append(graph) # retain even failed capture/instantiate objects
        addresses=self.wrapper._collect_tensor_addresses(args,kwargs)
        get_offloader().sync_prev_onload()
        try:
            pynccl_allocator.set_graph_pool_id(self.pool)
            if self.attention_state is not None:self.attention_state.begin_capture()
            with self.candidate_math(),self.torch.cuda.graph(graph,pool=self.pool,stream=self.torch.cuda.current_stream()):
                output=self.wrapper.runnable(*args,**kwargs)
                if self.kernels is not None:self.kernels.assert_drained()
                get_offloader().join_after_forward()
            if self.attention_state is not None:self.attention_state.finish_capture()
            graph.instantiate()
        finally:
            pynccl_allocator.set_graph_pool_id(saved_pool)
            # PyTorch's capture_end can raise before its stream context exits.
            # Do not promise recovery from an active/poisoned capture. Restore
            # current stream explicitly only after capture is known ended.
            if not self.torch.cuda.is_current_stream_capturing():self.torch.cuda.set_stream(saved_stream)
        # Strong output refs intentionally pin private candidate outputs. They
        # never use the baseline graph pool's weak output storage.
        self.entry=dict(graph=graph,output=output,addresses=addresses,pool=self.pool,attention_state=self.attention_state,attention_calls=self.attention_calls)
        return output
    def replay(self,entry,args,kwargs):
        from vllm.model_executor.offloader.base import get_offloader
        assert entry['addresses']==self.wrapper._collect_tensor_addresses(args,kwargs)
        get_offloader().sync_prev_onload();entry['graph'].replay();return entry['output']
    def dummy(self,mode,*,target_only=False):
        from vllm.config import CUDAGraphMode
        kwargs=dict(w26_target_bank_only=True) if target_only else {}
        from .initial_decode_contract import initial_decode
        context=initial_decode(self.runner,self.initial_consensus,self.initial_metadata_records) if target_only else contextlib.nullcontext()
        with context:
            return self.runner._dummy_run(6,cudagraph_runtime_mode=getattr(CUDAGraphMode,mode),force_attention=True,uniform_decode=True,uniform_query_len=6,allow_microbatching=False,skip_eplb=True,remove_lora=False,num_active_loras=0,is_graph_capturing=mode=='FULL',**kwargs)

def intercept(wrapper,ctx,args,kwargs):
    from .abc_runtime import intercept as abc_intercept  # STAR_ABC integration
    abc = abc_intercept(wrapper,ctx,args,kwargs)
    if abc is not None:return abc
    rt=RUNTIMES.get(id(wrapper))
    if rt is None:return False,None
    r2=sys.modules.get('hybrid.r2_bank')
    if r2 is not None:
        handled=r2.intercept(wrapper,ctx,args,kwargs)
        if handled is not None:return handled
    if ctx.cudagraph_runtime_mode.name!='FULL' or descriptor(ctx.batch_descriptor)!=(6,1,True,False,0):return False,None
    if rt.mode=='capture':return True,rt.capture_at_call(args,kwargs)
    if rt.mode=='qualify_replay':return True,rt.replay(rt.entry,args,kwargs)
    attention_adapter=getattr(rt,'attention_adapter',None)
    if attention_adapter is not None:
        selected=attention_adapter.lookup('FULL',descriptor(ctx.batch_descriptor))
        if selected is None:return False,None
        return True,rt.replay(selected,args,kwargs)
    # Attention preparation registers the runtime before publishing its worker
    # adapter. Baseline dummy execution must go straight to the incumbent;
    # BootNativeContract intentionally does not implement the old mHC Bundle API.
    if rt.kernels is None:return False,None
    selected=rt.bank.lookup(runtime_mode='FULL',descriptor=descriptor(ctx.batch_descriptor),current_bundle=rt.native.current_bundle())
    if selected is None:return False,None
    return True,rt.replay(selected,args,kwargs)

def prepare_after_baseline(worker,native_contract=None):
    """OLD mHC-only staged hook; never use as attention startup orchestration.

Returns disabled while integration is gated.

Required native_contract methods: current_bundle(), preflight(),
patch_and_qualify_private(entry), assert_baseline_unchanged(),
qualify_dummy_state(runner). The final method must prove all dummy writes
including compressor/MTP state are benign or restored; KV slot=-1 alone is
not proof of every stateful buffer. No invented existing runtime API is used.
"""
    if os.environ.get('W26_MHC_DUAL_GRAPH','0')!='1':return
    import torch,torch.distributed as dist
    from vllm.distributed import get_tp_group,graph_capture
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from w9_exl_ab import runtime as original
    runner=worker.model_runner;tp=get_tp_group();rt=None
    def consensus(x):
        values=[None]*tp.world_size;dist.all_gather_object(values,x,group=tp.cpu_group);return values
    def disabled(reason):
        if rt is not None:rt.bank.disabled(reason);RUNTIMES.pop(id(rt.wrapper),None)
        worker._w26_mhc_receipt=dict(ready=False,candidate_disabled=True,incumbent_retained=True,reason=reason)
    gate=MODEL_DUAL_GRAPH_QUALIFIED and native_contract is not None
    if not all(consensus(gate)):disabled('unqualified_dual_graph_or_missing_native_contract');return
    compatible=(type(runner.model) is BreakableCUDAGraphWrapper and runner.compilation_config.mode.name=='NONE' and worker.parallel_config.pipeline_parallel_size==1 and not worker.parallel_config.use_ubatching and original.S.ready and original.S.startup_capture_context is not None)
    if not all(consensus(compatible)):disabled('unsupported runner/compile/parallel/capture state');return
    # All recoverable preconditions must finish on ALL ranks BEFORE any model
    # collective. A rank-local exception during an in-flight collective is not
    # a safely recoverable qualification failure.
    error=None
    try:
        before_free=torch.cuda.mem_get_info(runner.device)[0]
        if before_free<FLOOR_BYTES+CAP_BYTES:raise ValueError('insufficient preallocation reserve')
        native_contract.preflight();native_contract.qualify_dummy_state(runner)
        layers=[m for m in runner.get_model().modules() if hasattr(m,'hc_attn_fn') and 0<=getattr(m,'_dsv41_idx',-1)<40]
        assert len(layers)==40
        kernels=overlap.OverlapKernels(layers,runner.device)
        rt=Runtime(runner,kernels,native_contract,before_free)
        kernels.qualify_locally();rt.guard(32*1024**2)
    except Exception as exc:
        # A healthy context allows rank-local optional JIT/setup failures to
        # disable everywhere. CUDA health failure propagates explicitly.
        torch.cuda.synchronize();error=repr(exc)
    if any(x is not None for x in consensus(error)):disabled('preparation resource/eligibility failure: '+str(error));return
    RUNTIMES[id(rt.wrapper)]=rt
    # Remaining stages are deliberately not reachable until the hard gate and
    # real native/dummy-state contract are qualified in an isolated harness.
    try:
        with graph_capture(runner.device,original.S.startup_capture_context):
            baseline=tuple(x.clone() for x in rt.dummy('FULL'))
            rt.mode='capture';capture_error=None
            try:rt.dummy('FULL');rt.guard()
            except Exception as exc:
                # Capture records GPU collectives but does not replay them.
                # All ranks agree before any candidate executable is launched.
                # A capture left active or unhealthy is NOT recoverable.
                if torch.cuda.is_current_stream_capturing():raise
                torch.cuda.synchronize();capture_error=repr(exc)
            if any(x is not None for x in consensus(capture_error)):
                disabled('optional capture failed: '+str(capture_error));return
            native_error=None;native_ok=False
            try:native_ok=bool(native_contract.patch_and_qualify_private(rt.entry))
            except Exception as exc:
                torch.cuda.synchronize();native_error=repr(exc)
            native_rows=consensus((native_ok,native_error))
            if not all(ok and error is None for ok,error in native_rows):disabled('private native function qualification failed');return
            rt.mode='qualify_replay';candidate=rt.dummy('FULL')
            torch.cuda.synchronize()
            parity=all(torch.isfinite(a).all().item() and torch.equal(a,b) for a,b in zip(baseline,candidate))
            qualified=all(consensus(parity))
            rt.mode='baseline';native_contract.assert_baseline_unchanged();memory=rt.guard()
            rt.bank.publish(native_contract.current_bundle(),rt.entry,parity=parity,native_patch_qualified=native_ok,all_rank_ready=qualified)
            if not qualified:disabled('engine dummy parity failed');return
            worker._w26_mhc_receipt=dict(ready=True,selected=False,memory=memory,descriptor=(6,1,True,False,0),pool='separate private pool')
    finally:
        overlap.ACTIVE=None
        if rt is not None:rt.mode='baseline'
