"""Future initial-only attention preparation. Default OFF; production disabled.
No user-facing control can capture. Every recoverable reject preserves baseline;
CUDA/communication faults or uncertain state restoration propagate as unsafe.
"""
import hashlib,importlib.util,json,os,sys,uuid
from pathlib import Path
class OptionalRejected(ValueError):pass
class UnsafePreparation(RuntimeError):pass

def source_attestation():
    manifest=json.loads(Path(__file__).with_name('attention-imports.json').read_text())
    component=Path(__file__).with_name('attention-component-receipt.json')
    if hashlib.sha256(component.read_bytes()).hexdigest()!=manifest['component_receipt_sha256']:
        raise OptionalRejected('Component numerical/layout receipt differs')
    for row in manifest['files']:
        module=sys.modules.get(row['module']);origin=getattr(module,'__file__',None)
        if origin is None:
            spec=importlib.util.find_spec(row['module']);origin=None if spec is None else spec.origin
        if origin is None or Path(origin).resolve()!=Path(row['target']).resolve():
            raise OptionalRejected('Unexpected import origin: '+row['module'])
        if hashlib.sha256(Path(origin).read_bytes()).hexdigest()!=row['sha256']:
            raise OptionalRejected('Source hash mismatch: '+row['module'])
    from .attention_profiles import next_profile
    required={'hybrid.attention_profiles','hybrid.attention_second_capture','hybrid.production_control'}
    if not required.issubset({row['module'] for row in manifest['files']}):
        raise OptionalRejected('Second-bank support source pins missing')
    profile=next_profile(manifest)
    if profile is not None:
        proof=Path(__file__).with_name('attention-next-component-receipt.json')
        if hashlib.sha256(proof.read_bytes()).hexdigest()!=manifest['next_attention']['component_receipt_sha256']:
            raise OptionalRejected('Next component receipt differs')
        data=json.loads(proof.read_text())
        if data.get('profile')!=profile.record() or data.get('eligible_on_all_ranks') is not True:
            raise OptionalRejected('Next profile lacks matching all-rank component proof')
    from .native_mixed_private import requested
    if requested(manifest):
        required={'hybrid.native_mixed_private','hybrid.attention_mixed_capture'}
        if not required.issubset({row['module'] for row in manifest['files']}):
            raise OptionalRejected('Private native source pins missing')
        if profile is None or profile.record()!=dict(splits=8,warps=8,direct=False):
            raise OptionalRejected('Mixed requires measured next 8/8/COPY profile')
    return manifest

async def initialize_owner(control):
    """Inside initialize, after baseline warmup and before HTTP admission."""
    if os.environ.get('W26_ATTENTION_BANK','0')!='1':return
    if getattr(control,'_w26_initial_attempted',False):raise UnsafePreparation('Repeated initial preparation')
    control._w26_initial_attempted=True
    if control.admission is None or control.admission.open or not await control.engine.is_paused():
        raise UnsafePreparation('Initial owner is not closed and paused')
    def disabled(reason):
        control.save('attention-initial.json',dict(ready=False,reason=reason,baseline_retained=True,production_enabled=False))
    if os.environ.get('W26_ATTENTION_SPLIT','0')!='1':disabled('split_flag_not_enabled');return
    try:source_attestation()
    except (ValueError,OSError,ImportError,AttributeError) as exc:disabled(repr(exc));return
    await control.arm('transpose')
    result=await control.collective('six')
    if result.get('outcome')!='committed':disabled('six_profile_rejected');return
    try:rows=await control.rpc('w26_attention_initial_prepare',('initial-'+uuid.uuid4().hex,),timeout=900)
    except Exception as exc:raise UnsafePreparation('Initial worker preparation failed or timed out') from exc
    if not isinstance(rows,list) or len(rows)!=4 or sorted(r.get('rank',-1) for r in rows)!=list(range(4)):
        raise UnsafePreparation('Missing or duplicate initial worker acknowledgements')
    if not all(r.get('ok') and r.get('baseline_retained') for r in rows):
        raise UnsafePreparation('Cannot attest healthy baseline on every rank')
    if all(r.get('ready') for r in rows) and len({r.get('identity_sha256') for r in rows})!=1:
        raise UnsafePreparation('Initial identity receipts disagree')
    if len({bool(r.get('ready')) for r in rows})!=1:
        await control.rpc('w26_attention_bank_abort',('initial readiness disagreed',))
        raise UnsafePreparation('Initial readiness disagreed')
    await control.qualify()
    control.save('attention-initial.json',dict(ready=all(r.get('ready') for r in rows),rows=rows,
        baseline_retained=True,selected_bank='baseline',production_enabled=False))

def _prepare_attention_worker(worker,owner_id):
    """Internal initial RPC only; marker and established drafter state required."""
    if os.environ.get('W26_ATTENTION_BANK','0')!='1':
        return dict(rank=int(worker.rank),ok=True,ready=False,baseline_retained=True,reason='default_off')
    import torch
    import torch.distributed as dist
    from vllm.distributed import get_tp_group,graph_capture
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from w9_exl_ab import runtime as original
    from .attention_contract import Identity,Qualification,AttentionControl
    from .attention_worker import WorkerAttention
    from .attention_qualification import qualify_replay,QualificationRejected
    from .initial_state_plan import collect,InitialJournal
    from .boot_native_contract import BootNativeContract
    from .w26_attention_split import AttentionState
    from .w26_dual_graph import Runtime,RUNTIMES,CAP_BYTES,FLOOR_BYTES,require_drafter_baseline
    runner=worker.model_runner;tp=get_tp_group();rt=None;journal=None;native=None;stage=0;roots=[]
    worker._w26_attention_roots=roots
    receipt=dict(rank=int(worker.rank),ok=True,ready=False,baseline_retained=True,production_enabled=False)
    def consensus(value):
        rows=[None]*tp.world_size;dist.all_gather_object(rows,value,group=tp.cpu_group);return rows
    def agree(value,reason):
        nonlocal stage
        stage+=1;tag=(stage,reason);rows=consensus(dict(phase=tag,ok=bool(value)))
        if not all(x.get('phase')==tag and x.get('ok') is True for x in rows):raise OptionalRejected(reason)
    def health():
        if torch.cuda.is_current_stream_capturing():raise UnsafePreparation('Capture did not end; no recovery claim')
        torch.cuda.synchronize()
        if native is not None:native.check_communication_health()
    def local_phase(name,fn):
        # CPU/generated local GPU or capture-only, never an all-rank replay.
        value=None;error=None
        try:
            value=fn()
            roots.append(value) # own successful local allocations BEFORE consensus can reject
        except Exception as exc:health();error=repr(exc)
        if error is not None:receipt.setdefault('local_errors',[]).append(dict(phase=name,error=error))
        agree(error is None,name);return value
    def restore():
        if journal is not None:
            journal.restore();rows=consensus(dict(restore=journal.verify()))
            if not all(x.get('restore') is True for x in rows):raise UnsafePreparation('State restoration/address identity failed')
    try:
        eligible=(not getattr(worker,'_w26_initial_attempted',False)
            and getattr(runner,'_w26_serving_seen',True) is False
            and type(runner.model) is BreakableCUDAGraphWrapper
            and runner.compilation_config.mode.name=='NONE' and tp.world_size==4
            and original.S.ready and original.S.startup_capture_context is not None
            and os.environ.get('W26_ATTENTION_SPLIT','0')=='1')
        worker._w26_initial_attempted=True;agree(eligible,'Initial-only baseline eligibility')
        manifest=local_phase('Pinned import/source attestation',source_attestation)
        agree(len({json.dumps(x,sort_keys=True) for x in consensus(manifest)})==1,'Source manifests equal')
        drafter=local_phase('Established baseline drafter',lambda:require_drafter_baseline(runner))
        before_free=torch.cuda.mem_get_info(runner.device)[0]
        agree(before_free>=CAP_BYTES+FLOOR_BYTES,'Private pool/journal reserve')
        native=local_phase('Boot native and six contract',lambda:BootNativeContract(runner));roots.append(native)
        from .metadata_census import inspect_initial_metadata
        receipt['initial_metadata']=local_phase('Non-mutating full metadata census',lambda:inspect_initial_metadata(runner))
        plan=local_phase('Exact initial state census',lambda:collect(runner,phase='initial_before_first_request'))
        journal=local_phase('Journal allocation/bindings',lambda:InitialJournal(plan));roots.append(journal)
        rt=local_phase('Separate private pool',lambda:Runtime(runner,None,native,before_free));roots.append(rt)
        rt.initial_consensus=consensus;rt.initial_metadata_records=[]
        receipt['actual_dummy_metadata']=rt.initial_metadata_records
        rt.attention_state=local_phase('Attention workspace',lambda:AttentionState(runner.device))
        probe=local_phase('Generated attention compile probe',rt.attention_state.compile_probe)
        local_phase('Post-probe resource bound',rt.guard)
        identity=Identity(owner_id,original.S.capture_generation,'transpose','boot',original.HASHES['transpose'],
            'six','incumbent',manifest['attention_source_sha256'],manifest['contract_sha256'])
        agree(len(set(consensus(identity.digest())))==1,'All-rank bank identity')
        controller=AttentionControl(identity)
        def current_identity():
            native.assert_baseline_unchanged()
            if require_drafter_baseline(runner)!=drafter:raise ValueError('Baseline drafter entries changed')
            return identity
        worker._w26_attention_adapter=WorkerAttention(worker,rt,controller,current_identity)
        RUNTIMES[id(rt.wrapper)]=rt
        with graph_capture(runner.device,original.S.startup_capture_context):
            restore();rt.mode='capture'
            # This dummy now contains an all-rank metadata guard. Never funnel
            # an arbitrary pre-guard exception into local_phase's next vote:
            # peers may still be entering that guard. Only its completed,
            # unanimous rejection is recoverable; unexpected failures are unsafe.
            from .initial_decode_contract import CompletedMetadataRejection
            try:rt.dummy('FULL',target_only=True)
            except CompletedMetadataRejection:raise
            except BaseException as exc:raise UnsafePreparation('Collective initial dummy/capture failed') from exc
            agree(True,'Private FULL capture completed on every rank')
            rt.mode='baseline';restore()
            nodes=local_phase('Private native/profile coverage',lambda:native.qualify_private(rt.entry))
            local_phase('Post-capture resource bound',rt.guard)
            # In-flight errors propagate; no unilateral next-arm replay.
            quality=qualify_replay(rt,journal,native,consensus)
        restore();health();local_phase('Final baseline identity',current_identity)
        memory=local_phase('Final physical bound',rt.guard)
        state=dict(replay=quality,journal_bytes=journal.gpu.bytes,memory=memory,bindings_verified=journal.verify(),drafter_entries_unchanged=True)
        state_sha=hashlib.sha256(json.dumps(state,sort_keys=True,allow_nan=False).encode()).hexdigest()
        agree(True,'All-rank preparation complete before publication')
        q=Qualification(source_layout_verified=True,resource_and_lifetime_verified=True,component_numerical_verified=True,
            model_state_restore_verified=True,native_collective_equivalent=True,all_rank_ready=True,
            numerical_receipt_sha256=manifest['component_receipt_sha256'],state_receipt_sha256=state_sha)
        if not controller.publish(rt.entry,q):raise UnsafePreparation('Publication rejected')
        # No profile is selected in this future overlay. Only a future frozen
        # manifest with matching component proof can request this extra bank.
        from .attention_profiles import next_profile
        from .attention_second_capture import second_runtime,capture_and_qualify
        profile=next_profile(manifest)
        next_receipt=dict(enabled=profile is not None,ready=False)
        if profile is not None:
            next_receipt['profile']=profile.record()
            try:
                before_second=local_phase('Before second attention global bound',rt.guard)
                pool=local_phase('Second private pool',torch.cuda.graph_pool_handle)
                with second_runtime(rt,pool,roots):
                    rt.attention_state=local_phase('Second attention workspace',lambda:AttentionState(runner.device,profile))
                    next_probe=local_phase('Second generated compile probe',rt.attention_state.compile_probe)
                    local_phase('Second post-probe global resource bound',rt.guard)
                    with graph_capture(runner.device,original.S.startup_capture_context):
                        next_nodes,next_quality=capture_and_qualify(rt,journal,native,consensus,restore,agree,local_phase,qualify_replay,UnsafePreparation)
                    restore();health();local_phase('Second baseline identity',current_identity)
                    next_memory=local_phase('Second final global physical bound',rt.guard)
                    next_state=dict(replay=next_quality,journal_bytes=journal.gpu.bytes,memory=next_memory,
                        before_second=before_second,bindings_verified=journal.verify(),drafter_entries_unchanged=True,
                        profile=profile.record(),attention_calls=rt.attention_calls)
                    agree(rt.attention_calls==38,'Second exact helper call coverage')
                    next_sha=hashlib.sha256(json.dumps(next_state,sort_keys=True,allow_nan=False).encode()).hexdigest()
                    next_q=Qualification(source_layout_verified=True,resource_and_lifetime_verified=True,component_numerical_verified=True,
                        model_state_restore_verified=True,native_collective_equivalent=True,all_rank_ready=True,
                        numerical_receipt_sha256=manifest['next_attention']['component_receipt_sha256'],state_receipt_sha256=next_sha)
                    local_phase('All-rank second publication preconditions',lambda:controller.validate_next(rt.entry,next_q,profile))
                    controller.publish_next(rt.entry,next_q,profile)
                    next_receipt.update(ready=True,state=next_state,probe=next_probe,native=next_nodes)
            except (OptionalRejected,QualificationRejected) as exc:
                # Completed, collectively classified rejection only. Unknown
                # capture/CUDA/communication failures escape to quarantine.
                rt.mode='baseline';restore();health()
                local_phase('Rejected second baseline identity',current_identity)
                # Failed candidates keep their graph/workspace roots. Identity
                # restoration does not undo their physical memory footprint.
                # local_phase votes rt.guard on every rank; do not let its
                # completed resource rejection reach the outer optional path.
                try:
                    retained_memory=local_phase('Rejected second retained resource bound',rt.guard)
                except OptionalRejected as retained_error:
                    raise UnsafePreparation('Retained second attention resources exceed the global bound') from retained_error
                next_receipt['retained_memory']=retained_memory
                next_receipt.update(reason=str(exc),candidate_disabled=True)
        from .attention_mixed_capture import prepare_mixed
        receipt['next_native_mixed']=prepare_mixed(worker,manifest,rt,journal,native,controller,roots,consensus,restore,agree,local_phase,health,current_identity,graph_capture,qualify_replay,OptionalRejected,UnsafePreparation)
        receipt['next_attention']=next_receipt
        receipt.update(ready=True,identity_sha256=identity.digest(),state=state,probe=probe,native=nodes,
            scope='BENCHMARK_ONLY; hidden diagnostics are not model quality',selected_bank='baseline')
    except (OptionalRejected,QualificationRejected) as exc:
        if rt is not None:rt.mode='baseline'
        restore();health()
        if native is not None:local_phase('Rejected candidate baseline identity',native.assert_baseline_unchanged)
        adapter=getattr(worker,'_w26_attention_adapter',None)
        if adapter is not None:adapter.abort(str(exc))
        receipt.update(reason=str(exc),candidate_disabled=True)
    except BaseException:
        receipt.update(ok=False,baseline_retained=False,unsafe=True)
        raise
    finally:
        if rt is not None:rt.mode='baseline'
        worker._w26_attention_roots=roots;worker._w26_attention_initial_receipt=receipt
    return receipt


def prepare_worker(worker,owner_id):
    # Initial RPC is outside the runner inference decorator. Own the context
    # across journals, input writes, candidate capture and cumulative sampler.
    import torch
    with torch.inference_mode():
        original_receipt = _prepare_attention_worker(worker,owner_id)
        if os.environ.get("W26_ATTENTION_BANK","0") != "1":
            return original_receipt
        from .cumulative_startup import extend_worker
        return extend_worker(worker,owner_id,original_receipt)
