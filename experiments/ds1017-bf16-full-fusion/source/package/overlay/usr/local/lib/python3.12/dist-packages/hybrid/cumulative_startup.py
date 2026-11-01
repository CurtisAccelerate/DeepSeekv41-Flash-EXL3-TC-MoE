"""Extend completed attention preparation; independent sampler availability.

Initial RPC only, no serving/lazy capture. This source is unqualified until its
actual-module scratch and future loaded gates pass; immutable attention-v2 is
not modified. Owner's existing initial pause/admission protocol is retained.
"""
from dataclasses import asdict
import hashlib,json,os
from pathlib import Path
from .cumulative_contract import FamilyIdentity,ArmControl,BASELINE,family_attention_receipts
from .cumulative_worker import WorkerCumulative

def extend_worker(worker,owner_id,attention_receipt):
    import torch,torch.distributed as dist
    from vllm.distributed import get_tp_group
    from w9_exl_ab import runtime as original
    from .attention_contract import Identity
    from .attention_startup import source_attestation
    from .boot_native_contract import BootNativeContract
    from .private_nodes import Rejected as NativeRejected
    from .w26_dual_graph import require_drafter_baseline
    from . import w26_sampler_bank as sampler
    runner=worker.model_runner;tp=get_tp_group();p=runner.drafter
    old=getattr(worker,'_w26_attention_adapter',None)
    roots=getattr(worker,'_w26_attention_roots',[]);worker._w26_cumulative_roots=roots
    def consensus(v):
        rows=[None]*tp.world_size;dist.all_gather_object(rows,v,group=tp.cpu_group);return rows
    error=None;manifest=None
    try:manifest=source_attestation()
    except (ValueError,OSError,ImportError,AttributeError) as exc:error=repr(exc)
    if any(x is not None for x in consensus(error)):
        if old is not None:old.abort('cumulative source attestation rejected')
        return dict(attention_receipt,ready=False,candidate_disabled=True,reason='cumulative source attestation rejected',baseline_retained=True)
    # Metadata/native proof can reject before optional sampler collectives.
    native=None;base=None;draft=None;error=None
    try:
        if getattr(runner,'_w26_serving_seen',True) is not False or getattr(worker,'_w26_cumulative_attempted',False):raise ValueError('Cumulative preparation requires first startup before any serving')
        if worker.parallel_config.pipeline_parallel_size!=1 or runner.compilation_config.mode.name!='NONE':raise ValueError('Unsupported initial runner/parallel mode')
        if tp.world_size!=4 or type(p).__name__!='DSparkProposer' or getattr(p,'_w26_capture_local',None) is not None:raise ValueError('Unsupported/legacy sampler')
        draft=require_drafter_baseline(runner)
        if old is not None:base=old.current_identity()
        else:
            native=BootNativeContract(runner);roots.append(native)
            base=Identity(owner_id,original.S.capture_generation,'transpose','boot',original.HASHES['transpose'],'six','incumbent',manifest['attention_source_sha256'],manifest['contract_sha256'])
    except (ValueError,AttributeError,TypeError,NativeRejected) as exc:error=repr(exc)
    if any(x is not None for x in consensus(error)):
        if old is not None:old.abort('cumulative baseline metadata rejected')
        return dict(attention_receipt,ready=False,candidate_disabled=True,reason='cumulative baseline metadata rejected',baseline_retained=True)
    worker._w26_cumulative_attempted=True
    state,sampler_receipt=sampler.prepare(p,tp.cpu_group,getattr(tp.device_communicator,'pynccl_comm',None))
    r2=None;r2_receipt=None
    if os.environ.get('DSV41_R2','0') not in ('','0'):
        from . import r2_startup
        _,(r2_start,r2_end)=sampler.eligibility(p)
        r2,r2_receipt=r2_startup.prepare(worker,p,old,consensus,r2_start,r2_end,p.model.logits_processor.org_vocab_size)
    # A sampler mismatch disables only sampler-containing arms. Existing
    # attention state/q stay strongly owned and independently selectable.
    module_sha=hashlib.sha256(Path(sampler.__file__).read_bytes()).hexdigest()
    contract_source=Path(__file__).with_name('cumulative_contract.py').read_bytes()
    draft_contract=dict(k=p.num_speculative_tokens,hidden=p.model.config.hidden_size,vocab=p.model.logits_processor.org_vocab_size,
        target_sampling='unchanged',draft_proposals='greedy; original full callback retained',control_source_sha256=hashlib.sha256(contract_source).hexdigest())
    attention_banks={}
    if old is not None:
        for key,q,profile in (('current',old.controller.qualification,None),
                              ('next',old.controller.next_qualification,old.controller.next_profile)):
            if q is not None and q.benchmark_ready():
                attention_banks[key]=dict(version='w26-compressed-t6-split16-v1' if profile is None else profile.version,
                    profile=None if profile is None else profile.record(),
                    numerical_receipt_sha256=q.numerical_receipt_sha256,state_receipt_sha256=q.state_receipt_sha256)
    mixed=getattr(worker,'_w35_mixed_bank',None)
    if mixed is not None:
        q=mixed['qualification'];profile=mixed['profile']
        attention_banks['next_mixed']=dict(version=profile.version,profile=profile.record(),native=mixed['component'],
            numerical_receipt_sha256=q.numerical_receipt_sha256,state_receipt_sha256=q.state_receipt_sha256)
    if r2 is not None:attention_banks['r2']=r2_startup.family_receipt(r2_receipt)
    attention_banks=family_attention_receipts(consensus(attention_banks))
    identity=FamilyIdentity(asdict(base),module_sha,hashlib.sha256(json.dumps(draft_contract,sort_keys=True).encode()).hexdigest(),attention_banks)
    if len(set(consensus(identity.digest())))!=1:raise RuntimeError('Cumulative all-rank source/config identity mismatch')
    controller=ArmControl(identity,None if old is None else old.controller.entry,
        None if old is None else old.controller.qualification,state,sampler_receipt,
        next_entry=None if old is None else old.controller.next_entry,
        next_qualification=None if old is None else old.controller.next_qualification,
        next_profile=None if old is None else old.controller.next_profile,mixed_bank=mixed)
    if len({json.dumps(x,sort_keys=True) for x in consensus(controller.availability())})!=1:
        raise RuntimeError('Cumulative component readiness differs across ranks')
    def current_identity():
        if identity.digest()!=controller._identity_digest:raise ValueError('Family identity contents changed')
        controller.assert_banks_intact()
        if controller.mixed_native is not None:controller.mixed_native.assert_intact()
        if old is not None:old.current_identity()
        else:native.assert_baseline_unchanged()
        if require_drafter_baseline(runner)!=draft:raise ValueError('Incumbent drafter graph bank changed')
        if getattr(p,'_w26_sampler_selector',None) is not adapter or getattr(p,'_w26_sampler_published_selector',None) is not adapter or getattr(worker,'_w26_attention_adapter',None) is not adapter:
            raise ValueError('Published sampler/worker adapter binding changed')
        if old is not None and old.runtime.attention_adapter is not adapter:raise ValueError('Published attention adapter binding changed')
        if getattr(p,'_w26_sampler_bank_state',None) is not controller.sampler_state:raise ValueError('Published sampler state binding changed')
        if state is not None:state.assert_intact()
        return identity
    adapter=WorkerCumulative(worker,None if old is None else old.runtime,controller,current_identity,p)
    worker._w26_attention_adapter=adapter;roots.extend((old,native,state,controller,adapter))
    if r2 is not None:
        from . import r2_bank
        r2.adapter=adapter;adapter.r2=r2;roots.append(r2);r2_bank.BANK=r2
    receipt=dict(rank=int(worker.rank),ok=True,ready=any(v for k,v in controller.availability().items() if k!=BASELINE),
        baseline_retained=True,selected_bank=BASELINE,identity_sha256=identity.arm_digest(BASELINE),family_sha256=identity.digest(),
        components=identity.arm_identity(BASELINE),available_arms=controller.availability(),sampler_preparation=sampler_receipt,
        attention_preparation=attention_receipt,r2_preparation=r2_receipt,production_enabled=False,scope='BENCHMARK_ONLY seven-arm extension; full model quality/performance untested')
    worker._w26_attention_initial_receipt=receipt
    if os.environ.get("STAR_ABC","0")=="1":
        from .abc_startup import prepare as abc_prepare  # STAR_ABC integration
        receipt["abc_preparation"]=abc_prepare(worker)
    return receipt
