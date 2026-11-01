"""Default-off optional prefill operator preparation after the original ABC bank."""
import hashlib
import os
import gc
import traceback
from pathlib import Path
from .prefill_contract import SUPPORTED, FEATURES, RAGGED_PROFILES, digest, sha256_string


def prepare(worker):
    if os.environ.get("STAR_PREFILL", "0") != "1":
        return dict(enabled=False,ready=False,default_profile="REFERENCE")
    # Deliberately no torch/vLLM/native import in the default-off branch.
    import torch
    import torch.distributed as dist
    from vllm.distributed import get_tp_group
    from vllm_exl3 import exl3
    from .prefill_runtime import Bank, NativeAdapter, binding
    from .prefill_qualification import qualify
    from .attention_startup import UnsafePreparation
    from .abc_kv_contract import requested_dtype
    if getattr(worker,"_prefill_attempted",False):
        raise UnsafePreparation("Repeated prefill startup preparation")
    worker._prefill_attempted=True
    tp=get_tp_group(); runner=worker.model_runner

    def consensus(row):
        values=[None]*tp.world_size
        dist.all_gather_object(values,row,group=tp.cpu_group)
        return values

    local=None; error=None; fatal=False
    combiner=None; dense=None; ragged=None; component_ok={"combine":False,"dense":False,"ragged":False}
    component_records={};component_errors={}
    try:
        if (tp.world_size!=4 or worker._abc_bank.selection.session is not None
                or getattr(runner,"_w26_serving_seen",True)):
            raise ValueError("Prefill preparation needs the closed initial TP4 ABC bank")
        maximum=runner.vllm_config.compilation_config.max_cudagraph_capture_size
        if maximum>48:
            raise ValueError("Prefill eager-only proof requires the retained <=48-row graph ladder")
        hook=exl3._tc_hook
        if hook is None or not callable(getattr(hook,'try_apply',None)):
            raise ValueError("Incumbent TC hook unavailable")
        original=hook.try_apply
        so=Path(os.environ.get("STAR_PREFILL_SO",""))
        expected=os.environ.get("STAR_PREFILL_SO_SHA256","")
        if not so.is_absolute() or not sha256_string(expected) or not so.is_file():
            raise ValueError("A pinned absolute prefill operator library is required")
        actual=hashlib.sha256(so.read_bytes()).hexdigest()
        if actual!=expected:
            raise ValueError("Prefill library hash mismatch")
        hook_sha=hashlib.sha256(Path(hook.__file__).read_bytes()).hexdigest()
        torch.ops.load_library(str(so))
        ops=torch.ops._dsv41_mm_try
        layers=[]; representatives={}
        for name,module in runner.get_model().named_modules():
            tc=getattr(module,"_exl3_tc",None)
            if tc is None:continue
            key=(tc["intermediate"],tc["bits_x2"])
            if key not in SUPPORTED or not tc["prefill"] or tc["prefill_rows"]!=2048 or tc["ptrs"].shape[0]!=384:
                raise ValueError("Incumbent layer geometry/format differs: "+name)
            if tc["workspace"].device!=runner.device:
                raise ValueError("Prefill workspace on wrong device")
            need=max(int(ops.moe_workspace_size_prefill_try(rows,5120,key[0],6,384))
                     for rows in (65,128,256,512,1024,2048))
            if need>tc["workspace"].numel():
                raise ValueError("Existing workspace cannot fit the candidate; no extra allocation allowed")
            layers.append((name,module,need));representatives.setdefault(key,module)
        if len(layers)!=40 or set(representatives)!=SUPPORTED:
            raise ValueError("Expected 40 prepared target layers and all four native formats")
        if len({m._exl3_tc["workspace"].data_ptr() for _,m,_ in layers})!=1:
            raise ValueError("Shared incumbent workspace ownership differs")
        stream=int(torch.cuda.current_stream(runner.device).cuda_stream)
        adapter=NativeAdapter(torch,ops)
        numerical=qualify(torch,original,adapter,representatives,runner.device)
        if original is not hook.try_apply:
            raise ValueError("TC hook changed during optional preparation")
        common=dict(schema="star.prefill.v1",operator_sha256=actual,hook_sha256=hook_sha,
                    cache_dtype=requested_dtype(),profiles=["REFERENCE","MOE_TRY"],
                    rows=[65,2048],formats=sorted(SUPPORTED),topk=6,experts=384,
                    qualifier_cases=numerical["cases"])
        local=dict(ok=True,rank=int(worker.rank),common=common,numerical=numerical,
                   layers=[dict(name=n,intermediate=m._exl3_tc["intermediate"],bits_x2=m._exl3_tc["bits_x2"],
                                needed_elements=need,existing_elements=m._exl3_tc["workspace"].numel()) for n,m,need in layers])
    except Exception as exc:
        error=repr(exc)
        try:torch.cuda.synchronize(runner.device)
        except Exception as cuda_exc:fatal=True;error+="; CUDA health: "+repr(cuda_exc)
    # New experiments reject independently; the already-qualified MoE survives
    # a nonfatal rejection. No candidate wrapper is published before consensus.
    if local is not None:
        try:
            from vllm.models.deepseek_v4_1.amd import rocm
            from .prefill_combine import combiner as combine_module
            from .prefill_combine.qualification import qualify as qualify_combine
            retained=getattr(rocm,'_star_combine_reference',None)
            if not callable(retained) or rocm._SPARSE_PREFILL_TOPK_ALIGNMENT!=combine_module.ALIGNMENT:
                raise ValueError('Actual retained combiner API/alignment unavailable')
            component_records['combine']=qualify_combine(torch,retained,combine_module.combine_topk_swa_indices,runner.device)
            combiner=combine_module;component_ok['combine']=True
        except Exception as exc:
            component_errors['combine']=repr(exc)
            try:torch.cuda.synchronize(runner.device)
            except Exception as cuda_exc:fatal=True;component_errors['combine']+='; CUDA health: '+repr(cuda_exc)
            traceback.clear_frames(exc.__traceback__);exc.__traceback__=None
        if not fatal:
            try:gc.collect();torch.cuda.empty_cache()
            except Exception as cleanup_exc:
                fatal=True;component_errors['combine_cleanup']=repr(cleanup_exc)
        dense_baseline=None
        try:
            if fatal:raise RuntimeError('Dense preparation skipped after unsafe CUDA failure')
            from .prefill_dense import prepare as prepare_dense
            from .prefill_interop import baseline as interop_baseline,qualify as interop_qualify,UnsafeWorkspace
            shared=layers[0][1]._exl3_tc['workspace']
            dense_baseline=interop_baseline(torch,original,representatives,runner.device)
            dense,component_records['dense']=prepare_dense(torch,runner.get_model(),shared,runner.device)
            component_records['dense']['moe_interop']=interop_qualify(torch,original,adapter,representatives,runner.device,dense_baseline,dense)
            component_ok['dense']=bool(component_records['dense']['passed'])
        except Exception as exc:
            component_errors['dense']=repr(exc);dense=None
            if type(exc).__name__=='UnsafeWorkspace':fatal=True
            try:torch.cuda.synchronize(runner.device)
            except Exception as cuda_exc:fatal=True;component_errors['dense']+='; CUDA health: '+repr(cuda_exc)
            traceback.clear_frames(exc.__traceback__);exc.__traceback__=None
            if not fatal:
                try:gc.collect();torch.cuda.empty_cache()
                except Exception as cleanup_exc:
                    fatal=True;component_errors['dense_cleanup']=repr(cleanup_exc)
                if not fatal and dense_baseline is not None:
                    try:component_records['dense_rejection_interop']=interop_qualify(torch,original,adapter,representatives,runner.device,dense_baseline)
                    except Exception as restore_exc:
                        fatal=True;component_errors['dense']+='; unsafe retained scratch: '+repr(restore_exc)
        if not fatal:
            try:
                from vllm.models.deepseek_v4_1.amd import rocm
                from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse_ops
                from .prefill_ragged import ragged as ragged_module
                from .prefill_ragged.qualification import qualify as qualify_ragged
                from .prefill_ragged.attention_qualification import qualify as qualify_ragged_attention
                if rocm.rocm_sparse_attn_prefill is not sparse_ops.rocm_sparse_attn_prefill:
                    raise ValueError('Actual prefill attention alias differs from qualified function')
                retained=rocm._star_combine_reference
                rec=qualify_ragged(torch,retained,sparse_ops.build_ragged_indices_from_dense,
                                  ragged_module.combine_ragged,runner.device,
                                  config_path=Path('/model-view/config.json'))
                rec['attention']=qualify_ragged_attention(torch,retained,ragged_module.combine_ragged,
                                                         sparse_ops.rocm_sparse_attn_prefill,runner.device)
                if rec['cases']!=85 or rec['attention']['cases']!=9 or not rec['attention']['passed']:
                    raise ValueError('Incomplete integer/attention qualification')
                component_records['ragged']=rec;ragged=ragged_module;component_ok['ragged']=True
            except Exception as exc:
                component_errors['ragged']=repr(exc)
                try:torch.cuda.synchronize(runner.device)
                except Exception as cuda_exc:fatal=True;component_errors['ragged']+='; CUDA health: '+repr(cuda_exc)
                traceback.clear_frames(exc.__traceback__);exc.__traceback__=None
            if not fatal:
                try:gc.collect();torch.cuda.empty_cache()
                except Exception as cleanup_exc:fatal=True;component_errors['ragged_cleanup']=repr(cleanup_exc)
        local['components']=component_records;local['component_errors']=component_errors
    votes=consensus(dict(ok=local is not None,error=error,fatal=fatal,rank=int(worker.rank),component_ok=component_ok,
                         common=None if local is None else local["common"],
                         numerical_digest=None if local is None else digest(dict(moe=local["numerical"],components=component_records,errors=component_errors))))
    if len(votes)!=4 or sorted(v["rank"] for v in votes)!=list(range(4)) or any(v["fatal"] for v in votes):
        raise UnsafePreparation("Unsafe or missing prefill startup ranks: "+str(votes))
    if not all(v["ok"] for v in votes):
        # No wrapper installed on any rank. Native library may remain resident but unused.
        return dict(enabled=True,ready=False,default_profile="REFERENCE",rank_votes=votes,
                    reason="Optional candidate rejected; incumbent hook/graphs retained",model_quality_verified=False)
    if len({digest(v["common"]) for v in votes})!=1:
        raise UnsafePreparation("Prefill format/library/cache identity differs across ranks")
    combine_ok=all(v['component_ok']['combine'] for v in votes)
    dense_ok=all(v['component_ok']['dense'] for v in votes)
    ragged_ok=all(v['component_ok']['ragged'] for v in votes)
    available=[name for name,(_,needs_combine,needs_dense) in FEATURES.items()
               if (not needs_combine or combine_ok) and (needs_dense is None or dense_ok)
               and (name not in RAGGED_PROFILES or ragged_ok)]
    common=dict(local['common'],profiles=available,qualified_components=dict(moe=True,combine=combine_ok,dense=dense_ok,ragged=ragged_ok))
    local['common']=common;local['rank_component_votes']=[dict(rank=v['rank'],ready=v['component_ok']) for v in votes]
    identity=digest(dict(common=common,rank_qualification=[v["numerical_digest"] for v in votes]))

    def mode():
        from vllm.forward_context import get_forward_context
        try:return get_forward_context().cudagraph_runtime_mode.name
        except (AssertionError,RuntimeError):return "UNKNOWN"

    bank=Bank(worker,original,torch,adapter,{id(m):binding(m._exl3_tc) for _,m,_ in layers},
              stream,mode,identity,local,available=tuple(available),
              combiner=combiner if combine_ok else None,dense=dense if dense_ok else None,ragged=ragged if ragged_ok else None)
    worker._w26_cumulative_roots.append(bank)
    worker._prefill_bank=bank
    from . import prefill_runtime
    prefill_runtime.BANK=bank
    hook.try_apply=bank.apply
    installed=consensus(dict(ok=hook.try_apply==bank.apply,identity=identity,rank=int(worker.rank)))
    if not all(v["ok"] and v["identity"]==identity for v in installed):
        raise UnsafePreparation("Prefill wrapper installation differs across ranks")
    # No candidate is selected here; every wrapper delegates to the retained original.
    return dict(enabled=True,ready=True,identity=identity,default_profile="REFERENCE",
                numerical=local["numerical"],available=available,qualified_components=common['qualified_components'],
                component_errors=component_errors,model_quality_verified=False,performance_measured=False)
