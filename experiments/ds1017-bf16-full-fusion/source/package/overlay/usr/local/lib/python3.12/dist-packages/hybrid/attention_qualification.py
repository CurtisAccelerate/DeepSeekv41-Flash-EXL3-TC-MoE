"""Attention BENCHMARK-ONLY hidden-output checks, not model-quality approval.

Cross-arm hidden differences are diagnostics without an invented tolerance.
Each arm must replay itself exactly after changed/restored inputs. This module
does not capture graphs, load weights, select a serving bank, or alter mHC gates.
"""
class QualificationRejected(ValueError):pass

def tensors(value,path='output'):
    import torch
    if isinstance(value,torch.Tensor):return [(path,value)]
    if isinstance(value,(tuple,list)):
        return sum((tensors(v,f'{path}/{i}') for i,v in enumerate(value)),[])
    if isinstance(value,dict):
        return sum((tensors(value[k],f'{path}/{k}') for k in sorted(value)),[])
    raise QualificationRejected('Unsupported model output structure at '+path)

def structure(value):
    import torch
    if isinstance(value,torch.Tensor):return ('tensor',tuple(value.shape),str(value.dtype),str(value.device))
    if isinstance(value,(tuple,list)):return (type(value).__name__,tuple(structure(v) for v in value))
    if isinstance(value,dict):return ('dict',tuple((k,structure(value[k])) for k in sorted(value)))
    raise QualificationRejected('Unsupported model output structure')

def clone(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().clone()
    if isinstance(value,tuple):return tuple(clone(x) for x in value)
    if isinstance(value,list):return [clone(x) for x in value]
    if isinstance(value,dict):return {k:clone(v) for k,v in value.items()}
    raise QualificationRejected('Unsupported model output structure')

def finite(value):
    import torch
    return all(torch.isfinite(x).all().item() for _,x in tensors(value))

def exact(a,b):
    import torch
    return structure(a)==structure(b) and all(torch.equal(x,y) for (_,x),(_,y) in zip(tensors(a),tensors(b)))

def hidden_diagnostics(a,b):
    import torch
    if structure(a)!=structure(b):raise QualificationRejected('Hidden output structure/shape/dtype/device differs')
    if not finite(a) or not finite(b):raise QualificationRejected('Nonfinite model hidden output')
    rows=[]
    for (name,x),(_,y) in zip(tensors(a),tensors(b)):
        # Small hidden outputs only, never weight/KV payloads. CPU FP64 keeps
        # diagnostic allocations outside the graph allocator pools.
        xd=x.detach().to(device='cpu',dtype=torch.float64);yd=y.detach().to(device='cpu',dtype=torch.float64)
        delta=yd-xd;an=float(xd.norm());bn=float(yd.norm());dn=float(delta.norm())
        rows.append(dict(path=name,maxabs=float(delta.abs().max()) if delta.numel() else 0.0,
            baseline_norm=an,candidate_norm=bn,difference_norm=dn,relative_l2=dn/an if an else None,
            norm_ratio=bn/an if an else None,bit_exact=torch.equal(x,y)))
    return dict(scope='hidden diagnostics only; NOT model quality',outputs=rows)

def qualify_replay(rt,journal,native,consensus):
    """Already-built graphs only. Every rank must call in identical order.

    consensus(record) returns all rank records using the CPU group. Preparation and
    restore checks happen before each GPU collective phase. An in-flight CUDA/
    communicator exception propagates; it is never converted to an optional
    local rejection followed by unilateral next-arm execution.
    """
    from .private_nodes import Rejected
    import hashlib
    torch=rt.torch;records=[];input_records=[];first={};last={};unsafe_failure=False;phase_index=0
    def agree(value,reason):
        nonlocal phase_index
        phase_index+=1;tag=(phase_index,reason)
        votes=consensus(dict(phase=tag,ok=bool(value)))
        if not all(v.get('phase')==tag and v.get('ok') is True for v in votes):raise QualificationRejected(reason)
    def run(mode,seed):
        n=getattr(rt,'rows',6)
        journal.restore();agree(journal.verify(),'State bytes/addresses did not restore before replay')
        local_ok=True
        try:native.assert_baseline_unchanged()
        except (ValueError,Rejected):local_ok=False
        agree(local_ok,'Native/collective identity precondition changed')
        rt.runner.input_ids.gpu[:n].fill_(seed)
        if rt.runner.supports_mm_inputs:
            embedding=rt.runner.inputs_embeds.gpu[:n]
            generator=torch.Generator(device='cpu');generator.manual_seed(seed)
            generated=torch.randn(tuple(embedding.shape),generator=generator,device='cpu',dtype=torch.float32).mul_(0.125)
            embedding.copy_(generated.to(dtype=embedding.dtype)) # non-collinear actual input, no global RNG mutation
        inputs={name:hashlib.sha256(value.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            for name,value in (('raw_ids',rt.runner.input_ids.gpu[:n]),('embeddings',rt.runner.inputs_embeds.gpu[:n]))}
        input_records.append(dict(seed=seed,mode=mode,sha256=inputs))
        rt.mode=mode
        output=clone(rt.dummy('FULL',target_only=True))
        torch.cuda.synchronize();native.check_communication_health()
        agree(finite(output),'Nonfinite output on a rank')
        memory_ok=True
        try:rt.guard()
        except ValueError:memory_ok=False
        agree(memory_ok,'Per-rank candidate memory cap failed')
        return output
    try:
        # A -> changed B -> restored A; each arm is also repeated immediately.
        for index,seed in enumerate((17,23,17)):
            current={}
            for arm,mode in (('baseline','baseline'),('candidate','qualify_replay')):
                a=run(mode,seed);b=run(mode,seed)
                agree(exact(a,b),'Own-capture repeatability failed: '+arm)
                current[arm]=a
                if index==0:first[arm]=a
                if index==2:last[arm]=a
            agree(structure(current['baseline'])==structure(current['candidate']),'Cross-arm structure differs')
            records.append(dict(seed=seed,own_replays_exact=True,hidden=hidden_diagnostics(current['baseline'],current['candidate'])))
        agree(all(exact(first[k],last[k]) for k in first),'Restored-input replay differed from original input')
        native_ok=True
        try:native.assert_baseline_unchanged()
        except (ValueError,Rejected):native_ok=False
        agree(native_ok,'Final native/collective identity changed')
        agree(all(len({x['sha256'][k] for x in input_records if x['seed']==seed})==1 for seed in (17,23) for k in ('raw_ids','embeddings'))
            and all(len({x['sha256'][k] for x in input_records})==2 for k in ('raw_ids','embeddings')),'Actual input byte variation/restoration failed')
        return dict(benchmark_component_replay_pass=True,hidden_metrics=records,actual_input_records=input_records,model_quality_verified=False,production_enabled=False)
    except QualificationRejected:
        raise
    except BaseException:
        # No speculative cleanup copies or mismatched consensus after an
        # unknown CUDA/communicator fault. Owner must remain paused/uncertain.
        unsafe_failure=True;raise
    finally:
        rt.mode='baseline'
        # A failure here is not a claim that the baseline is healthy. The owner
        # must stay paused/quarantined if exact restore cannot be established.
        if not unsafe_failure:
            journal.restore()
            agree(journal.verify(),'Final state restoration failed; keep owner paused')
