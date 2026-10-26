"""W26 same-load B1/K5 sampler extension; default OFF, never a hot patch.

prepare() must run collectively after loaded weights and NCCL are ready, before
serving. It is deliberately not called lazily by sample_or_fallback(). Initial numerical/runtime checks passed; limited matched task tests had mixed quality;
broader numerical accuracy and task-quality verification is still warranted. enabled() controls PREPARATION ONLY. The caller explicitly selects a prepared state for sampling. Target temperature, rejection sampler and target RNG remain
untouched. The replicated full/local Markov arithmetic derives from v3r3.
"""
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import threading

VERSION='captured-local-nccl-b1k5-v4'
class OptionalRejected(ValueError):pass
class UnsafeSamplerPreparation(RuntimeError):pass


def enabled():
    return os.environ.get('W26_SAMPLER_BANK', '0') == '1'


def eligibility_report(p):
    m=p.model; lp=m.logits_processor; h=m.lm_head; w=m.model.markov_head.markov_w2.weight; s=h.shard_indices
    confidence=p._conf_threshold()
    predicates=dict(depth5=p.num_speculative_tokens==5,topk_none=p._draft_topk is None,
        confidence_zero=confidence==0,head_no_cast_BF16=lp.head_dtype in (None,torch.bfloat16),
        scale_one=lp.scale==1.0,soft_cap_none=lp.soft_cap is None,logits_as_input_false=not lp.logits_as_input,
        all_gather=bool(lp.use_all_gather),head_tp4=h.tp_size==4,
        identity_draft_ids=getattr(m,'draft_id_to_target_id',None) is None,
        no_added_vocab=getattr(s,'num_added_elements',None)==0,
        no_original_padding=getattr(s,'num_org_vocab_padding',None)==0,
        vocab129280=lp.org_vocab_size==129280,markov_shape=tuple(w.shape)==(129280,256),
        markov_BF16=w.dtype==torch.bfloat16,markov_contiguous=w.is_contiguous(),
        shard32320=s.org_vocab_end_index-s.org_vocab_start_index==32320)
    values=dict(depth=p.num_speculative_tokens,draft_topk=p._draft_topk,confidence=confidence,
        head_dtype=str(lp.head_dtype),qualified_hidden_dtype=str(torch.bfloat16),scale=lp.scale,soft_cap=lp.soft_cap,
        logits_as_input=lp.logits_as_input,use_all_gather=lp.use_all_gather,head_tp=h.tp_size,
        has_draft_id_map=getattr(m,'draft_id_to_target_id',None) is not None,
        num_added_elements=getattr(s,'num_added_elements',None),num_org_vocab_padding=getattr(s,'num_org_vocab_padding',None),
        org_vocab=lp.org_vocab_size,markov_shape=list(w.shape),markov_dtype=str(w.dtype),markov_contiguous=w.is_contiguous(),
        shard_start=s.org_vocab_start_index,shard_end=s.org_vocab_end_index)
    return dict(eligible=all(predicates.values()),predicates=predicates,values=values,
        failed_predicates=[k for k,v in predicates.items() if not v])


def eligibility(p):
    report=eligibility_report(p);v=report['values']
    return report['eligible'],(v['shard_start'],v['shard_end'])


class CapturedLocal:
    def __init__(self,p,nccl,start,end):
        self.p=p; self.nccl=nccl; self.start=start; self.end=end
        self.weight=p.model.model.markov_head.markov_w2.weight
        self.weight_pointer=self.weight.data_ptr()
        self.local_weight=self.weight[start:end]
        self.base=torch.zeros((1,5,32320),dtype=torch.bfloat16,device=self.weight.device)
        self.anchor=torch.zeros(1,dtype=torch.int64,device=self.weight.device)
        self.graph=None; self.output=None; self.ready=False
        self.serving_stream=torch.cuda.current_stream().cuda_stream

    def gather(self,x):
        # Exact PyNccl output layout used by the qualified scratch fixture.
        y=torch.empty((x.shape[0]*4,)+x.shape[1:],device=x.device,dtype=x.dtype)
        self.nccl.all_gather(y,x.contiguous())
        return y.reshape((4,)+x.shape).movedim(0,x.ndim-1).reshape(x.shape[:-1]+(x.shape[-1]*4,))

    def chain(self,full=False):
        base=self.gather(self.base) if full else self.base
        prev=self.anchor; result=[]
        for i in range(5):
            embed=self.p.model.markov_embed(prev)
            bias=self.p.model.markov_bias(embed) if full else F.linear(embed,self.local_weight)
            logits=base[:,i]+bias
            if full:
                prev=logits.argmax(-1)
            else:
                value,index=logits.max(-1)
                pair=torch.stack((value.float(),(index+self.start).float()),-1)
                pairs=self.gather(pair).reshape(1,4,2)
                winner=pairs[:,:,0].argmax(-1,keepdim=True)
                prev=pairs[:,:,1].to(torch.int64).gather(-1,winner).squeeze(-1)
            result.append(prev)
        return torch.stack(result,-1)

    def sample(self,hidden):
        p=self.p; m=p.model
        assert self.ready and self.weight.data_ptr()==self.weight_pointer
        assert torch.cuda.current_stream().cuda_stream==self.serving_stream, 'Unqualified concurrent/cross-stream sampling'
        # Retain the existing norm/local head arithmetic and shape. Only the
        # gather is skipped. Head GEMM is outside the captured sampling chain.
        base=m.logits_processor(m.lm_head,m.model.norm(hidden),skip_gather=True)
        assert base.dtype==torch.bfloat16 and tuple(base.shape)==(5,32320)
        self.base.copy_(base.view(1,5,32320))
        self.anchor.copy_(p.input_ids[p._anchor_idx[:1]])
        self.graph.replay()
        # Proposer output outlives this scratch output across verification.
        # Copy on the current stream before the next invocation may overwrite.
        return self.output.clone()


class BankSampler(CapturedLocal):
    """Own copied outputs and serialize static buffers across caller streams."""
    def __init__(self,*args):
        super().__init__(*args)
        self.lock=threading.Lock();self.previous=torch.cuda.Event()
        self.previous.record(torch.cuda.current_stream())
        self.weights=tuple((v,v.data_ptr()) for v in (
            self.weight,self.p.model.model.markov_head.markov_w1.weight,
            self.p.model.lm_head.weight,self.p.model.model.norm.weight))
        self.communicator=self.nccl
    def assert_intact(self):
        if not self.ready or self.nccl is not self.communicator or not self.nccl.available or self.nccl.disabled:
            raise RuntimeError('Prepared sampler communicator/readiness changed')
        current=(self.p.model.model.markov_head.markov_w2.weight,self.p.model.model.markov_head.markov_w1.weight,self.p.model.lm_head.weight,self.p.model.model.norm.weight)
        if any(now is not old or now.data_ptr()!=ptr for now,(old,ptr) in zip(current,self.weights)):raise RuntimeError('Prepared sampler weight binding moved')
    def sample(self,hidden):
        self.assert_intact()
        if not self.lock.acquire(blocking=False):raise RuntimeError('Concurrent host sampler invocation unsupported')
        try:
            current=torch.cuda.current_stream()
            base=self.p.model.logits_processor(self.p.model.lm_head,self.p.model.model.norm(hidden),skip_gather=True)
            if base.dtype!=torch.bfloat16 or tuple(base.shape)!=(5,32320):raise RuntimeError('Unqualified head shape/dtype')
            current.wait_event(self.previous)
            self.base.copy_(base.view(1,5,32320));self.anchor.copy_(self.p.input_ids[self.p._anchor_idx[:1]])
            self.graph.replay();owned=self.output.clone()
            self.previous.record(current)
            return owned
        finally:self.lock.release()


@torch.inference_mode()
def prepare(p,cpu_group,nccl,max_growth_mib=96):
    """Initial-only caller supplies all-rank ordering and holds private admission.

    No shared communicator is created/aborted/destroyed. Exceptions after entry
    into collective/capture are UNSAFE: copied PyNccl has no async-health API.
    Completed numerical mismatches and precollective metadata/resource errors
    disable the optional sampler on all ranks; attention availability is separate.
    """
    assert not torch.cuda.is_current_stream_capturing()
    world=dist.get_world_size(cpu_group);state=None;roots=[];records=[];anchor_saved=None;anchor_index=None;communication_started=False;descriptions=[]
    def consensus(value):
        rows=[None]*world;dist.all_gather_object(rows,value,group=cpu_group);return rows
    def agree(value,reason):
        if not all(consensus(bool(value))):raise OptionalRejected(reason)
    def local(name,fn):
        value=None;error=None
        try:value=fn();roots.append(value)
        except OptionalRejected as exc:error=repr(exc)
        except (AttributeError,TypeError,ValueError) as exc:
            if name!='Sampler metadata':raise UnsafeSamplerPreparation('Unexpected local phase failure '+name) from exc
            error=repr(exc)
        except torch.OutOfMemoryError as exc:
            if communication_started:raise UnsafeSamplerPreparation('Allocation failure after shared communication began') from exc
            torch.cuda.synchronize();error=repr(exc)
        except BaseException as exc:raise UnsafeSamplerPreparation('Unsafe local CUDA/execution phase '+name) from exc
        agree(error is None,name+': '+str(error));return value
    def collective(name,fn):
        try:return fn()
        except BaseException as exc:raise UnsafeSamplerPreparation('Uncertain collective/capture phase '+name) from exc
    def resources():
        torch.cuda.synchronize();free=torch.cuda.mem_get_info()[0]
        return before-free<=max_growth_mib*1024**2 and free>=1024**3
    def restore():
        if anchor_saved is not None:
            try:
                p.input_ids.index_copy_(0,anchor_index,anchor_saved);torch.cuda.synchronize()
                restored=consensus(torch.equal(p.input_ids[anchor_index],anchor_saved))
            except BaseException as exc:raise UnsafeSamplerPreparation('Cannot prove proposer anchor restoration') from exc
            if not all(restored):raise UnsafeSamplerPreparation('Proposer anchor exact restoration failed')
    try:
        flags=consensus(enabled())
        if not all(flags):return None,dict(ready=False,reason='default_off_or_flag_mismatch',incumbent_retained=True)
        agree(not getattr(p,'_w26_sampler_prepare_attempted',False),'Sampler preparation is initial-only once')
        p._w26_sampler_prepare_attempted=True
        p._w26_sampler_roots=roots
        def metadata():
            report=eligibility_report(p);v=report['values']
            communicator=dict(present=nccl is not None,available=bool(getattr(nccl,'available',False)),disabled=bool(getattr(nccl,'disabled',True)),world=world)
            return ((report['eligible'],(v['shard_start'],v['shard_end'])),communicator['present'] and communicator['available'] and not communicator['disabled'],hasattr(p,'_sample_sequential_incumbent'),report,communicator)
        description=local('Sampler metadata',metadata)
        descriptions=consensus(description)
        agree(world==4 and all(x[0][0] and x[1] and x[2] for x in descriptions)
            and [tuple(x[0][1]) for x in descriptions]==[(i*32320,(i+1)*32320) for i in range(4)],'Unsupported loaded sampler or communicator')
        torch.cuda.synchronize();before=torch.cuda.mem_get_info()[0]
        agree(before-max_growth_mib*1024**2>=1024**3,'Sampler preallocation floor')
        state=local('Static sampler allocation',lambda:BankSampler(p,nccl,*description[0][1]))
        anchor_index=local('Anchor index journal',lambda:p._anchor_idx[:1].clone())
        anchor_saved=local('Anchor value journal',lambda:p.input_ids[anchor_index].clone())
        hidden=local('Hidden fixture allocation',lambda:torch.empty((5,p.model.config.hidden_size),device=state.weight.device,dtype=torch.bfloat16))
        stream=local('Private capture stream',torch.cuda.Stream)
        local('Resource reserve',lambda:None if resources() else (_ for _ in ()).throw(OptionalRejected('Sampler budget')))
        # Precollective shape-specific full/local projection comparison using
        # loaded weights. No changes to weights, target sampling or target RNG.
        for seed in (1017,991,7043):
            def projection():
                state.anchor.fill_(seed);embed=p.model.markov_embed(state.anchor)
                return torch.equal(p.model.markov_bias(embed)[:,state.start:state.end],F.linear(embed,state.local_weight))
            agree(local('Loaded Markov local arithmetic',projection),'Loaded Markov full/local bitwise mismatch')
        stream.wait_stream(torch.cuda.current_stream())
        def warm():
            with torch.cuda.stream(stream):
                for _ in range(2):state.chain()
            stream.synchronize()
        communication_started=True
        collective('warmup',warm);dist.barrier(group=cpu_group)
        state.graph=torch.cuda.CUDAGraph(keep_graph=True)
        def capture():
            with torch.cuda.graph(state.graph,stream=stream):state.output=state.chain()
        collective('raw capture',capture)
        # Capture completed on all ranks before instantiate/replay. Unlike
        # numerical rejects, capture/communicator exceptions are not recovered.
        agree(True,'Raw capture ended on all ranks')
        collective('Graph instantiate',state.graph.instantiate)
        agree(local('Postcapture resources',resources),'Postcapture sampler budget')
        state.ready=True
        held=None;held_copy=None;first=None
        for seed in (17,23,17):
            def fill():
                generator=torch.Generator(device=hidden.device).manual_seed(seed)
                hidden.copy_(torch.randn(hidden.shape,device=hidden.device,dtype=hidden.dtype,generator=generator))
                p.input_ids.index_fill_(0,anchor_index,seed)
            local('Whole-call generated inputs',fill)
            full_base=collective('incumbent norm/head logits',lambda:p.model.compute_draft_logits(hidden))
            local_base=local('candidate norm/head logits',lambda:p.model.logits_processor(p.model.lm_head,p.model.model.norm(hidden),skip_gather=True))
            agree(torch.equal(full_base[:,state.start:state.end],local_base),'Loaded norm/head projection mismatch')
            reference=collective('incumbent whole draft call',lambda:p._sample_sequential_incumbent(1,hidden))
            actual=collective('candidate whole draft call',lambda:state.sample(hidden))
            torch.cuda.synchronize()
            exact=torch.equal(reference,actual)
            if held is not None:exact=exact and torch.equal(held,held_copy)
            if first is not None and seed==17:exact=exact and torch.equal(actual,first)
            agree(exact,'Loaded hidden whole-call/reuse/restoration draft IDs mismatch')
            if first is None:first=actual.clone();roots.append(first)
            held=actual;held_copy=actual.clone();roots.extend((actual,held_copy))
            records.append(dict(seed=seed,whole_call_bitwise=True,returned_output_owned=True))
        restore();agree(local('Final sampler resources',resources),'Final sampler resource gate')
        state.previous.record(torch.cuda.current_stream());torch.cuda.synchronize()
        receipt=dict(ready=True,eligibility_descriptions=descriptions,version=VERSION,all_rank_ready=True,whole_call_exact=True,projection_exact=True,norm_head_projection_exact=True,
            changed_input_records=records,anchor_restored=True,physical_growth=before-torch.cuda.mem_get_info()[0],
            target_sampling_unchanged=True,production_qualified=False,model_task_qualified=False)
        p._w26_sampler_bank_state=state
        return state,receipt
    except OptionalRejected as exc:
        restore()
        if state is not None:state.ready=False
        return None,dict(ready=False,candidate_disabled=True,incumbent_retained=True,reason=str(exc),records=records,eligibility_descriptions=descriptions)
    except BaseException:
        # Retain roots for diagnosis; never abort or replace the shared NCCL.
        # The initial owner must treat this as unsafe instead of reporting a
        # healthy baseline when CUDA/communication state is uncertain.
        raise


def sample_or_fallback(state,p,num_reqs,hidden,fallback):
    """Explicit selected-state dispatch; pass None for the reference arm.

    All ranks must choose the same state/arm before entering this call.
    The flag gates preparation, not the target sampler or speculative depth.
    """
    if state is None:return fallback(num_reqs,hidden)
    if (num_reqs!=1 or p.num_speculative_tokens!=5 or p._draft_topk is not None
        or p._conf_threshold()!=0 or os.environ.get('W26_OBSERVE_MARGINS','0')=='1'
        or tuple(hidden.shape)!=(5,p.model.config.hidden_size) or hidden.dtype!=torch.bfloat16
        or torch.cuda.is_current_stream_capturing()):
        return fallback(num_reqs,hidden)
    return state.sample(hidden)
