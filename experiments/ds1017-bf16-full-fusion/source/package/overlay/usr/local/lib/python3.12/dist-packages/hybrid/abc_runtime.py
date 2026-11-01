"""Precaptured ABC replay only. No lazy allocation/capture or environment mutation."""
from collections import Counter
from .abc_contract import PROFILES, descriptor, family_key

BANK=None
CAPTURE=None

def context(): return CAPTURE
def current(): return None if BANK is None or BANK.selection.session is None else PROFILES[BANK.selection.selected]
def target_feature(name):
    c=CAPTURE
    return c is not None and bool(getattr(c.profile,name,False))

def reference_sampler(bank,b):
    state=bank.reference_sampler if b==1 else bank.r2.greedy.get(b)
    if state is None or not callable(getattr(state,'sample',None)):
        raise ValueError('Qualified original sampler missing for B'+str(b))
    return state

class Bank:
    def __init__(self,worker,reference,reference_sampler,r2):
        self.worker=worker;self.runner=worker.model_runner;self.reference=reference
        self.reference_sampler=reference_sampler;self.r2=r2;self.targets={};self.samplers={}
        self.sampled={};self.receipts={};self.selection=None;self.hits=Counter();self.gate=None
        self.routes=None
    def reference_entry(self,b,k):
        if k!=5:return None
        if b==1:return self.reference
        return None if self.r2 is None else self.r2.entries.get(descriptor(8 if b==7 else b))
    def status(self):
        return dict(rank=int(self.worker.rank),ok=True,**self.selection.record(),hits=dict(self.hits),
                    targets=[list(k) for k in self.targets],receipt={k:self.receipts.get(k) for k in ('enabled','ready','identity','available','memory','initial_memory','capture_scope','candidate_batch_sizes','candidate_widths','reference_retained','model_quality_verified')},
                    full_receipt='/r1out/abc-preparation-rank'+str(self.worker.rank)+'.json',
                    routes=self.routes,model_quality_verified=False)
    def entry(self,profile,b,k):
        if profile.experts=='routed':
            if self.routes is None:return None
            family=(True,True,self.routes['experts'][str(b)])
            return self.targets.get(family_key(family,b,k))
        return self.reference_entry(b,k) if profile.reference else self.targets.get(family_key(profile.family(),b,k))

def intercept(wrapper,ctx,args,kwargs):
    from .w26_dual_graph import descriptor as desc_of
    from vllm.model_executor.offloader.base import get_offloader
    rt=CAPTURE
    d=desc_of(ctx.batch_descriptor)
    if rt is not None and wrapper is rt.wrapper and d==rt.capture_desc and ctx.cudagraph_runtime_mode.name=='FULL':
        if rt.mode=='warm':return True,wrapper.runnable(*args,**kwargs)
        if rt.mode=='capture':return True,rt.capture_at_call(args,kwargs)
        if rt.mode=='qualify_replay':return True,rt.replay(rt.entry,args,kwargs)
        if rt.mode=='baseline':
            if rt.reference is not None:return True,rt.replay(rt.reference,args,kwargs)
            # k3 has no incumbent graph. Compare incumbent math eagerly with the
            # same metadata; never let the stock wrapper lazily capture a new key.
            return True,wrapper.runnable(*args,**kwargs)
        return None
    bank=BANK;p=current()
    if bank is None or p is None or wrapper is not bank.runner.model:return None
    if bank.selection.pending is not None:raise RuntimeError('ABC replay during transaction')
    if ctx.cudagraph_runtime_mode.name!='FULL' or not (d[2] is True and d[3] is False and d[4]==0):
        bank.hits['target_fallback:mode']+=1;return None
    real=int(bank.runner.input_batch.num_reqs)
    k=d[0]//d[1]-1 if d[1] and d[0]%d[1]==0 else -1
    if getattr(bank.runner,'_abc_layout',None)!=(real,k):
        bank.hits['target_fallback:not_uniform_spec_decode']+=1;return None
    entry=bank.entry(p,real,k)
    if entry is None:
        bank.hits[f'target_fallback:{p.name}:B{real}:k{k}']+=1;return None
    if entry['addresses']!=wrapper._collect_tensor_addresses(args,kwargs):raise RuntimeError('ABC input storage moved')
    get_offloader().sync_prev_onload();entry['graph'].replay()
    bank.hits[f'target:{p.name}:B{real}:k{k}']+=1
    return True,entry['output']

def try_sample(p,b,hidden):
    bank=BANK;profile=current()
    if bank is None or profile is None:return None
    import torch,os
    if (p.num_speculative_tokens!=5 or p._draft_topk is not None or p._conf_threshold()!=0
        or os.environ.get('W26_OBSERVE_MARGINS','0')=='1' or torch.cuda.is_current_stream_capturing()
        or tuple(hidden.shape)!=(5*b,p.model.config.hidden_size) or hidden.dtype!=torch.bfloat16):
        bank.hits['sampler_fallback:eligibility']+=1;return None
    plan=getattr(p,'_dsv41_r1_plan',None)
    if plan is not None and (plan.any_sample or plan.any_glog):
        # Retain exact existing proposal q/RNG. FP8/gating never leak into sampled/mixed requests.
        if profile.reference and bank.r2 is not None and b in bank.r2.sampled:
            from .r2_sampler import fallback_reason
            from . import dsv41_r1 as r1
            if fallback_reason(p,b,hidden,plan,bank.r2) is None and r1.sampling_supported(p,bank.r2.vocab,plan):
                ids,q=bank.r2.sampled[b].sample(hidden,r1.tq_tensor(plan,p.device))
                p._last_draft_probs=q;p._dsv41_r1_path='abc_reference_sampled';bank.hits[f'sampled:A:B{b}']+=1;return ids
        if not plan.any_glog and not profile.reference:
            from . import dsv41_r1 as r1
            state=bank.sampled.get(b) if b==1 else (None if bank.r2 is None else bank.r2.sampled.get(b))
            if state is not None and r1.sampling_supported(p,129280,plan):
                ids,q=state.sample(hidden,r1.tq_tensor(plan,p.device));p._last_draft_probs=q
                p._dsv41_r1_path='abc_bf16_sampled';bank.hits[f'sampled:{profile.name}:B{b}']+=1;return ids
        bank.hits[f'sampler_fallback:sampled_or_glog:B{b}']+=1;return None
    if profile.reference:
        state=bank.reference_sampler if b==1 else (None if bank.r2 is None else bank.r2.greedy.get(b))
    else:state=bank.samplers.get((b,profile.fp8_draft and b<=3,profile.fused_markov,profile.gate))
    if state is None:bank.hits[f'sampler_fallback:not_prepared:B{b}']+=1;return None
    out=state.sample(hidden)
    if plan is not None:p._dsv41_r1_path='abc_'+profile.name
    bank.hits[f'sampler:{profile.name}:B{b}']+=1
    return out

