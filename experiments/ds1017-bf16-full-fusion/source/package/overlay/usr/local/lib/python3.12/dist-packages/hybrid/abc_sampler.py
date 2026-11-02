"""Initial-only integrated norm/head/Markov captures; draft FP8 never replaces target head."""
import torch
import torch.nn.functional as F
from .r2_sampler import _Captured, gather_pairs, pair_select, local_pair

_CAPTURE_STREAMS={}
def capture_stream(device):
    key=torch.device(device).index
    if key is None:key=torch.cuda.current_device()
    if key not in _CAPTURE_STREAMS:_CAPTURE_STREAMS[key]=torch.cuda.Stream(device=device)
    return _CAPTURE_STREAMS[key]

def quantize_head(p):
    w=p.model.lm_head.weight
    if w.dtype!=torch.bfloat16 or w.ndim!=2 or w.shape[1]%128:raise ValueError('BF16 target/draft head required')
    from . import dsv41_fp8_gemv as g
    N,K=w.shape;nb=(N+127)//128
    q=torch.empty(N,K,dtype=torch.uint8,device=w.device)
    s=torch.empty(nb,K//128,dtype=torch.float32,device=w.device)
    for b0 in range(0,nb,32):
        r0,r1=b0*128,min(N,(b0+32)*128)
        v=w[r0:r1].float();v=F.pad(v,(0,0,0,(-v.shape[0])%128)).view(-1,128,K//128,128)
        sc=v.abs().amax(dim=(1,3)).clamp_min(1e-12)/448.
        s[b0:b0+sc.shape[0]]=sc
        q[r0:r1]=(v/sc[:,None,:,None]).to(torch.float8_e4m3fn).view(-1,K)[:r1-r0].view(torch.uint8)
    g._TUNED.setdefault((N,K),(64,128,2,4,3))
    return q,s

class Integrated(_Captured):
    def __init__(self,p,nccl,start,end,b,fp8=False,fused=False,gate=False,q8=None):
        super().__init__(p,b)
        self.nccl=nccl;self.start=start;self.end=end;self.fp8=fp8;self.fused=fused;self.gate=gate;self.q8=q8
        self.q8_binding=None if q8 is None else tuple((id(t),t.data_ptr()) for t in q8)
        self.hidden=torch.zeros((b*5,p.model.config.hidden_size),dtype=torch.bfloat16,device=p.device)
        self.anchor=torch.zeros(b,dtype=torch.int64,device=p.device)
        self.output=None;self.keep=None
    def assert_intact(self):
        super().assert_intact()
        if self.q8 is not None and self.q8_binding!=tuple((id(t),t.data_ptr()) for t in self.q8):raise RuntimeError('ABC FP8 draft-head storage changed')
    def chain(self):
        p=self.p;m=p.model;xh=m.model.norm(self.hidden)
        if self.fp8:
            from . import dsv41_fp8_gemv as g
            local=g.fp8_linear(xh.contiguous(),*self.q8)
        else:local=m.logits_processor(m.lm_head,xh,skip_gather=True)
        if local.dtype!=torch.bfloat16 or local.shape!=(self.b*5,self.end-self.start):raise ValueError('Unexpected integrated draft logits')
        local=local.view(self.b,5,-1);w=m.model.markov_head.markov_w2.weight[self.start:self.end]
        prev=self.anchor;drafts=[];embeds=[]
        for i in range(5):
            e=m.markov_embed(prev)
            if self.gate:embeds.append(e)
            if self.fused:
                from . import dsv41_markov as mk
                pair=mk.markov_step_pair(local[:,i],e.to(torch.bfloat16).contiguous(),w,self.start)
            else:pair=local_pair(local[:,i]+F.linear(e,w),self.start)
            prev=m.map_draft_to_target(pair_select(gather_pairs(self.nccl,pair)))
            drafts.append(prev)
        out=torch.stack(drafts,dim=1)
        if self.gate:
            conf=m.compute_confidence(self.hidden,torch.stack(embeds,dim=1).flatten(0,1)).view(self.b,5).float()
            from .abc_gate_check import confidence_keep
            self.keep=confidence_keep(conf)
        return out
    def sample(self,hidden):
        def fill():
            self.hidden.copy_(hidden)
            self.anchor.copy_(self.p.input_ids[self.p._anchor_idx[:self.b]])
        out=self._run(fill,lambda:self.output.clone())
        if self.gate:
            from . import abc_gate
            abc_gate.publish(self.keep,self.b)
        return out

def prepare(p,nccl,start,end,b,fp8,fused,gate,q8,consensus,*,pool=None):
    state=Integrated(p,nccl,start,end,b,fp8,fused,gate,q8)
    idx=p._anchor_idx[:b].clone();saved=p.input_ids[idx].clone()
    # Reuse the initial-capture stream/CUBLAS scratch. Graph pools, static
    # inputs and owned outputs remain separate. Replay is still restricted
    # to the original serialized serving stream by _Captured._run.
    stream=capture_stream(p.device);stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):state.chain()
    stream.synchronize()
    state.graph=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(state.graph,stream=stream,pool=pool):state.output=state.chain()
    state.graph.instantiate();state.ready=True
    records=[];first=None
    try:
        for seed in (17,23,17):
            gen=torch.Generator(device=p.device).manual_seed(seed)
            hidden=torch.randn(state.hidden.shape,device=p.device,dtype=torch.bfloat16,generator=gen)
            p.input_ids.index_copy_(0,idx,(torch.arange(b,device=p.device,dtype=p.input_ids.dtype)*7919+seed)%129280)
            reference=p._sample_sequential_incumbent(b,hidden)
            actual=state.sample(hidden);repeat=state.sample(hidden)
            torch.cuda.synchronize()
            valid=bool(((actual>=0)&(actual<129280)).all())
            own=torch.equal(actual,repeat) and (seed!=17 or first is None or torch.equal(actual,first))
            same=torch.equal(reference,actual)
            ok=valid and own and (same or fp8 or fused)
            votes=consensus(dict(ok=bool(ok),phase=('abc_sampler',b,fp8,fused,gate,seed)))
            if not all(v==votes[0] and v['ok'] for v in votes):raise RuntimeError('ABC sampler qualification differs/fails across ranks')
            records.append(dict(seed=seed,valid_ids=valid,own_exact=own,reference_ids_exact=same))
            if first is None:first=actual.clone()
    finally:
        p.input_ids.index_copy_(0,idx,saved);torch.cuda.synchronize()
    return state,dict(ready=True,fp8_draft=fp8,fused=fused,gate=gate,capture_stream=int(stream.cuda_stream),records=records,target_head_replaced=False,model_quality_verified=False)


@torch.inference_mode()
def qualify_interleaved(p,states,consensus):
    """Exercise arbitrary serialized replay order and retained output ownership.

    Integrated graphs share scratch only. Inputs/anchors and the FP8 weights
    remain separately owned. Every returned token tensor is cloned before the
    next graph runs; gate publication completes in the same serving stream.
    RNG-bearing sampled graphs keep their own existing pool and generator.
    """
    from . import abc_gate as G
    states=list(states)
    if not states:raise ValueError('No integrated samplers to qualify')
    idx=p._anchor_idx[:max(s.b for s in states)].clone()
    saved=p.input_ids[idx].clone();expected=[];held=[];records=[]
    try:
        # Separate generators keep this proof out of serving RNG state.
        fixtures=[]
        for i,state in enumerate(states):
            gen=torch.Generator(device=p.device).manual_seed(27017+i)
            hidden=torch.randn(state.hidden.shape,device=p.device,dtype=torch.bfloat16,generator=gen)
            anchor=(torch.arange(state.b,device=p.device,dtype=p.input_ids.dtype)*7919+27017+i)%129280
            fixtures.append((hidden,anchor))
            p.input_ids.index_copy_(0,idx[:state.b],anchor);G.reset()
            out=state.sample(hidden)
            expected.append((out.clone(),None if not state.gate else state.keep.clone()))
            held.append(out)
        orders=[list(reversed(range(len(states)))),list(range(1,len(states)))+[0]]
        for order_id,order in enumerate(orders):
            for i in order:
                state=states[i];hidden,anchor=fixtures[i]
                p.input_ids.index_copy_(0,idx[:state.b],anchor);G.reset()
                actual=state.sample(hidden);ref,keep=expected[i]
                exact=torch.equal(actual,ref)
                gate_exact=not state.gate or (torch.equal(state.keep,keep) and torch.equal(G.MASK[G.INDICES[state.b]],keep.reshape(-1)))
                owned=all(torch.equal(value,exp[0]) for value,exp in zip(held,expected))
                ok=bool(exact and gate_exact and owned)
                votes=consensus(dict(ok=ok,phase=('abc_sampler_interleave',order_id,i,state.b,state.gate)))
                if not all(v==votes[0] and v['ok'] for v in votes):raise RuntimeError('Shared sampler scratch replay/ownership proof failed')
                records.append(dict(order=order_id,index=i,b=state.b,gate=state.gate,ids_exact=exact,gate_exact=gate_exact,retained_outputs_exact=owned))
    finally:
        p.input_ids.index_copy_(0,idx,saved);G.reset();torch.cuda.synchronize()
    return dict(passed=True,cases=len(records),samplers=len(states),arbitrary_order_bitwise=True,retained_outputs_owned=True,records=records)
