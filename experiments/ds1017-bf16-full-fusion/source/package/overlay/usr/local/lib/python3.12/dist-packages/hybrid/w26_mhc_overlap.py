"""Future source-only mHC seam. Active only during candidate construction.

Replay selects a prebuilt graph; it never toggles Python model code. Not installed
in the retained owner. Only SM80, H5120/hc4/T6, original tilelang math is allowed.
"""
ACTIVE=None

def begin(layer,phase,original,*args,**kwargs):
    if ACTIVE is None:return original(*args,**kwargs)
    return ACTIVE.begin(layer,phase,original,*args,**kwargs)

def join(layer,phase):
    if ACTIVE is not None:ACTIVE.join(layer,phase)

class OverlapKernels:
    def __init__(self,layers,device):
        import torch
        self.torch=torch;self.side=torch.cuda.Stream(device=device);self.slots={};self.pending=set();self.layers={id(layer):layer for layer in layers}
        for layer in layers:
            for phase in ('attn','ffn'):
                # First broadcast attention stays exactly incumbent.
                if phase=='attn' and getattr(layer,'_dsv41_idx',-1)==0:continue
                key=(id(layer),phase)
                self.slots[key]=dict(mix=torch.empty((1,6,24),device=device,dtype=torch.float32),sqr=torch.empty((1,6),device=device,dtype=torch.float32),post=torch.empty((6,4),device=device,dtype=torch.float32),comb=torch.empty((6,16),device=device,dtype=torch.float32),inp=torch.empty((6,5120),device=device,dtype=torch.bfloat16),pre=torch.empty((6,4),device=device,dtype=torch.float32),ready=torch.cuda.Event(),done=torch.cuda.Event())
        pointers=[v.untyped_storage().data_ptr() for b in self.slots.values() for v in b.values() if isinstance(v,torch.Tensor)]
        assert len(pointers)==len(set(pointers))
    def tensor_bytes(self):
        return sum(v.numel()*v.element_size() for b in self.slots.values() for v in b.values() if isinstance(v,self.torch.Tensor))
    def begin(self,layer,phase,original,residual,fn,scale,base,rms_eps,pre_eps,sink_eps,post_alpha,iters,pre_mix=None,x=None,norm_weight=None,norm_eps=1e-6):
        torch=self.torch;key=(id(layer),phase)
        if key not in self.slots:
            return original(residual,fn,scale,base,rms_eps,pre_eps,sink_eps,post_alpha,iters,pre_mix=pre_mix,x=x,norm_weight=norm_weight,norm_eps=norm_eps)
        assert key not in self.pending,'Unconsumed coefficients would be overwritten'
        assert x is None and residual.shape==(6,4,5120) and residual.dtype==torch.bfloat16 and residual.is_contiguous()
        assert fn.shape==(24,20480) and fn.dtype==torch.float32
        assert pre_mix is not None and pre_mix.shape==(6,4) and pre_mix.dtype==torch.float32
        assert norm_weight is not None and norm_weight.shape==(5120,) and norm_weight.dtype==torch.bfloat16
        from vllm.model_executor.kernels.mhc.tilelang import _tilelang_hc_prenorm_gemm
        from hybrid.w26_mhc_kernels import w26_mhc_pre_split
        b=self.slots[key];main=torch.cuda.current_stream(residual.device)
        args=[b['mix'],b['sqr'],scale,base,residual,b['post'],b['comb'],b['inp'],norm_weight,pre_mix,b['pre']]
        kw=dict(hidden_size=5120,rms_eps=rms_eps,hc_pre_eps=pre_eps,hc_sinkhorn_eps=sink_eps,hc_post_mult_value=post_alpha,sinkhorn_repeat=iters,norm_eps=norm_eps,n_splits=1,hc_mult=4,use_pre_mix_in=True,save_pre_mix=True,rms_numel=20480)
        b['ready'].record(main);self.side.wait_event(b['ready'])
        residual.record_stream(self.side)
        with torch.cuda.stream(self.side):
            _tilelang_hc_prenorm_gemm(residual.view(6,20480),fn,b['mix'],b['sqr'],20480,1)
            w26_mhc_pre_split(*args,**kw,split_mode='stats');b['done'].record(self.side)
        w26_mhc_pre_split(*args,**kw,split_mode='input')
        self.pending.add(key)
        return b['post'].unsqueeze(-1),b['comb'].view(6,4,4),b['inp'],b['pre']
    def join(self,layer,phase):
        key=(id(layer),phase)
        if key in self.pending:
            self.torch.cuda.current_stream().wait_event(self.slots[key]['done'])
            self.pending.remove(key)
    def assert_drained(self):
        assert not self.pending,'Candidate exits with an unjoined stats stream'
    def qualify_locally(self):
        # No GPU communicator calls here. JIT/resource/arithmetic failures can
        # therefore be agreed across the CPU group before any model collective.
        torch=self.torch
        from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_delayed_tilelang as original
        device=next(iter(self.slots.values()))['inp'].device
        generator=torch.Generator(device=device).manual_seed(261025)
        residual=torch.randn((6,4,5120),device=device,dtype=torch.bfloat16,generator=generator)
        pre=torch.full((6,4),.25,device=device,dtype=torch.float32)
        for key in self.slots:
            layer=self.layers[key[0]];phase=key[1]
            args=(residual,getattr(layer,'hc_'+phase+'_fn'),getattr(layer,'hc_'+phase+'_scale'),getattr(layer,'hc_'+phase+'_base'),layer.rms_norm_eps,layer.hc_eps,layer.hc_eps,layer.hc_post_alpha,layer.hc_sinkhorn_iters)
            norm=getattr(layer,phase+'_norm')
            kw=dict(pre_mix=pre,norm_weight=norm.weight,norm_eps=norm.variance_epsilon)
            ref=original(*args,**kw)
            got=self.begin(layer,phase,original,*args,**kw);self.join(layer,phase)
            torch.cuda.synchronize()
            if not all(torch.equal(a,b) for a,b in zip(ref,got)):raise ValueError('mHC local loaded-coefficient parity failure')
        self.assert_drained()
