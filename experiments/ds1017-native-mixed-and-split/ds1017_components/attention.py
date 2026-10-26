"""Explicit FULL target-bank compressed attention component.

Passed initial numerical/runtime checks; evaluated in limited matched end-to-end
tests with mixed task-quality results.
Broader numerical accuracy and task-quality verification is still warranted. The bank coordinator owns all-rank disable, resource checks,
state restoration, graph lifetime and benchmark-only publication. This module
never selects a bank, changes an installed kernel, or enables serving capture.
"""
import hashlib
import inspect
import os
from pathlib import Path
from .attention_profiles import Profile,CURRENT

VERSION='w26-compressed-t6-split16-v1'
OP_SHA='a36c0e13cf63a77c804a5d5d60ec2f6d3e62bec51de3afac4f28619f91f5f8fc'
FP8_SHA='4ad3bd77073051c0cb2db2ef3d307ff5047e9a29dab3714fcc65843e87a7eb6e'
EXPECTED_LAYERS=tuple(range(2,40))
PHYSICAL_STRIDE=230400
WORKSPACE_BYTES=(6*16*16*2+6*16*16*512)*4+6*16*512*2+4

def _layout(t,block,device):
    if t is None or not t.is_cuda or t.device!=device or str(t.dtype)!='torch.uint8':return False
    if t.ndim!=3 or tuple(t.shape[1:])!=(block,584) or t.stride()!=(PHYSICAL_STRIDE,584,1) or t.shape[0]<1:return False
    # Do not assume layer offsets or compact storage; check actual byte extent.
    end=t.storage_offset()+(t.shape[0]-1)*PHYSICAL_STRIDE+block*584
    return end<=t.untyped_storage().nbytes()

def _vector(t,device,capacity=None):
    return (t is not None and t.is_cuda and t.device==device and str(t.dtype)=='torch.int32'
            and t.ndim==1 and t.is_contiguous() and (capacity is None or t.numel()==capacity))

def eligibility(layer,k,device):
    """Metadata only: never reads model data or synchronizes inside capture."""
    lid=getattr(layer,'layer_id',None)
    ratio=getattr(layer,'compress_ratio',None)
    if lid not in EXPECTED_LAYERS or ratio!=(2 if lid<20 else 1):return 'layer_or_compression'
    if k['swa_only'] or k['topk_indices'] is not None:return 'not_compressed_ragged'
    q=k['q'];out=k['output']
    for t in (q,out):
        if (not t.is_cuda or t.device!=device or str(t.dtype)!='torch.bfloat16'
            or tuple(t.shape)!=(6,16,512) or t.stride()!=(8192,512,1)):return 'query_or_output'
    if (k['head_dim'],k['nope_head_dim'],k['rope_head_dim'])!=(512,448,64) or k['scale']!=512**-.5:return 'dimensions_scale'
    if not _layout(k['swa_k_cache'],32,device) or not _layout(k['kv_cache'],128//ratio,device):return 'cache_layout'
    for name in ('swa_ragged_indices','topk_ragged_indices'):
        if not _vector(k[name],device):return 'indices'
    for name in ('swa_ragged_indptr','topk_ragged_indptr'):
        if not _vector(k[name],device,7):return 'indptr'
    if k.get('adaptive_splits',False):return 'adaptive_splits'
    sink=k['attn_sink']
    if sink is not None and (not sink.is_cuda or sink.device!=device or sink.ndim!=1 or sink.numel()<16 or not sink.is_contiguous() or str(sink.dtype)!='torch.float32'):return 'sink'
    return None

class AttentionState:
    """One serial-capture-stream workspace, strongly owned by Runtime/entry."""
    def __init__(self,device,profile=CURRENT):
        if type(profile) is not Profile:raise ValueError('Expected immutable supported profile')
        self.profile=profile
        import torch
        import vllm.v1.attention.ops.rocm_aiter_mla_sparse as op
        import vllm.v1.attention.ops.fp8_sm80 as fp8
        self.torch=torch;self.op=op;self.device=torch.device(device)
        if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        assert self.device.type=='cuda' and torch.cuda.get_device_capability(self.device)==(8,0)
        assert not torch.cuda.is_current_stream_capturing()
        assert not op._ON_GFX942 and not op._ON_GFX950 and not op.current_platform.is_fp8_fnuz()
        for mod,pin in ((op,OP_SHA),(fp8,FP8_SHA)):
            assert hashlib.sha256(Path(inspect.getfile(mod)).read_bytes()).hexdigest()==pin,'retained source pin'
        self.m=torch.empty((6,profile.splits,16),device=self.device,dtype=torch.float32)
        self.l=torch.empty_like(self.m)
        self.acc=torch.empty((6,profile.splits,16,512),device=self.device,dtype=torch.float32)
        self.temp=torch.empty((6,16,512),device=self.device,dtype=torch.bfloat16)
        self.dummy_sink=torch.empty(1,device=self.device,dtype=torch.float32)
        self.lut=op.get_e4m3fn_bf16_lut(self.device)
        self.prepared=False;self.capturing=False;self.layers=[];self.rejections=[];self.stream=None

    def _launch(self,k):
        op=self.op;q=k['q'];main=k['swa_k_cache'];extra=k['kv_cache'];m=self.m;l=self.l;a=self.acc
        profile=self.profile;split=profile.splits
        op._sparse_attn_decode_partial_kernel[(6,split,1)](
            q,main,k['swa_ragged_indices'],k['swa_ragged_indptr'],extra,k['topk_ragged_indices'],k['topk_ragged_indptr'],m,l,a,self.lut,
            q.stride(0),q.stride(1),main.stride(0),extra.stride(0),m.stride(0),m.stride(1),a.stride(0),a.stride(1),a.stride(2),
            main.shape[0]*main.shape[1],extra.shape[0]*extra.shape[1],main.shape[1],extra.shape[1],k['scale'],16,
            HAS_EXTRA=True,NOPE_DIM=448,NOPE_BLOCK=512,ROPE_DIM=64,IS_FNUZ_MAIN=False,IS_FNUZ_EXTRA=False,
            BLOCK_H=16,BLOCK_K=32,NUM_SPLITS=split,NUM_STAGES=1,num_warps=profile.warps)
        sink=k['attn_sink']
        dest=k['output'] if profile.direct else self.temp
        op._sparse_attn_decode_reduce_kernel[(6,16)](
            m,l,a,sink if sink is not None else self.dummy_sink,dest,
            dest.stride(0),dest.stride(1),m.stride(0),m.stride(1),a.stride(0),a.stride(1),a.stride(2),16,
            HAS_ATTN_SINK=sink is not None,ADAPTIVE_SPLITS=False,COMB_DIM=512,BLOCK_H=1,NUM_SPLITS=split,SPLITS_PAD=split,num_warps=4)
        if not profile.direct:k['output'].copy_(self.temp)

    def compile_probe(self):
        """Compile generated empty segments in both cache geometries; no model reads.

        This tests initialization and empty-output semantics only, not model
        qualification. Coordinator checks physical memory before/after and
        agrees on recoverable failures across ranks before model capture.
        """
        t=self.torch
        assert not t.cuda.is_current_stream_capturing() and not self.capturing
        q=t.zeros((6,16,512),device=self.device,dtype=t.bfloat16);out=t.empty_like(q)
        raw=t.zeros(2*PHYSICAL_STRIDE,device=self.device,dtype=t.uint8)
        main=raw.as_strided((1,32,584),(PHYSICAL_STRIDE,584,1),0)
        ids=t.full((1,),-1,device=self.device,dtype=t.int32);ptr=t.zeros(7,device=self.device,dtype=t.int32)
        sink=t.zeros(16,device=self.device,dtype=t.float32)
        for block in (64,128):
            extra=raw.as_strided((1,block,584),(PHYSICAL_STRIDE,584,1),PHYSICAL_STRIDE)
            for active_sink in (None,sink):
                k=dict(q=q,output=out,swa_k_cache=main,kv_cache=extra,swa_ragged_indices=ids,swa_ragged_indptr=ptr,topk_ragged_indices=ids,topk_ragged_indptr=ptr,attn_sink=active_sink,scale=512**-.5)
                self._launch(k)
                t.cuda.synchronize(self.device)
                assert bool((out==0).all()),'empty split probe'
        self.prepared=True
        return dict(version=self.profile.version,profile=self.profile.record(),compiled=True,probe='generated empty segments; not engine qualification',workspace_bytes=self.profile.workspace_bytes)

    def begin_capture(self):
        assert self.prepared and not self.capturing and not self.torch.cuda.is_current_stream_capturing()
        self.layers=[];self.rejections=[];self.capturing=True
        self.stream=self.torch.cuda.current_stream(self.device).cuda_stream

    def finish_capture(self):
        assert self.capturing and not self.torch.cuda.is_current_stream_capturing()
        self.capturing=False
        if self.rejections or tuple(self.layers)!=EXPECTED_LAYERS:
            raise ValueError('incomplete compressed attention coverage: '+repr((self.layers,self.rejections)))
        return dict(version=self.profile.version,profile=self.profile.record(),layers=self.layers.copy(),calls=38,split=self.profile.splits,scope='compressed target T6/B1 only')

    def run(self,layer,k):
        if not self.prepared or not self.capturing:raise ValueError('attention state not in prepared capture')
        if not self.torch.cuda.is_current_stream_capturing() or self.torch.cuda.current_stream(self.device).cuda_stream!=self.stream:
            raise ValueError('shared workspace requires the fixed private capture stream')
        reason=eligibility(layer,k,self.device)
        if reason is not None:
            self.rejections.append((getattr(layer,'layer_id',None),reason));return False
        self._launch(k);self.layers.append(layer.layer_id);return True

def prepare_attention(device):
    """Explicit startup opt-in, before target graph capture; no lazy preparation."""
    if os.environ.get('W26_ATTENTION_SPLIT','0')!='1':return None
    state=AttentionState(device, Profile(splits=8,warps=8,direct=False))
    state.compile_probe()
    return state

def decode_during_capture(state,layer,fallback,**kwargs):
    """Caller passes the state for its exclusively owned target capture only."""
    if state is None or kwargs['swa_only'] or getattr(layer,'layer_id',None) not in EXPECTED_LAYERS:
        return fallback(**kwargs)
    if state.run(layer,kwargs):return None
    return fallback(**kwargs)
