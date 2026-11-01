"""Candidate-only native-FP8/wo_a/lane dispatch; reference packs stay intact."""
import os
from . import abc_runtime as R
from .abc_contract import lane_rows

PACK_BYTES=0
PACKS=[]
COUNTS={}
def count(name):
    c=R.context()
    if c is not None:c.paths[name]=c.paths.get(name,0)+1

def prepare_dense(layer):
    if os.environ.get('STAR_ABC','0')!='1':return
    import torch
    N,K=layer.output_size_per_partition,layer.input_size_per_partition
    if layer.weight.dtype!=torch.float8_e4m3fn or tuple(layer.weight.shape)!=(N,K) or getattr(layer,'bias',None) is not None:return
    from vllm.model_executor.kernels.linear.mxfp8 import dsv41_fp8w as g
    global PACK_BYTES
    # Bound retained ALTERNATE packing. Stock Marlin always remains the A implementation.
    estimate=N*K+N*K//32+((N//16+3)//4)*4
    cap=int(os.environ.get('STAR_ABC_PACK_MIB','4096'))*1024**2
    if N*K>g.MAX_BYTES or N%16 or K%64:return
    if PACK_BYTES+estimate>cap:raise RuntimeError('ABC native-FP8 retained-pack cap exceeded before allocation')
    candidate=g.prepare(layer.weight.data.view(torch.uint8),layer.weight_scale.data.view(torch.uint8))
    if candidate is not None:
        from .abc_numeric import dense as check_dense
        check_dense(layer,candidate)
        W,S,cnt,_=candidate;layer._abc_w8=(W,S,cnt);PACK_BYTES+=sum(t.numel()*t.element_size() for t in (W,S,cnt));PACKS.append(layer)

def dense(layer,x,bias):
    from .prefill_runtime import dense_candidate
    prefill = dense_candidate(layer,x,bias)
    if prefill is not None:return prefill
    if not R.target_feature('dense') or bias is not None or not hasattr(layer,'_abc_w8'):return None
    from vllm.model_executor.kernels.linear.mxfp8 import dsv41_fp8w as g
    if layer.input_size_per_partition*layer.output_size_per_partition<=g.DUAL_BYTES and x.numel()//x.shape[-1]>8:
        count('dense_retained_marlin_mid');return None
    W,S,cnt=layer._abc_w8;count('dense_native_fp8')
    # Candidate private decode only; preserve all reference and prefill packs/routes.
    return g._fp8_linear(x,W,S,cnt,None,None,None,layer.output_size_per_partition,layer.input_size_per_partition)

def prepare_woa(model):
    import torch
    from vllm.models.deepseek_v4_1.ampere import dsv41_wo_a as w
    rows=[]
    for name,m in model.named_modules():
        if not all(hasattr(m,k) for k in ('wo_a','n_local_groups','o_lora_rank','rope_head_dim')):continue
        weight=getattr(m.wo_a,'_dsv4_wo_a_bf16',None)
        if weight is None:
            weight=m.wo_a.weight.data.view(m.n_local_groups,m.o_lora_rank,-1).to(torch.bfloat16).contiguous()
        q=w.recover_fp8(weight)
        m._abc_woa=q
        checks=None
        if q is not None:
            from .abc_numeric import woa as check_woa
            checks=check_woa(m,weight,q)
        rows.append(dict(module=name,exact_weight_recovery=q is not None,bytes=0 if q is None else sum(t.numel()*t.element_size() for t in q),numeric=checks))
    return rows

def woa(m,o,positions):
    if not R.target_feature('woa') or o.shape[0]>16 or not getattr(m,'_abc_woa',None):return None
    import torch
    from vllm.models.deepseek_v4_1.ampere import dsv41_wo_a as w
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fused_inverse_rope_gptj
    q,s=m._abc_woa;G,rank=m.n_local_groups,m.o_lora_rank
    count('woa_native_fp8')
    if w.cuda_small_ok(o) and o.dim()==3 and m.rotary_emb.cos_sin_cache.dtype==torch.float32:
        return m.wo_b(torch.ops.vllm.dsv41_wo_a_small(o,positions,m.rotary_emb.cos_sin_cache,q,s,G,rank,m.rope_head_dim))
    x=_fused_inverse_rope_gptj(o,positions,m.rotary_emb.cos_sin_cache,m.rope_head_dim).reshape(o.shape[0],-1).contiguous()
    return m.wo_b(w.wo_a(x,q,s,G,rank,x.shape[1]//G))

LANE_OPS=None
LANE_WS=None
def prepare_lane(device):
    import torch
    global LANE_OPS,LANE_WS
    torch.ops.load_library('/opt/star-abc/lib/_dsv41_mm_v2try.abi3.so')
    LANE_OPS=torch.ops._dsv41_mm_v2try
    need=int(LANE_OPS.moe_workspace_size_v2_try(64,5120,640,6,384))
    LANE_WS=torch.empty(need,dtype=torch.float32,device=device)
    return dict(elements=need,bytes=need*4)

def experts(x,ids,weights,layer,expert_map,limit):
    c=R.context()
    if c is None or not lane_rows(c.profile.experts,c.plan.b,c.plan.k):return None
    import torch
    tc=getattr(layer,'_exl3_tc',None)
    if tc is None or x.dtype!=torch.bfloat16 or not 1<=x.shape[0]<=64 or ids.shape[-1]!=6:return None
    emap=tc['emap'] if expert_map is None else expert_map
    if emap.device!=x.device:return None
    out=torch.empty_like(x)
    LANE_OPS.moe_out_v2_try(x.contiguous(),ids,weights.float().contiguous(),tc['ptrs'],tc['meta'],emap,tc['intermediate'],tc['bits_x2'],out,LANE_WS,float(limit or 0))
    from . import r2_experts
    r2_experts.record('tc');count('experts_lane')
    return out

