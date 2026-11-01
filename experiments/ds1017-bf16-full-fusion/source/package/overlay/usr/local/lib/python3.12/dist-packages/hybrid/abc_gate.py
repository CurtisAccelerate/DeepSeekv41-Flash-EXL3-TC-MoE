# Masked router derived from GNmac/Morrow, Apache-2.0. Exact pinned donor in SOURCES.json.
import dataclasses
import torch
import triton
import triton.language as tl

@triton.jit
def _dsv41_topk_masked_kernel(gating_output_ptr, correction_bias_ptr, topk_weights_ptr, topk_ids_ptr,
                              routed_scaling_factor, input_ids_ptr, bias_vl_ptr, image_sentinel_lo, mask_ptr,
                              NUM_EXPERTS: tl.constexpr, BLOCK_N: tl.constexpr, HAS_VL: tl.constexpr):
    # _dsv4_topk_kernel (sm80: no PDL) plus the token mask
    row = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_N)
    expert_mask = expert_offsets < NUM_EXPERTS
    bias = tl.load(correction_bias_ptr + expert_offsets, mask=expert_mask, other=0.0).to(tl.float32)
    if HAS_VL:
        token_id = tl.load(input_ids_ptr + row).to(tl.int64)
        bias_vl = tl.load(bias_vl_ptr + expert_offsets, mask=expert_mask, other=0.0).to(tl.float32)
        is_image = (token_id >= image_sentinel_lo) & (token_id < image_sentinel_lo + 5)
        bias = tl.where(is_image, bias_vl, bias)
    logits = tl.load(gating_output_ptr + row * NUM_EXPERTS + expert_offsets, mask=expert_mask, other=0.0).to(tl.float32)
    weights = tl.sqrt(tl.where(logits > 20.0, logits, tl.log(1.0 + tl.exp(logits))))
    current = tl.where(expert_mask, weights + bias, -float("inf"))
    current = tl.where(current == current, current, -1e30)
    topk_offsets = tl.arange(0, 8)
    selected_weights = tl.zeros([8], dtype=tl.float32)
    selected_ids = tl.zeros([8], dtype=tl.int32)
    for slot in tl.static_range(6):
        max_value = tl.max(current, axis=0)
        candidate = tl.where(current == max_value, expert_offsets, NUM_EXPERTS)
        expert_id = tl.min(candidate, axis=0).to(tl.int32)
        selected_weight = tl.sum(tl.where(expert_offsets == expert_id, weights, 0.0), axis=0)
        is_slot = topk_offsets == slot
        selected_weights = tl.where(is_slot, selected_weight, selected_weights)
        selected_ids = tl.where(is_slot, expert_id, selected_ids)
        current = tl.where(expert_offsets == expert_id, -float("inf"), current)
    weight_sum = tl.sum(selected_weights, axis=0)
    selected_weights *= routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)
    keep = tl.load(mask_ptr + row) != 0
    selected_weights = tl.where(keep, selected_weights, 0.0)
    selected_ids = tl.where(keep, selected_ids, NUM_EXPERTS)
    output_mask = topk_offsets < 6
    output_offsets = row * 6 + topk_offsets
    tl.store(topk_weights_ptr + output_offsets, selected_weights, mask=output_mask)
    tl.store(topk_ids_ptr + output_offsets, selected_ids, mask=output_mask)


MASK=None
INDICES={}
ORIGINAL=None
@torch.inference_mode()
def reset():
    if MASK is not None:MASK.fill_(1)
@torch.inference_mode()
def publish(keep,b):
    if keep.dtype!=torch.int32:raise ValueError('Router keep mask must be int32')
    MASK.fill_(1)
    MASK.index_copy_(0,INDICES[b],keep.reshape(-1))
def topk(gating_output,correction_bias,indices_dtype,routed_scaling_factor,input_ids=None,bias_vl=None,image_sentinel_lo=0):
    from . import abc_runtime as R
    ctx=R.context()
    if ctx is None or ctx.plan.k!=5:
        return ORIGINAL(gating_output,correction_bias,indices_dtype,routed_scaling_factor,input_ids=input_ids,bias_vl=bias_vl,image_sentinel_lo=image_sentinel_lo)
    n,experts=gating_output.shape
    out_w=gating_output.new_empty((n,6),dtype=torch.float32)
    out_i=gating_output.new_empty((n,6),dtype=indices_dtype)
    _dsv41_topk_masked_kernel[(n,)](gating_output,correction_bias,out_w,out_i,routed_scaling_factor,input_ids,bias_vl,image_sentinel_lo,MASK,NUM_EXPERTS=experts,BLOCK_N=triton.next_power_of_2(experts),HAS_VL=bias_vl is not None and image_sentinel_lo>0,num_warps=1)
    ctx.paths['gate_capable_router']=ctx.paths.get('gate_capable_router',0)+1
    return out_w,out_i
def prepare(device):
    global MASK,ORIGINAL
    from vllm.model_executor.layers.fused_moe.router import dsv4_topk as tk
    from vllm.model_executor.layers.fused_moe.router import fused_topk_bias_router as fr
    if MASK is not None:raise RuntimeError('ABC gate prepared twice')
    MASK=torch.ones(50,dtype=torch.int32,device=device)
    for b in range(1,9):INDICES[b]=torch.tensor([6*r+1+j for r in range(b) for j in range(5)],dtype=torch.long,device=device)
    ORIGINAL=tk.dsv4_topk
    # Compare actual old router, including VL correction, ties and sparse masks.
    gen=torch.Generator(device=device).manual_seed(1017)
    for ties in (False,True):
        x=torch.zeros(6,384,device=device) if ties else torch.randn((6,384),device=device,generator=gen)*3
        bias=torch.zeros(384,device=device);bvl=torch.randn((384,),device=device,generator=gen)*.1
        ids=torch.tensor([5,1000,1001,1004,1005,7],device=device)
        refw,refi=ORIGINAL(x,bias,torch.int32,2.5,input_ids=ids,bias_vl=bvl,image_sentinel_lo=1000)
        for pattern in ([1]*6,[1,1,0,0,0,0],[1,0,0,0,0,0]):
            MASK[:6].copy_(torch.tensor(pattern,dtype=torch.int32,device=device))
            w=torch.empty_like(refw);i=torch.empty_like(refi)
            _dsv41_topk_masked_kernel[(6,)](x,bias,w,i,2.5,ids,bvl,1000,MASK,NUM_EXPERTS=384,BLOCK_N=512,HAS_VL=True,num_warps=1)
            keep=MASK[:6].bool()
            if not (torch.equal(w[keep],refw[keep]) and torch.equal(i[keep],refi[keep]) and bool((w[~keep]==0).all()) and bool((i[~keep]==384).all())):raise RuntimeError('ABC router mask/reference parity failed')
    reset();fr.dsv4_topk=topk
    return dict(router_bitwise=True,patterns=3,ties=True,VL=True)
def before(runner,so,mode,desc):
    from . import abc_probe
    abc_probe.before(runner,so)
    from . import abc_runtime as R
    from .abc_contract import eligible_layout
    p=R.current();layout=eligible_layout(so.num_scheduled_tokens,so.scheduled_spec_decode_tokens)
    md=runner.input_batch.sampling_metadata
    runner._abc_layout=layout
    # Prefill of six tokens, mixed/sampled, nonuniform or narrow steps MUST clear the old mask.
    use=(p is not None and p.gate and layout is not None and layout[1]==5 and md.all_greedy and mode.name=='FULL' and R.BANK.entry(p,*layout) is not None)
    runner._abc_gate_step=bool(use)
    if not use:reset()
def rejection(metadata,sampling_metadata):
    from . import abc_runtime as R
    bank=R.BANK
    if bank is None or not getattr(bank.runner,'_abc_gate_step',False) or not sampling_metadata.all_greedy:return metadata
    b=len(metadata.num_draft_tokens)
    if metadata.num_draft_tokens!=[5]*b or metadata.draft_token_ids.numel()!=5*b:raise RuntimeError('ABC gated rejection geometry differs from router')
    return dataclasses.replace(metadata,draft_token_ids=torch.where(MASK[INDICES[b]]!=0,metadata.draft_token_ids,torch.full_like(metadata.draft_token_ids,-1)))
