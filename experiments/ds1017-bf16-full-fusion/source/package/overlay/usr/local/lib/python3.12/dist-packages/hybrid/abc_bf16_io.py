"""BF16 paged prefill gather; packed FP8 dispatch stays on the retained path."""
import torch
from vllm.triton_utils import triton,tl

@triton.jit
def _gather(out,cache,seq,lens,table,OS0:tl.constexpr,OS1:tl.constexpr,
            CS0:tl.constexpr,CS1:tl.constexpr,TS0:tl.constexpr,
            BLOCK:tl.constexpr,OFFSET:tl.constexpr):
    row=tl.program_id(0);worker=tl.program_id(1)
    seq_len=tl.load(seq+row)
    if lens is None:count=seq_len
    else:count=tl.load(lens+row)
    d=tl.arange(0,512)
    for i in range(worker,count,tl.num_programs(1)):
        pos=seq_len-count+i
        physical=tl.load(table+row*TS0+pos//BLOCK).to(tl.int64)
        value=tl.load(cache+physical*CS0+(pos%BLOCK)*CS1+d)
        tl.store(out+row*OS0+(OFFSET+i)*OS1+d,value)

def gather_dispatch(out,k_cache,seq_lens,gather_lens,block_table,block_size,offset,use_fnuz=False):
    if k_cache.dtype!=torch.bfloat16:
        from vllm.models.deepseek_v4_1.common.ops import dequantize_and_gather_k_cache
        return dequantize_and_gather_k_cache(out,k_cache,seq_lens,gather_lens,block_table,block_size,offset,use_fnuz=use_fnuz)
    if (k_cache.ndim!=3 or tuple(k_cache.shape[1:])!=(block_size,512)
            or k_cache.stride(1)!=512 or k_cache.stride(2)!=1
            or out.dtype!=torch.bfloat16 or out.shape[-1]!=512 or out.stride(-1)!=1
            or block_table.stride(-1)!=1 or use_fnuz):
        raise ValueError('Unqualified plain BF16 prefill gather')
    _gather[(seq_lens.numel(),128)](out,k_cache,seq_lens,gather_lens,block_table,
        out.stride(0),out.stride(1),k_cache.stride(0),k_cache.stride(1),block_table.stride(0),
        block_size,offset,num_warps=4)
