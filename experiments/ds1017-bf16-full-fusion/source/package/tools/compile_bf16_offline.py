"""Compile BF16 read/gather SM80 variants with devices hidden; never run kernels."""
import hashlib,json
import torch,triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from hybrid import abc_bf16_read as read,abc_bf16_io as io

def compile_one(fn,constants,pointers,warps=4):
    signature={n:pointers.get(n,'i32') for n in fn.arg_names if n not in constants}
    if 'scale' in signature:signature['scale']='fp32'
    obj=triton.compile(ASTSource(fn,signature,constants),target=GPUTarget('cuda',80,32),options={'num_warps':warps})
    return dict(kernel=fn.__name__,constants=constants,num_warps=warps,ptx_sha256=hashlib.sha256(obj.asm['ptx'].encode()).hexdigest())

def main():
    rows=[]
    for has_extra in (False,True):
        for sink in (False,True):
            constants=dict(HAS_EXTRA=has_extra,HAS_ATTN_SINK=sink,NOPE_DIM=448,NOPE_BLOCK=512,ROPE_DIM=64,
                IS_FNUZ_MAIN=False,IS_FNUZ_EXTRA=False,BLOCK_H=16,BLOCK_K=16)
            ptrs={n:'*bf16' for n in read._sparse_attn_decode_ragged_kernel.arg_names if n.endswith('_ptr')}
            ptrs.update({n:'*i32' for n in ptrs if 'indices' in n or 'indptr' in n});ptrs['attn_sink_ptr']='*fp32'
            rows.append(compile_one(read._sparse_attn_decode_ragged_kernel,constants,ptrs,8))
        for splits,warps in ((4,4),(8,8),(16,4)):
            constants=dict(HAS_EXTRA=has_extra,NOPE_DIM=448,NOPE_BLOCK=512,ROPE_DIM=64,
                IS_FNUZ_MAIN=False,IS_FNUZ_EXTRA=False,BLOCK_H=16,BLOCK_K=32,NUM_SPLITS=splits,NUM_STAGES=1)
            ptrs={n:'*bf16' for n in read._sparse_attn_decode_partial_kernel.arg_names if n.endswith('_ptr')}
            ptrs.update({n:'*i32' for n in ptrs if 'indices' in n or 'indptr' in n})
            ptrs.update({n:'*fp32' for n in ptrs if n.startswith('part_')})
            rows.append(compile_one(read._sparse_attn_decode_partial_kernel,constants,ptrs,warps))
    for block in (32,64,128):
        for tail in (False,True):
            constants=dict(OS0=270336,OS1=512,CS0=196608,CS1=512,TS0=2,BLOCK=block,OFFSET=4)
            ptrs=dict(out='*bf16',cache='*bf16',seq='*i32',lens='*i32',table='*i32')
            if not tail:constants['lens']=None
            rows.append(compile_one(io._gather,constants,ptrs))
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(sm=80,offline_compiled=True,kernels_executed=False,cuda_initialized=False,variants=rows),indent=2))

if __name__=='__main__':main()
