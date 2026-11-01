"""Exercise actual runtime allocator/spec layouts on CPU, including FP32 state.

Only CUDA eligibility and GPU numerical qualification are replaced for this
fixture. Production configure() has neither bypass. No model/cache is read.
"""
import os,json
from types import SimpleNamespace as N
from unittest.mock import patch
import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec,SlidingWindowMLASpec,KVCacheConfig,KVCacheGroupSpec,KVCacheTensor
from vllm.v1.worker.utils import allocate_kv_cache
from vllm.v1.kv_cache_interface import KVCacheLayout
from hybrid import abc_kv_contract as kv,abc_bf16_check

def main():
    def mla(ratio,dtype=torch.bfloat16,head=512,cache='bfloat16'):
        return MLAAttentionSpec(block_size=128,num_kv_heads=1,head_size=head,dtype=dtype,tokens_per_state=ratio,cache_dtype_str=cache,alignment=512)
    def swa(dtype,head,cache=None):
        return SlidingWindowMLASpec(block_size=32,num_kv_heads=1,head_size=head,dtype=dtype,sliding_window=128,cache_dtype_str=cache,alignment=512)
    specs=[mla(2),mla(1),swa(torch.bfloat16,512,'bfloat16'),mla(2,torch.uint8,132,None),swa(torch.float32,1024),swa(torch.bfloat16,512,'bfloat16')]
    counts=[2,1,2,1,2,1]
    groups=[KVCacheGroupSpec(layer_names=[f'g{i}_l{j}' for j in range(n)],kv_cache_spec=s) for i,(s,n) in enumerate(zip(specs,counts))]
    size=max(s.page_size_bytes*n for s,n in zip(specs,counts));blocks=3
    tensors=[KVCacheTensor(size=size*blocks,layers=g.layer_names,layer_stride=s.page_size_bytes,block_stride=size,offset=0) for s,g in zip(specs,groups)]
    cfg=KVCacheConfig(num_blocks=blocks,kv_cache_tensors=tensors,kv_cache_groups=groups)
    layout=KVCacheLayout['BLHNC']
    caches=allocate_kv_cache(cfg,torch.device('cpu'),layout,[s.block_size for s in specs])
    runner=N(cache_config=N(cache_dtype='bfloat16'),kv_cache_config=cfg,shared_kv_cache_layers={},kv_caches=list(caches.values()),
             _kernel_block_sizes=[s.block_size for s in specs],device=torch.device('cpu'))
    actual=kv.validate_view
    with patch.dict(os.environ,{'STAR_ABC':'1','STAR_ABC_KV_DTYPE':'bfloat16'}),patch.object(kv,'_geometry',None),patch.object(kv,'_receipt',None),\
         patch.object(kv,'validate_view',side_effect=lambda t,b,d,s:actual(t,b,d,s,cuda=False)),\
         patch.object(abc_bf16_check,'qualify',return_value={'cpu_fixture_only':True,'gpu_qualified':False}):
        result=kv.configure(runner)
        assert 'torch.float32' in result['retained_state_dtypes']
        assert result['bf16_views']==6 and result['block_bytes']==size
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(actual_runtime_allocator_cpu=True,geometry=result,model_loaded=False,gpu_qualified=False,cuda_initialized=False)))

if __name__=='__main__':main()
