"""Optional bounded startup proof through the SAME retained attention kernel.

No torch import or execution on import. No speed/quality claim from this test.
"""
import gc
import inspect
from .qualification import fixture
from .source_contract import check_function_source


def qualify(torch, reference_combine, candidate, attention, device):
    check_function_source(inspect.getsource(attention), 'rocm_sparse_attn_prefill')
    records=[]
    for ratio in (0,1,2):
        for requests in (1,2,8):
            args=dense=lens=flat=ptr=q=kv=sink=expected=actual=None
            try:
                if torch.cuda.mem_get_info(device)[0]-128*1024**2 < 1024**3:
                    raise ValueError('Attention fixture cannot retain startup floor/reserve')
                spec=dict(rows=129,requests=requests,ratio=ratio,window=128,
                          topk=0 if ratio==0 else 512,input_width=512,base=97,
                          dtype='int32',metadata_dtype='int32',layout='2d',lead_empty=True)
                args=list(fixture(torch,spec,device))
                # Bounded physical KV capacity while preserving the exact same
                # integer contract; current uncompressed max-context pool would
                # consume unnecessary256MiB per rank just for this fixture.
                args[-2],args[-1]=1024,0 if ratio==0 else 512
                limit=requests*1024
                dense,lens=reference_combine(*args)  # ACTUAL CUDA ratio0 call.
                flat,ptr=candidate(*args,num_rows=limit)
                gen=torch.Generator(device=device).manual_seed(40468)
                q=torch.randn((129,16,512),dtype=torch.bfloat16,device=device,generator=gen)*.125
                kv=torch.randn((limit,1,512),dtype=torch.bfloat16,device=device,generator=gen)*.125
                sink=torch.zeros((16,),dtype=torch.float32,device=device)
                expected=torch.empty_like(q);actual=torch.empty_like(q)
                common=dict(q=q,kv=kv,scale=512**-.5,head_dim=512,nope_head_dim=448,
                            rope_head_dim=64,attn_sink=sink)
                attention(**common,indices=dense,topk_length=lens,output=expected)
                attention(**common,indices=None,topk_length=None,output=actual,
                          ragged_indices=flat,ragged_indptr=ptr)
                torch.cuda.synchronize(device)
                exact=bool(torch.equal(expected.view(torch.int16),actual.view(torch.int16)))
                finite=bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
                records.append(dict(ratio=ratio,requests=requests,rows=129,
                                    bitwise_equal=exact,finite=finite))
                if not exact or not finite:
                    raise ValueError('Same-kernel ragged attention parity failure: '+str(records[-1]))
                if torch.cuda.mem_get_info(device)[0] < 1024**3:
                    raise ValueError('Attention fixture crossed1GiB floor')
            finally:
                args=dense=lens=flat=ptr=q=kv=sink=expected=actual=None
                # Drop dict-held q/kv references too, including on a rejection.
                if 'common' in locals():common.clear()
                gc.collect();torch.cuda.empty_cache()
    return dict(passed=True,cases=len(records),records=records,
                same_attention_kernel=True,model_quality_verified=False,performance_measured=False)
