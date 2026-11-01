"""Mandatory generated GPU fixtures before BF16 graph publication; no model data.

These check implementation/layout, not task quality or a BF16-vs-FP8 ranking.
No RNG state is consumed and all allocations are local, bounded fixtures.
"""
def qualify(device,block_bytes):
    import torch
    from . import abc_bf16_read as op
    from .abc_bf16_io import gather_dispatch
    from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import rope_quant_insert
    assert not torch.cuda.is_current_stream_capturing()
    if torch.cuda.get_device_capability(device)!=(8,0):raise ValueError('BF16 twin is qualified only for SM80')
    rows=[];stride=block_bytes//2
    def values(shape,phase):
        import math
        n=math.prod(shape)
        return (torch.sin(torch.arange(n,device=device,dtype=torch.float32)*.137+phase)*.7).reshape(shape).to(torch.bfloat16)
    def cache(block):
        raw=torch.full((3*stride,),-9.,device=device,dtype=torch.bfloat16)
        t=raw.as_strided((3,block,512),(stride,512,1))
        t.copy_(values(t.shape,.43+block))
        return raw,t
    def record(label,actual,expected,*,atol=.012,rtol=.025):
        expected=expected.to(device=device,dtype=torch.float64)
        if not bool(torch.isfinite(actual).all()):raise ValueError(label+': nonfinite result')
        err=(actual.double()-expected).abs()
        if not bool((err<=atol+rtol*expected.abs()).all()):
            raise ValueError(label+': independent reference bound exceeded '+str(float(err.max())))
        rows.append(dict(fixture=label,max_abs=float(err.max()),atol=atol,rtol=rtol))
    def rotate(x,pos,cs):
        y=x.double().clone();z=y[...,448:].reshape(*y.shape[:-1],32,2)
        co=cs[pos,:32].double();si=cs[pos,32:].double()
        while co.ndim<z.ndim-1:co=co.unsqueeze(1);si=si.unsqueeze(1)
        even=z[...,0].clone();odd=z[...,1].clone()
        z[...,0]=even*co-odd*si;z[...,1]=even*si+odd*co
        return y
    # Native fused Q normalization/RoPE and BF16 insert: noncompact physical
    # block stride, cross-page writes, negative slot and guard bytes.
    raw,main=cache(32);before=raw.clone();q=values((3,16,512),.22);kv=values((3,512),.67)
    pos=torch.tensor([0,17,16383],device=device,dtype=torch.int64)
    angle=torch.arange(16384*32,device=device,dtype=torch.float32).reshape(16384,32)*.003
    cs=torch.cat((angle.cos(),angle.sin()),1).contiguous()
    slots=torch.tensor([-1,32,65],device=device,dtype=torch.int64)
    q_input=q.clone()
    qr=rotate(q,pos,cs)  # Actual serving path passes apply_q_norm=False.
    kr=rotate(kv,pos,cs)
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(q,kv,main,slots,pos,cs,1e-6,32,False)
    record('native_rope_without_duplicate_qnorm',q,qr,atol=.004,rtol=.008)
    # A cache-precision comparison must not secretly change Q normalization.
    fp8raw=torch.zeros(3*230400,device=device,dtype=torch.uint8)
    fp8cache=fp8raw.as_strided((3,32*584),(230400,1))
    q_fp8=torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(q_input,kv,fp8cache,slots,pos,cs,16,1e-6,32,False)
    if not torch.equal(q_fp8,q):raise ValueError('FP8 and BF16 native insert paths changed Q; cache-only isolation unavailable')
    rows.append(dict(fixture='native_q_identical_to_fp8_writer',bitwise=True))
    q_true=q_input.clone()
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(q_true,kv,main,slots,pos,cs,1e-6,32,True)
    qr_true=rotate(q_input.double()*torch.rsqrt(q_input.double().square().mean(-1,keepdim=True)+1e-6),pos,cs)
    record('native_optional_qnorm_rope',q_true,qr_true,atol=.004,rtol=.008)
    q_fp8_true=torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(q_input,kv,fp8cache,slots,pos,cs,16,1e-6,32,True)
    rows.append(dict(fixture='optional_qnorm_fp8_writer_comparison_not_serving_path',bitwise=bool(torch.equal(q_fp8_true,q_true))))
    expected=before.clone();view=expected.as_strided(main.shape,main.stride())
    for i,slot in enumerate((-1,32,65)):
        if slot>=0:view[slot//32,slot%32]=kr[i].to(torch.bfloat16)
    record('native_bf16_insert_and_guard',raw,expected,atol=.004,rtol=.008)
    for block in (64,128):
        eraw,extra=cache(block);saved=eraw.clone();latent=values((4,512),.76)
        p=torch.tensor([0,1,2,3],device=device,dtype=torch.int64)
        sl=torch.tensor([0,block,block+1,-1],device=device,dtype=torch.int64)
        for ratio in (1,2):
            eraw.copy_(saved)
            rope_quant_insert(latent,p,cs,extra,sl,ratio)
            expected=saved.clone();view=expected.as_strided(extra.shape,extra.stride())
            for i,slot in enumerate((0,block,block+1,-1)):
                if slot>=0 and (i+1)%ratio==0:
                    view[slot//block,slot%block]=rotate(latent[i:i+1],(p[i:i+1]//ratio)*ratio,cs)[0].to(torch.bfloat16)
            record(f'compressed_insert_{block}_ratio{ratio}_guards',eraw,expected,atol=.004,rtol=.008)
        # Prefill gather must respect request-specific tail lengths and output offsets.
        seq=torch.tensor([block+3,block+1],device=device,dtype=torch.int32)
        table=torch.tensor([[2,0],[1,2]],device=device,dtype=torch.int32)
        for tail in (None,torch.tensor([5,3],device=device,dtype=torch.int32)):
            out=torch.full((2,2*block+9,512),-7.,device=device,dtype=torch.bfloat16)
            expected=out.clone()
            for r,n in enumerate((block+3,block+1)):
                count=n if tail is None else (5,3)[r]
                for i,j in enumerate(range(n-count,n)):
                    physical=((2,0),(1,2))[r][j//block]
                    expected[r,4+i]=extra[physical,j%block]
            gather_dispatch(out,extra,seq,tail,table,block,4)
            if not torch.equal(out,expected):raise ValueError('BF16 gather lost bytes or wrote guards')
            rows.append(dict(fixture=f'gather_{block}_tail{tail is not None}',bitwise=True))
        # Both unsplit fallback and optimized split attention read identical
        # BF16 rows. Reference computes FP64 softmax/value sum with sink.
        for n in (1,4,6,12,48):
            q=values((n,16,512),.32)
            mi=torch.tensor([-1,0,33,64,0,999999],device=device,dtype=torch.int32).repeat(n)
            ei=torch.tensor([block+1,0,-1,2*block+2],device=device,dtype=torch.int32).repeat(n)
            mp=torch.arange(n+1,device=device,dtype=torch.int32)*6
            ep=torch.arange(n+1,device=device,dtype=torch.int32)*4
            # Last row is a genuinely empty segment.
            if n>1:mp[-1]=mp[-2];ep[-1]=ep[-2]
            for sink in (None,torch.linspace(-1,1,16,device=device,dtype=torch.float32)):
                selected=torch.cat((main[0,0],main[1,1],main[2,0],main[0,0],extra[1,1],extra[0,0],extra[2,2])).reshape(7,512).double()
                score=q.double()@selected.T*(512**-.5)
                maxv=score.max(-1,keepdim=True).values
                if sink is not None:maxv=torch.maximum(maxv,sink.double()[None,:,None])
                probs=(score-maxv).exp();denom=probs.sum(-1,keepdim=True)
                if sink is not None:denom+=(sink.double()[None,:,None]-maxv).exp()
                expected=(probs@selected)/denom
                if n>1:expected[-1]=0
                got=op._rocm_sparse_attn_decode_ragged_triton(q,main,mi,mp,512**-.5,sink,448,64,extra,ei,ep)
                record(f'ragged_{block}_rows{n}_sink{sink is not None}',got,expected)
                if n==6:
                    sw=selected[:4];scores=q.double()@sw.T*(512**-.5)
                    mx=scores.max(-1,keepdim=True).values
                    if sink is not None:mx=torch.maximum(mx,sink.double()[None,:,None])
                    pp=(scores-mx).exp();dd=pp.sum(-1,keepdim=True)
                    if sink is not None:dd+=(sink.double()[None,:,None]-mx).exp()
                    want=(pp@sw)/dd;want[-1]=0
                    got_swa=op._rocm_sparse_attn_decode_ragged_triton(q,main,mi,mp,512**-.5,sink,448,64)
                    record(f'swa_only_{block}_sink{sink is not None}',got_swa,want)
                for splits in (4,8,16):
                    m=torch.empty((n,splits,16),device=device,dtype=torch.float32);l=torch.empty_like(m)
                    a=torch.empty((n,splits,16,512),device=device,dtype=torch.float32);out=torch.empty_like(q)
                    op._sparse_attn_decode_partial_kernel[(n,splits,1)](q,main,mi,mp,extra,ei,ep,m,l,a,q,
                        q.stride(0),q.stride(1),main.stride(0),extra.stride(0),m.stride(0),m.stride(1),a.stride(0),a.stride(1),a.stride(2),
                        96,3*block,32,block,512**-.5,16,HAS_EXTRA=True,NOPE_DIM=448,NOPE_BLOCK=512,ROPE_DIM=64,
                        IS_FNUZ_MAIN=False,IS_FNUZ_EXTRA=False,BLOCK_H=16,BLOCK_K=32,NUM_SPLITS=splits,NUM_STAGES=1,num_warps=8 if splits==8 else 4)
                    op._sparse_attn_decode_reduce_kernel[(n,16)](m,l,a,sink if sink is not None else q,out,
                        out.stride(0),out.stride(1),m.stride(0),m.stride(1),a.stride(0),a.stride(1),a.stride(2),16,
                        HAS_ATTN_SINK=sink is not None,ADAPTIVE_SPLITS=False,COMB_DIM=512,BLOCK_H=1,NUM_SPLITS=splits,SPLITS_PAD=splits,num_warps=4)
                    record(f'split{splits}_{block}_rows{n}_sink{sink is not None}',out,expected)
    torch.cuda.synchronize(device)
    return dict(passed=True,generated_only=True,model_quality_verified=False,fixtures=rows)
