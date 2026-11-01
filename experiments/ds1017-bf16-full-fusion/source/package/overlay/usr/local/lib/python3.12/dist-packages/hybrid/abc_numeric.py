"""Independent bounded operator checks at first load; not model-quality scores."""
import os
DENSE=[]
SHAPES=set()

def metrics(actual,reference):
    import torch
    a,r=actual.double(),reference.double();d=a-r
    if not bool(torch.isfinite(a).all() and torch.isfinite(r).all()):raise RuntimeError('Nonfinite operator qualification fixture')
    worst=int(d.abs().argmax());flat=d.flatten();rf=r.flatten();af=a.flatten()
    return dict(max_abs=float(d.abs().max()),relative_l2=float(d.norm()/r.norm().clamp_min(1e-300)),rmse=float(d.square().mean().sqrt()),
                worst_flat_index=worst,worst_actual=float(af[worst]),worst_reference=float(rf[worst]),worst_error=float(flat[worst]))

def dense(layer,candidate):
    import torch
    import dsv41_fp8w_ext as ext
    from vllm.model_executor.kernels.linear.mxfp8 import dsv41_fp8w as g
    W,S,cnt,_=candidate;N,K=layer.weight.shape
    # This reference never uses the candidate's fragment layout or decode kernel.
    source=layer.weight.data.float()
    scale=torch.exp2(layer.weight_scale.data[:N,:K//32].view(torch.uint8).float()-127)
    exact=(source*scale.repeat_interleave(32,dim=1)).to(torch.bfloat16)
    back=torch.empty_like(exact);ext.dequant(W,S,back,N,K)
    bitwise=torch.equal(exact.view(torch.int16),back.view(torch.int16))
    if not bitwise:raise RuntimeError('Native-FP8 pack/dequant changed source weight bits')
    result=dict(N=N,K=K,source_weight_bits_exact=True,multiply_checks=[])
    if (N,K) not in SHAPES and os.environ.get('STAR_ABC_OPERATOR_CHECKS','1')=='1':
        SHAPES.add((N,K));cols=torch.cat((torch.arange(min(32,N),device=W.device),torch.arange(max(32,N-32),N,device=W.device)))
        weight=exact[cols].double();gen=torch.Generator(device=W.device).manual_seed(1017)
        for rows in (4,6,12,48):
            for kind in ('normal','alternating'):
                x=torch.randn((rows,K),dtype=torch.bfloat16,device=W.device,generator=gen)*.125
                if kind=='alternating':x[:,1::2]=-x[:,::2]
                a=g._fp8_linear(x,W,S,cnt,None,None,None,N,K)[:,cols]
                ref=x.double()@weight.T;absolute_sum=x.double().abs()@weight.abs().T
                # Conservative textbook forward-error guard: BF16 rounding plus
                # FP32 multiply/add summation. This is NOT an accuracy-loss bound.
                eps32=torch.finfo(torch.float32).eps;gamma=(2*K+64)*eps32/(1-(2*K+64)*eps32)
                bound=torch.finfo(torch.bfloat16).eps*ref.abs()+gamma*absolute_sum+torch.finfo(torch.bfloat16).tiny
                if not bool(((a.double()-ref).abs()<=bound).all()):raise RuntimeError('Native-FP8 multiply exceeded conservative FP32/BF16 operator bound')
                result['multiply_checks'].append(dict(rows=rows,input=kind,columns=len(cols),reference='independently decoded original codes/scales; FP64 dot',
                    conservative_forward_error_bound_pass=True,max_bound_fraction=float(((a.double()-ref).abs()/bound).max()),**metrics(a,ref)))
    DENSE.append(result)

def woa(module,weight,q):
    import torch
    from vllm.models.deepseek_v4_1.ampere import dsv41_wo_a as w
    G,I,D=weight.shape;codes,scales=q
    decoded=(codes.view(torch.float8_e4m3fn).float().view(G,I//32,32,D//32,32)*scales[:,:,None,:,None]).to(torch.bfloat16).view_as(weight)
    if not torch.equal(weight.view(torch.int16),decoded.view(torch.int16)):raise RuntimeError('wo_a recovery changed weight bits')
    records=[]
    if os.environ.get('STAR_ABC_OPERATOR_CHECKS','1')=='1':
        gen=torch.Generator(device=weight.device).manual_seed(1017)
        cols=torch.cat((torch.arange(min(16,I),device=weight.device),torch.arange(max(16,I-16),I,device=weight.device)))
        for rows in (4,6,12):
            x=torch.randn((rows,G,D),device=weight.device,dtype=torch.bfloat16,generator=gen)*.125
            a=w.wo_a(x.reshape(rows,G*D),codes,scales,G,I,D).view(rows,G,I)[:,:,cols]
            ref=torch.einsum('tgd,grd->tgr',x.double(),weight[:,cols].double())
            records.append(dict(rows=rows,component='projection_after_RoPE',reference='FP64 grouped dot of retained original BF16 weights',**metrics(a,ref)))
        # End-to-end fused inverse-RoPE+CUDA projection, real cos/sin cache and
        # populated positional range. Keep witnesses; do not invent a semantic tolerance.
        cache=module.rotary_emb.cos_sin_cache;head_dim=module.head_dim;heads=D*G//head_dim
        for rows in (4,6):
            pos=torch.tensor([0,1,min(16383,len(cache)-1),len(cache)-1]+[7,31][:rows-4],device=weight.device,dtype=torch.long)
            o=torch.randn((rows,heads,head_dim),device=weight.device,dtype=torch.bfloat16,generator=gen)*.125
            if not w.cuda_small_ok(o) or cache.dtype!=torch.float32:continue
            a=w._wo_a_small(o,pos,cache,codes,scales,G,I,module.rope_head_dim).view(rows,G,I)[:,:,cols]
            half=module.rope_head_dim//2;rot=o.float().clone();pair=o[...,-module.rope_head_dim:].float().reshape(rows,heads,half,2)
            cs=cache[pos].float();cos=cs[:,:half,None].transpose(1,2);sin=cs[:,half:half*2,None].transpose(1,2)
            # GPT-J inverse rotation: [x*cos+y*sin, y*cos-x*sin], BF16
            # activation boundary matches the unfused projection contract.
            rotated=torch.stack((pair[...,0]*cos+pair[...,1]*sin,pair[...,1]*cos-pair[...,0]*sin),dim=-1)
            rot[...,-module.rope_head_dim:]=rotated.flatten(-2)
            x=rot.to(torch.bfloat16).reshape(rows,G,D)
            ref=torch.einsum('tgd,grd->tgr',x.double(),weight[:,cols].double())
            records.append(dict(rows=rows,component='fused_inverse_RoPE_and_projection',positions=pos.cpu().tolist(),reference='independent torch GPT-J inverse rotation then FP64 grouped dot',**metrics(a,ref)))
    return dict(weight_bits_exact=True,checks=records,model_quality_verified=False)
