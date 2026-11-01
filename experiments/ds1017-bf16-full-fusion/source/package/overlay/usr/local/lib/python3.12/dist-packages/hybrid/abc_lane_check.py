"""Bounded actual-pack lane fixtures against separate reconstructed FP64 math.

The ExLlamaV3 codec decoder is shared with the established reference; the new
lane routing, input transform, GEMM, activation and reduction are not reused.
"""
def had128(x):
    import math
    shape=x.shape;y=x.reshape(-1,128)
    for stride in (1,2,4,8,16,32,64):
        z=y.reshape(-1,128//(2*stride),2,stride)
        a,b=z[:,:,0],z[:,:,1]
        y=__import__('torch').stack((a+b,a-b),dim=2).reshape(-1,128)
    return (y/math.sqrt(128)).reshape(shape)

def prepare(model):
    import torch
    from vllm_exl3.exl3 import load_exllamav3_ext
    from . import abc_kernels as K
    from .abc_numeric import metrics
    ext=load_exllamav3_ext();seen=set();records=[]
    for name,layer in model.named_modules():
        tc=getattr(layer,'_exl3_tc',None);inners=getattr(layer,'_exl3_inners',None)
        if tc is None or not inners:continue
        geometry=(tc['intermediate'],tc['bits_x2'])
        if geometry in seen:continue
        seen.add(geometry);decoded={}
        for e in (0,1):
            for p in ('gate','up','down'):
                linear=inners[e][p];shape=(linear.trellis.shape[0]*16,linear.trellis.shape[1]*16)
                weight=torch.empty(shape,device=linear.trellis.device,dtype=torch.float16)
                ext.reconstruct(weight,linear.trellis,linear.K,linear.mcg,linear.mul1)
                decoded[e,p]=weight
        def project(x,e,p):
            linear=inners[e][p]
            return had128(had128(x*linear.suh.double())@decoded[e,p].double())*linear.svh.double()
        for rows in (4,6,12,48):
            gen=torch.Generator(device=tc['workspace'].device).manual_seed(1017+rows)
            x=torch.randn((rows,5120),device=tc['workspace'].device,dtype=torch.bfloat16,generator=gen)*.125
            for kind in ('repeated','mixed_sentinels','all_sentinels'):
                ids=(torch.arange(rows*6,device=x.device).reshape(rows,6)%2).long()
                if kind=='repeated':ids.zero_()
                elif kind=='mixed_sentinels':ids[:,1]=-1;ids[:,4]=384
                else:ids.fill_(384)
                weights=torch.full((rows,6),1/6,device=x.device,dtype=torch.float32)
                def run():
                    out=torch.empty_like(x)
                    K.LANE_OPS.moe_out_v2_try(x,ids,weights,tc['ptrs'],tc['meta'],tc['emap'],tc['intermediate'],tc['bits_x2'],out,K.LANE_WS,10.)
                    return out
                actual=run();repeat=run()
                if not torch.equal(actual,repeat):raise RuntimeError('Lane output did not own-repeat exactly')
                ref=torch.zeros_like(x,dtype=torch.float64)
                for e in (0,1):
                    g=project(x.double(),e,'gate').clamp(max=10.)
                    u=project(x.double(),e,'up').clamp(-10.,10.)
                    act=g/(1+torch.exp(-g))*u
                    down=project(act,e,'down')
                    coeff=torch.where(ids==e,weights,0.).double().sum(1)
                    ref+=down*coeff[:,None]
                if kind=='all_sentinels' and (not bool((actual==0).all()) or not bool((ref==0).all())):raise RuntimeError('Invalid expert sentinels contributed to output')
                records.append(dict(module=name,intermediate=geometry[0],bits_x2=geometry[1],rows=rows,routes=kind,own_exact=True,
                    reference='ExLlamaV3 reconstructed inner weights + independent torch FP64 normalized H128, clamp, SiLU, expert weighted sum',
                    shared_reference_component='Established ExLlamaV3 codec decode only',model_quality_verified=False,**metrics(actual,ref)))
        del decoded
    if not records:raise RuntimeError('No actual lane geometry had an independent reference fixture')
    return dict(actual_geometries=[list(x) for x in sorted(seen)],unexercised_geometries=[list(x) for x in ((512,4),(512,6),(640,4),(640,6)) if x not in seen],records=records,model_quality_verified=False)
