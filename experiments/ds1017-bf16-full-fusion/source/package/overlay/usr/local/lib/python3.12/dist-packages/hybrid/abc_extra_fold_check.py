"""Future first-load extra-row checks. Staged only; never invoke on serving traffic."""
def prepare(runner):
    import torch
    from . import abc_extra_fold as N,abc_runtime
    from .abc_lane_check import had128
    from w9_exl_ab import fold_serve as retained_fold
    from .abc_numeric import metrics
    from vllm_exl3.exl3 import load_exllamav3_ext
    if getattr(runner,'_w26_serving_seen',True) or getattr(runner,'_extra_fold_attempted',False):raise RuntimeError('Narrow checks require initial closed-owner preparation exactly once')
    if abc_runtime.context() is not None or torch.cuda.is_current_stream_capturing():raise RuntimeError('Narrow operator checks must precede target capture')
    runner._extra_fold_attempted=True
    ext=load_exllamav3_ext();N._extra_lib(2);N._extra_lib(3);records=[];seen=set()
    for name,layer in runner.get_model().named_modules():
        tc=getattr(layer,'_exl3_tc',None);inners=getattr(layer,'_exl3_inners',None);ptrs=getattr(layer,'_exl3_ptrs',None)
        if tc is None or not inners or not ptrs or getattr(layer,'_exl3_mixedk_unified',False):continue
        I,k=int(tc['intermediate']),int(getattr(layer,'_exl3_k',0));geometry=(I,k)
        if geometry in seen:continue
        if geometry not in ((512,2),(512,3),(640,2),(640,3)):raise RuntimeError('Unexpected actual uniform EXL3 geometry')
        if tc['bits_x2']!=2*k or len(inners)!=384:raise RuntimeError('Actual-pack narrow geometry/TP layout differs')
        seen.add(geometry);decoded={}
        for e in (0,1):
            for p in ('gate','up','down'):
                linear=inners[e][p]
                if (linear.K,linear.mcg,linear.mul1)!=(k,False,True):raise RuntimeError('Actual-pack narrow codec contract differs')
                shape=(linear.trellis.shape[0]*16,linear.trellis.shape[1]*16)
                weight=torch.empty(shape,device=linear.trellis.device,dtype=torch.float16)
                ext.reconstruct(weight,linear.trellis,linear.K,linear.mcg,linear.mul1);decoded[e,p]=weight
        def project(x,e,p):
            linear=inners[e][p]
            return had128(had128(x*linear.suh.double())@decoded[e,p].double())*linear.svh.double()
        def args_for(x,ids,rw):
            rows=len(x);slots=rows*6;smax=4*slots;dev=x.device
            hadg=torch.empty((slots,5120),device=dev,dtype=torch.float16);hadu=torch.empty_like(hadg)
            gug=torch.empty((smax,1,I),device=dev,dtype=torch.float16);guu=torch.empty_like(gug);act=torch.empty_like(gug)
            dout=torch.empty((smax,1,5120),device=dev,dtype=torch.float32)
            ctr=torch.zeros(smax*(I//128)+rows*40+2*smax+3,device=dev,dtype=torch.int32)
            out=torch.zeros((rows,5120),device=dev,dtype=torch.float32)
            keys=('gate_trellis','gate_suh','gate_svh','up_trellis','up_suh','up_svh','down_trellis','down_suh','down_svh')
            return (x,ids,rw,0,384,5120,*(ptrs[key] for key in keys),None,None,None,k,k,k,False,True,0,10.,True,hadg,hadu,gug,guu,act,dout,ctr,out,None,None)
        for rows in (4,8,12,16,20,24,28,32):
            gen=torch.Generator(device=tc['workspace'].device).manual_seed(1017+rows)
            x=(torch.randn((rows,5120),device=tc['workspace'].device,dtype=torch.bfloat16,generator=gen)*.125).half()
            for kind in ('repeated','mixed_sentinels','all_sentinels'):
                ids=(torch.arange(rows*6,device=x.device).reshape(rows,6)%2).long()
                if kind=='repeated':ids.zero_()
                elif kind=='mixed_sentinels':ids[:,1]=-1;ids[:,4]=384
                else:ids.fill_(384)
                rw=torch.full((rows,6),1/6,device=x.device,dtype=torch.float16)
                reference_args=args_for(x,ids,rw);ext.exl3_moe_coop(*reference_args);old=reference_args[33].clone()
                launcher=N.fixture_launch if rows in N.EXTRA_ROWS else retained_fold.try_launch
                new_args=args_for(x,ids,rw)
                if not launcher(new_args):raise RuntimeError('Narrow row missed the required original/extra specialization')
                new=new_args[33].clone();repeat_args=args_for(x,ids,rw)
                if not launcher(repeat_args):raise RuntimeError('Narrow own-repeat changed dispatch')
                repeat=repeat_args[33].clone()
                torch.cuda.synchronize()
                if not torch.equal(new,repeat):raise RuntimeError('Extra-row output did not own-repeat bitwise')
                if not torch.equal(new,old):raise RuntimeError('Extra-row specialization changed incumbent output bits: '+str((name,rows,kind,metrics(new,old))))
                ref=torch.zeros_like(x,dtype=torch.float64)
                for e in (0,1):
                    g=project(x.double(),e,'gate').clamp(max=10.)
                    u=project(x.double(),e,'up').clamp(-10.,10.)
                    act=g/(1+torch.exp(-g))*u
                    down=project(act,e,'down');coeff=torch.where(ids==e,rw,0.).double().sum(1);ref+=down*coeff[:,None]
                if kind=='all_sentinels':
                    if not bool((new==0).all() and (old==0).all() and (ref==0).all()):raise RuntimeError('Invalid expert sentinel contributed to output')
                elif not bool((new!=0).any()):raise RuntimeError('Nonempty actual expert fixture produced no output')
                records.append(dict(module=name,intermediate=I,kbits=k,rows=rows,routes=kind,implementation='separate_extra_library' if rows in N.EXTRA_ROWS else 'retained_original_library',incumbent_bitwise=True,own_repeat_bitwise=True,reference='Actual EXL3 inner weights reconstructed with established codec; independent FP64 normalized Hadamard, clamp, SiLU and weighted sum. Incumbent FP16 fold rounding retained.',**metrics(new,ref)))
                del reference_args,new_args,repeat_args,old,new,repeat,ref
        del decoded
    if seen!={(512,2),(512,3),(640,2),(640,3)} or len(records)!=96:raise RuntimeError('Missing actual-pack extra-row cases')
    N.READY=True
    return dict(passed=True,cases=len(records),geometries=[list(x) for x in sorted(seen)],rows=[4,8,12,16,20,24,28,32],extra_rows=list(N.EXTRA_ROWS),all_incumbent_bits_exact=True,all_own_repeat_bits_exact=True,records=records,model_quality_verified=False)
