"""Startup proof that dense scratch writes preserve both retained MoE paths."""
import ctypes
import gc
import hashlib

class UnsafeWorkspace(RuntimeError):
    pass

def digest_output(torch, value):
    if not bool(torch.isfinite(value).all()):
        raise UnsafeWorkspace('Nonfinite MoE scratch interoperability output')
    cpu = value.contiguous().view(torch.uint8).cpu()
    return hashlib.sha256(ctypes.string_at(cpu.data_ptr(), cpu.numel())).hexdigest()

def fixture(torch, device, rows):
    g=torch.Generator(device=device).manual_seed(40468)
    x=torch.randn((rows,5120),device=device,dtype=torch.bfloat16,generator=g)*.125
    ids=torch.arange(rows*6,device=device,dtype=torch.int64).view(rows,6)%384
    weights=torch.rand((rows,6),device=device,dtype=torch.float32,generator=g)
    weights/=weights.sum(-1,keepdim=True)
    return x,ids,weights

def baseline(torch, original, representatives, device):
    results={}
    for key,layer in sorted(representatives.items()):
        for rows in (65,1024,2048):
            x=ids=w=y=None
            try:
                if torch.cuda.mem_get_info(device)[0]-128*1024**2<1024**3:
                    raise ValueError('Interop fixtures cannot preserve startup reserve')
                x,ids,w=fixture(torch,device,rows)
                y=original(x,ids,w,layer,None,7.0)
                results[(*key,rows)]=digest_output(torch,y)
            finally:
                x=ids=w=y=None;gc.collect();torch.cuda.empty_cache()
    return results

def qualify(torch, original, adapter, representatives, device, expected, dense=None):
    records=[]
    binding=None if dense is None else max(dense.bindings.values(),key=lambda b:b['N']*b['K'])
    for key,layer in sorted(representatives.items()):
        for rows in (65,1024,2048):
            x=ids=w=y=dx=dy=None
            try:
                if torch.cuda.mem_get_info(device)[0]-128*1024**2<1024**3:
                    raise UnsafeWorkspace('Interop fixtures cannot preserve startup reserve')
                x,ids,w=fixture(torch,device,rows)
                if binding is not None:
                    dx=torch.zeros((65,binding['K']),device=device,dtype=torch.bfloat16)
                row=dict(intermediate=key[0],bits_x2=key[1],rows=rows,dense_write=binding is not None)
                for name in ('retained','candidate'):
                    if binding is not None:
                        dy=dense._compute(binding,dx);dy=None
                    y=(original(x,ids,w,layer,None,7.0) if name=='retained'
                       else adapter.apply(x,ids,w,layer._exl3_tc,None,7.0))
                    actual=digest_output(torch,y);y=None
                    row[name+'_bitwise_equal']=actual==expected[(*key,rows)]
                    if not row[name+'_bitwise_equal']:
                        raise UnsafeWorkspace('Dense write changed '+name+' MoE: '+str(row))
                records.append(row)
            finally:
                x=ids=w=y=dx=dy=None;gc.collect();torch.cuda.empty_cache()
    return dict(passed=True,cases=len(records),records=records,
                baseline_digests=[dict(intermediate=k[0],bits_x2=k[1],rows=k[2],sha256=v) for k,v in sorted(expected.items())])
