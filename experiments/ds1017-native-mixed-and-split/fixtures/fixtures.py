"""CPU generated fp8_ds_mla byte layout and independent FP64 reference.
No model tensors, no selection change between attention arms.
"""
import numpy as np
import torch

def cache(rows,block=128,seed=26):
    assert rows%block==0
    rng=np.random.default_rng(seed)
    raw=np.zeros((rows//block,block*584),dtype=np.uint8)
    data=raw[:,:block*576].reshape(-1,block,576)
    # Finite E4M3 magnitudes and signs, independent BF16 RoPE tail.
    data[:,:,:448]=rng.integers(32,64,size=(rows//block,block,448),dtype=np.uint8)
    data[:,:,:448] |= rng.integers(0,2,size=data[:,:,:448].shape,dtype=np.uint8)*128
    rope=torch.from_numpy(rng.normal(0,.15,(rows//block,block,64)).astype(np.float32)).to(torch.bfloat16)
    data[:,:,448:576]=rope.view(torch.uint8).numpy()
    raw[:,block*576:]=rng.integers(124,129,size=(rows//block,block*8),dtype=np.uint8)
    return torch.from_numpy(raw.reshape(rows//block,block,584))

def decode(cache,ids):
    """Decode only chosen physical rows, preserving per-operation BF16 rounding."""
    block=cache.shape[1];raw=cache.reshape(cache.shape[0],-1)
    out=torch.zeros(len(ids),512,dtype=torch.bfloat16)
    for j,i in enumerate(ids):
        if i<0 or i>=cache.shape[0]*block:continue
        b,p=divmod(i,block)
        d=raw[b,p*576:(p+1)*576].contiguous()
        codes=d[:448].view(torch.float8_e4m3fn).to(torch.bfloat16)
        e=raw[b,block*576+p*8:block*576+(p+1)*8][:7]
        scales=torch.exp2(e.float()-127).to(torch.bfloat16).repeat_interleave(64)
        vals=codes*scales
        vals=torch.where(vals.isnan(),0,vals)
        rope=d[448:576].view(torch.bfloat16)
        out[j]=torch.cat((vals,torch.where(rope.isnan(),0,rope)))
    return out

def indexes(rows,context,ratio=1,case='regular',seed=26):
    rng=np.random.default_rng(seed);main=[];extra=[];visible=[]
    for r in range(rows):
        n=(context-(rows-1-r))//ratio;visible.append(n)
        position=context-(rows-1-r)-1
        m=[(position-127+j)%256 for j in range(128)]
        if case=='blocked':
            blocks=rng.choice(n//8,min(2048,n//8),replace=False)
            allowed=np.sort((blocks[:,None]*8+np.arange(8)).reshape(-1))
        else:allowed=np.arange(n)
        ex=sorted(rng.choice(allowed,min(512,len(allowed)),replace=False).tolist())
        if case=='unsorted':rng.shuffle(ex)
        if case=='empty':m=[];ex=[]
        if case=='short':m=[r%256] if r%2 else [];ex=[max(n-1,0)] if r%3 else []
        if case=='invalid':m=[-1,256];ex=[-1,context//ratio]
        if case=='swa_only':ex=[]
        main.append(m);extra.append(ex)
    return main,extra,visible

def ragged(rows):
    lengths=[len(r) for r in rows]
    return torch.tensor([v for r in rows for v in r],dtype=torch.int32),torch.tensor([0]+np.cumsum(lengths).tolist(),dtype=torch.int32)

def reference(q,main_cache,extra_cache,main,extra,sink,scale):
    outputs=[]
    for r in range(q.shape[0]):
        mv=[i for i in main[r] if 0<=i<main_cache.shape[0]*main_cache.shape[1]]
        ev=[i for i in extra[r] if 0<=i<extra_cache.shape[0]*extra_cache.shape[1]]
        kv=torch.cat((decode(main_cache,mv),decode(extra_cache,ev))).double()
        if not len(kv):outputs.append(torch.zeros_like(q[r],dtype=torch.float64));continue
        logits=q[r].double()@kv.T*scale
        if sink is None:probs=logits.softmax(-1)
        else:probs=torch.cat((logits,sink.double()[:,None]),-1).softmax(-1)[:,:-1]
        outputs.append(probs@kv)
    return torch.stack(outputs)

if __name__=='__main__':
    c=cache(256);assert c.shape==(2,128,584)
    assert torch.equal(decode(c,[-1,256]),torch.zeros(2,512,dtype=torch.bfloat16))
    q=torch.zeros(1,16,512,dtype=torch.bfloat16)
    out=reference(q,c,c,[[0]],[[1]],torch.zeros(16),1.)
    expected=(decode(c,[0,1]).double().sum(0)/3).expand(16,512)
    assert torch.allclose(out[0],expected,atol=1e-15,rtol=1e-14)
    assert not reference(q,c,c,[[]],[[]],None,1.).any()
    m,e,v=indexes(6,32768,case='blocked')
    assert all(len(set(x))==512 and max(x)<n for x,n in zip(e,v))
    m,e,v=indexes(6,131072,case='unsorted')
    assert all(x!=sorted(x) and len(set(x))==512 and max(x)<n for x,n in zip(e,v))
    print('PASS fp8_ds_mla layout, invalid rows, empty rows, exact single sink, blocked/unsorted unique visible IDs')
