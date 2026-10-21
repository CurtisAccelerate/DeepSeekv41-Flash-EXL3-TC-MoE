"""Existing coordinate/logical-rounding proof. No torch or CUDA imports.
Does not emulate hardware MMA internal accumulation or prove GPU equivalence.
"""
import json
import numpy as np
from cpu_decoder import windows_bits, mul1
def mapping_and_folds():
    rng=np.random.default_rng(101721);checks={}
    for bits in (2,3):
        words=bytes(rng.integers(0,256,32*bits,dtype=np.uint8))
        values=np.array([mul1(s) for s in windows_bits(words,bits)],np.uint16).view(np.float16)
        w=np.empty((16,16),np.float16);A=np.empty((16,16),np.float16)
        for lane in range(32):
            group=lane//4;k=2*(lane%4);v=values[lane*8:lane*8+8]
            for j in range(8):w[k+(j%2)+8*((j%4)//2),group+8*(j//4)]=v[j]
            # Original f0=(v0v1,v2v3), f1=(v4v5,v6v7).
            for pair,(row,kk) in zip((v[:2],v[4:6],v[2:4],v[6:8]),((group,k),(group+8,k),(group,k+8),(group+8,k+8))):A[row,kk:kk+2]=pair
        assert np.array_equal(A.view(np.uint16),w.T.copy().view(np.uint16))
        checks[f'K{bits}_decoded_fragment_transpose_no_shuffle_or_repack']=True
    for nrows in range(1,9):
        counts=np.zeros((4,nrows,128),np.int8)
        for warp in range(16):
            wn=warp%4;wk=warp//4
            for lane in range(32):
                token=2*(lane&3);weightcol=lane>>2
                for t in range(2):
                    for f in range(2):
                        col=wn*32+t*16+f*8+weightcol
                        for row in (token,token+1):
                            if row<nrows:counts[wk,row,col]+=1
        assert np.all(counts==1)
    checks['unchanged_shared_reduction_layout_single_writer_all_1_to_8_rows']=True
    # Logical f16-round/f32-fold model with identical ordered K sums. This is not
    # a claim to emulate hardware MMA's undocumented internal accumulation order.
    for nrows in (1,6):
        for slices in (8,10,80):
            old=np.zeros((nrows,16),np.float16);new=np.zeros((16,nrows),np.float16)
            oldf=np.zeros_like(old,np.float32);newf=np.zeros_like(new,np.float32)
            for s in range(slices):
                x=rng.normal(0,.03,(nrows,16)).astype(np.float16)
                w=rng.normal(0,.3,(16,16)).astype(np.float16)
                po=np.zeros_like(old,np.float64);pn=np.zeros_like(new,np.float64)
                for k in range(16):
                    po+=x[:,k,None].astype(np.float64)*w[k,None,:].astype(np.float64)
                    pn+=w[k,:,None].astype(np.float64)*x[None,:,k].astype(np.float64)
                old=(old.astype(np.float64)+po).astype(np.float16);new=(new.astype(np.float64)+pn).astype(np.float16)
                if (s+1)%4==0 or s+1==slices:
                    oldf+=old.astype(np.float32);newf+=new.astype(np.float32);old[:]=0;new[:]=0
                    assert np.array_equal(oldf,newf.T)
            assert np.array_equal(old,new.T)
    checks['FOLD4_both_halves_reset_tail8_10_80_slices_logical_model']=True
    return checks

if __name__ == "__main__":
    print(json.dumps(mapping_and_folds(), indent=2))
