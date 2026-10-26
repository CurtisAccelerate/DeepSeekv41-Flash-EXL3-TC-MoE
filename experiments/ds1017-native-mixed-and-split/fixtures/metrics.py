"""CPU numerical diagnostics, including worst ULP witness and reference scale."""
import torch
def metrics(got,base,reference):
    got=got.cpu();base=base.cpu();v=got.double();b=base.double()
    bits=got.view(torch.int16).int()&65535;bb=base.view(torch.int16).int()&65535
    order=lambda z:torch.where((z&32768)!=0,65535-z,z+32768)
    ulp=(order(bits)-order(bb)).abs();ulp=torch.where((v==0)&(b==0),0,ulp)
    ix=int(ulp.flatten().argmax());den=float(reference.norm());bden=float(b.norm())
    return {'finite':bool(v.isfinite().all()),'bit_exact_rate':float((bits==bb).float().mean()),
        'max_abs_vs_baseline':float((v-b).abs().max()),'l2_vs_baseline':float((v-b).norm()),
        'relative_l2_vs_baseline':float((v-b).norm())/bden if bden else None,
        'relative_l2_vs_fp64':float((v-reference).norm())/den if den else None,
        'max_abs_vs_fp64':float((v-reference).abs().max()),'reference_l2_norm':den,
        'baseline_l2_norm':bden,'reference_absmax':float(reference.abs().max()),
        'ulp_max':int(ulp.max()),'ulp_p99':float(torch.quantile(ulp.float().flatten(),.99)),
        'worst_ulp':{'flat_index':ix,'candidate':float(v.flatten()[ix]),'baseline':float(b.flatten()[ix]),'fp64':float(reference.flatten()[ix]),'abs_error':float((v-b).abs().flatten()[ix])}}
if __name__=='__main__':
    x=torch.tensor([0.,-0.,1.,-2.],dtype=torch.bfloat16);y=x.clone();y[2]=1.0078125
    m=metrics(y,x,x.double());assert m['ulp_max']==1 and m['worst_ulp']['flat_index']==2
    z=torch.zeros_like(x);assert metrics(z,z,z.double())['relative_l2_vs_fp64'] is None
    e=torch.tensor([float('nan'),float('inf'),float('-inf'),0.],dtype=torch.bfloat16)
    assert not metrics(e,e,e.double())['finite']
    print('PASS CPU numerical witness, ULP, zero reference, nonfinite diagnostics')
