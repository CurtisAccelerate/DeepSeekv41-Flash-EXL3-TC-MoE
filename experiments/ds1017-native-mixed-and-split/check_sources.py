"""CPU-only source/dependency checks. Does not import torch or initialize CUDA."""
import argparse,ast,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PINS={
    'vllm/v1/attention/ops/rocm_aiter_mla_sparse.py':'a36c0e13cf63a77c804a5d5d60ec2f6d3e62bec51de3afac4f28619f91f5f8fc',
    'vllm/v1/attention/ops/fp8_sm80.py':'4ad3bd77073051c0cb2db2ef3d307ff5047e9a29dab3714fcc65843e87a7eb6e',
}

def check(root):
    for rel,pin in PINS.items():
        actual=hashlib.sha256((root/rel).read_bytes()).hexdigest()
        if actual!=pin:raise ValueError('Retained dependency differs: '+rel)

def main():
    p=argparse.ArgumentParser();p.add_argument('--site-packages',type=Path);a=p.parse_args()
    check(ROOT/'retained')
    if a.site_packages:check(a.site_packages)
    for path in ROOT.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=path.name)
    source=ast.parse((ROOT/'retained/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py').read_text())
    funcs={n.name:n for n in source.body if isinstance(n,ast.FunctionDef)}
    count=0
    for node in ast.walk(ast.parse((ROOT/'ds1017_components/attention.py').read_text())):
        if not isinstance(node,ast.Call):continue
        fn=node.func.value if isinstance(node.func,ast.Subscript) else node.func
        if not isinstance(fn,ast.Attribute) or fn.attr not in funcs:continue
        f=funcs[fn.attr];names=[x.arg for x in f.args.args]
        used=names[:len(node.args)]+[x.arg for x in node.keywords if x.arg not in ('num_warps','num_stages')]
        assert len(used)==len(set(used)) and set(used)<=set(names)
        assert set(names[:len(names)-len(f.args.defaults)])<=set(used)
        count+=1
    assert count==2
    print('PASS: Python syntax, both attention launch signatures, retained dependency SHA256s')

if __name__=='__main__':main()
