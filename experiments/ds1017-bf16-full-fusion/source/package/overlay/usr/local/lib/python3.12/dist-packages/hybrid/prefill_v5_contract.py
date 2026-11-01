"""Audit the effective V5 integration without importing GPU backends."""
import ast
import hashlib
import importlib.util
from pathlib import Path

SITE='usr/local/lib/python3.12/dist-packages'
ROCM='vllm/models/deepseek_v4_1/amd/rocm.py'

def require(ok,message):
    if not ok:raise ValueError(message)

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def audit(package,effective_site=None):
    package=Path(package);source=package/'overlay'/SITE
    effective=source if effective_site is None else Path(effective_site)
    h=effective/'hybrid'
    pinned=load(h/'prefill_combine/source_contract.py','v5_combine_source_audit')
    original=pinned.check_original_file(package/'prefill-reference/original-rocm.py')
    rocm=(effective/ROCM).read_text();tree=ast.parse(rocm)
    retained=pinned.check_reference_source(rocm)
    statements=[ast.unparse(n) for n in tree.body]
    require('_star_combine_reference = combine_topk_swa_indices' in statements,'Retained combiner alias missing')
    require('combine_topk_swa_indices = _star_combine_dispatch' in statements,'Combiner dispatch not installed')
    dispatch=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_star_combine_dispatch')
    require([a.arg for a in dispatch.args.args]==pinned.ARGS,'Effective combiner dispatch API differs')
    require('return _star_combine_reference(*args) if candidate is None else candidate' in ast.unparse(dispatch),'Retained fallback missing')
    kernels=ast.parse((h/'abc_kernels.py').read_text())
    dense=next(n for n in kernels.body if isinstance(n,ast.FunctionDef) and n.name=='dense')
    require('prefill = dense_candidate(layer, x, bias)' in ast.unparse(dense),'Effective dense integration missing')
    require('if prefill is not None:' in ast.unparse(dense),'Dense fallback no longer preserves incumbent')
    copied=[]
    for name,tests in [('prefill_combine',package/'prefill-combine-tests'),('prefill_dense',package/'prefill-dense-tests/prefill_v5_dense')]:
        for file in ('combiner.py','qualification.py','source_contract.py') if name=='prefill_combine' else ('dense.py','qualification.py','source_contract.py'):
            a=h/name/file;b=tests/file
            require(a.read_bytes()==b.read_bytes(),'CPU fixture/runtime source differs: '+str(a))
            copied.append(dict(file=name+'/'+file,sha256=hashlib.sha256(a.read_bytes()).hexdigest()))
    init=ast.parse((h/'prefill_dense/__init__.py').read_text())
    require(any(isinstance(n,ast.ImportFrom) and n.module=='dense' and any(a.name=='prepare' for a in n.names) for n in init.body),'Actual dense package does not export prepare')
    return dict(passed=True,original=original,retained=retained,component_sources=copied,
                effective_site=str(effective),gpu_qualified=False)
