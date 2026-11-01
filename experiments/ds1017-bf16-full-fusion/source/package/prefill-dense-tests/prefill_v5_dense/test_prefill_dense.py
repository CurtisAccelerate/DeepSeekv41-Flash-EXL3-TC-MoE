"""CPU mocks and exact pinned-source contracts; never calls CUDA."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prefill_v5_dense.dense import DenseRunner, workspace_elements, ROWS
from prefill_v5_dense.qualification import within_bound
from prefill_v5_dense.source_contract import check_runtime

BASE = Path(__file__).resolve().parents[2]/'overlay/usr/local/lib/python3.12/dist-packages'


class FakeCUDA:
    stream = 7
    capture = False
    def current_stream(self, device): return SimpleNamespace(cuda_stream=self.stream)
    def is_current_stream_capturing(self): return self.capture


def setup_runner(fail_mm=False):
    backend = SimpleNamespace(allow_bf16_reduced_precision_reduction=True)
    seen = []
    def mm(x, w):
        seen.append(backend.allow_bf16_reduced_precision_reduction)
        if fail_mm: raise RuntimeError('injected mm failure')
        return torch.empty((x.shape[0], w.shape[1]), dtype=torch.bfloat16)
    shim = SimpleNamespace(cuda=FakeCUDA(), bfloat16=torch.bfloat16,
                           backends=SimpleNamespace(cuda=SimpleNamespace(matmul=backend)), mm=mm)
    N, K = 5120, 1024
    ws = torch.empty(workspace_elements(N, K), dtype=torch.float32)
    layer = SimpleNamespace(output_size_per_partition=N, input_size_per_partition=K,
                            _abc_w8=(torch.empty(4), torch.empty(4), torch.empty(1)),
                            weight=torch.empty(1), weight_scale=torch.empty(1), workspace=torch.empty(1), bias=None)
    dequant = []
    ext = SimpleNamespace(dequant=lambda W,S,WB,n,k: dequant.append((WB.data_ptr(), n, k)))
    calls = []
    def marlin(**kw): calls.append(kw); return None
    runner = DenseRunner(shim, ws, torch.device('cpu'), ext, marlin)
    runner.bind('actual-shaped-mock', layer)
    return runner, layer, backend, seen, dequant, calls


class CPUContract(unittest.TestCase):
    def test_workspace_formula_alias_and_three_tuple(self):
        runner, layer, _, _, _, _ = setup_runner()
        b = runner.bindings[id(layer)]
        self.assertEqual(b['wb'].data_ptr(), runner.workspace.data_ptr())
        self.assertEqual(b['wb'].numel() * 2, runner.workspace.numel() * 4)
        self.assertEqual(tuple(b['wb'].shape), (5120, 1024))
        layer._abc_w8 += (False,)
        with self.assertRaises(ValueError): runner.bind('bad', layer)
        for N,K in ((1024,1024),(16384,4096),(513,1024)):
            with self.assertRaises(ValueError): workspace_elements(N,K)

    def test_flag_restore_success_and_failure(self):
        for fail in (False,True):
            runner, layer, backend, seen, dequant, _ = setup_runner(fail)
            runner.qualified.update((5120,1024,m) for m in ROWS)
            x = torch.zeros((65,1024), dtype=torch.bfloat16)
            if fail:
                with self.assertRaisesRegex(RuntimeError,'injected'): runner.apply(layer,x)
            else:
                self.assertEqual(tuple(runner.apply(layer,x).shape),(65,5120))
            self.assertTrue(backend.allow_bf16_reduced_precision_reduction)
            self.assertEqual(seen,[False])
            self.assertFalse(runner.busy)
            self.assertEqual(dequant,[(runner.workspace.data_ptr(),5120,1024)])

    def test_exact_m_tuned_and_stream_capture_guards(self):
        runner, layer, _, _, _, _ = setup_runner()
        x = torch.empty((65,1024),dtype=torch.bfloat16)
        self.assertIsNone(runner.apply(layer,x))
        runner.qualified.update((5120,1024,m) for m in ROWS)
        self.assertIsNone(runner.apply(layer,x,'tuned'))
        runner.tuned.add((5120,1024,65))
        self.assertIsNotNone(runner.apply(layer,x,'tuned'))
        runner.torch.cuda.stream=8
        self.assertFalse(runner.eligible(layer,x,None))
        runner.torch.cuda.stream=7;runner.torch.cuda.capture=True
        self.assertFalse(runner.eligible(layer,x,None))
        runner.torch.cuda.capture=False
        self.assertFalse(runner.eligible(layer,x,torch.empty(1)))
        layer._abc_w8=(torch.empty(4),*layer._abc_w8[1:])
        with self.assertRaises(RuntimeError): runner.eligible(layer,x,None)

    def test_exact_marlin_keywords(self):
        runner, layer, _, _, _, calls = setup_runner()
        runner.reference(layer,torch.empty((65,1024),dtype=torch.bfloat16))
        self.assertEqual(set(calls[0]),{'input','weight','weight_scale','workspace','size_n','size_k','bias'})
        tree=ast.parse((BASE/'vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_mxfp8_marlin_linear')
        self.assertEqual([a.arg for a in fn.args.args],['input','weight','weight_scale','workspace','size_n','size_k','bias','use_fp32_reduce'])
        self.assertEqual(ROWS,(65,128,256,512,1024,2032,2048))

    def test_existing_gamma_formula_and_bound(self):
        old=ast.parse((BASE/'hybrid/abc_numeric.py').read_text())
        new=ast.parse((Path(__file__).parent/'qualification.py').read_text())
        def gamma(t):
            return next(n.value for n in ast.walk(t) if isinstance(n,ast.Assign)
                        and any(isinstance(s,ast.Name) and s.id=='gamma' for s in n.targets))
        self.assertEqual(ast.dump(gamma(old)),ast.dump(gamma(new)))
        ref=torch.tensor([[.125,-.5]],dtype=torch.float64)
        absolute=torch.tensor([[1.,1.]],dtype=torch.float64)
        self.assertEqual(within_bound(torch,ref,ref,absolute,1024),0)
        with self.assertRaises(ValueError): within_bound(torch,ref+1,ref,absolute,1024)

    def test_runtime_contract_and_actual_pack_assignment(self):
        def actual(input,weight,weight_scale,workspace,size_n,size_k,bias=None,use_fp32_reduce=True): pass
        self.assertTrue(check_runtime(actual,SimpleNamespace(DENSE=[]))['passed'])
        with self.assertRaises(ValueError): check_runtime(lambda x:x,SimpleNamespace(DENSE=[]))
        tree=ast.parse((BASE/'hybrid/abc_kernels.py').read_text())
        assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Attribute) and t.attr=='_abc_w8' for t in n.targets)]
        self.assertEqual(len(assignments),1)
        self.assertEqual([n.id for n in assignments[0].value.elts],['W','S','cnt'])

    def test_preserve_false_reduction_flag(self):
        runner,layer,backend,seen,_,_=setup_runner(True)
        backend.allow_bf16_reduced_precision_reduction=False
        runner.qualified.update((5120,1024,m) for m in ROWS)
        with self.assertRaises(RuntimeError): runner.apply(layer,torch.empty((65,1024),dtype=torch.bfloat16))
        self.assertFalse(backend.allow_bf16_reduced_precision_reduction)
        self.assertEqual(seen,[False])

    def test_no_torch_import_or_runtime_sync(self):
        for name in ('dense.py','qualification.py'):
            t=ast.parse((Path(__file__).parent/name).read_text())
            imports=[n for n in ast.walk(t) if isinstance(n,(ast.Import,ast.ImportFrom))]
            self.assertFalse(any((isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names))
                                 or (isinstance(n,ast.ImportFrom) and n.module=='torch') for n in imports))
        text=(Path(__file__).parent/'dense.py').read_text()
        self.assertNotIn('synchronize(',text)
        self.assertNotIn('empty_cache(',text)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__': unittest.main()
