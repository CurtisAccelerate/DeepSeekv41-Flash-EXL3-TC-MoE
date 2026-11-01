"""CPU/source tests only; the fused GPU kernel remains uncompiled/unrun."""
import ast
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

import torch
from .semantics import floor_prefix, linear_prefix, fused
from .qualification import cases, fixture
from .source_contract import check_ops_file, check_model_config, check_function_source

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT/'prefill-reference/live-api/rocm_aiter_mla_sparse.py'
CONFIG = ROOT/'prefill-reference/live-api/model-config.json'
COMBINE = ROOT/'prefill-reference/original-rocm.py'



class CPUPack:
    """CPU translation of the pinned pack kernel, not an executed CUDA kernel."""
    def __getitem__(self, grid):
        def run(indices, lengths, indptr, out, stride, limit, width, BLOCK_SIZE):
            for row in range(indices.shape[0]):
                count = int(lengths[row]); start = int(indptr[row])
                values = indices[row, :count].to(torch.int32)
                if limit >= 0:
                    values = torch.where((values >= 0) & (values < limit), values, -1)
                out[start:start+count] = values
        return run


def extracted(path, name, ns):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return ns[name]


def actual_pipeline():
    combine = extracted(COMBINE, 'combine_topk_swa_indices', {'torch':torch, '_SPARSE_PREFILL_TOPK_ALIGNMENT':128})
    pack = extracted(OPS, 'build_ragged_indices_from_dense', {'torch':torch,
        'triton':SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b), '_pack_dense_prefix_to_ragged_kernel':CPUPack()})
    return combine, pack


class RaggedCPU(unittest.TestCase):
    def test_actual_source_and_model_domain(self):
        self.assertEqual(len(check_ops_file(OPS)),4)
        self.assertEqual(check_model_config(CONFIG)['ratio_counts'],{0:2,1:20,2:18})
        check_function_source(COMBINE.read_text(),'combine_topk_swa_indices')

    def test_closed_forms_against_explicit_sums(self):
        for ratio in (1,2,4,16,128):
            for cap in (0,1,7,512):
                for value in (0,1,ratio-1,ratio,ratio+1,127,128,129,1024,262144):
                    self.assertEqual(floor_prefix(value,ratio,cap),sum(min(u//ratio,cap) for u in range(1,value+1)))
        self.assertEqual(floor_prefix(262144,0,0),0)
        for cap in (0,1,7,128,129):
            for value in (0,1,7,127,128,129,1024,262144):
                self.assertEqual(linear_prefix(value,cap),sum(min(u,cap) for u in range(1,value+1)))

    def test_actual_torch_pipeline_vs_fused_prefix_and_postwrap_masks(self):
        combine,pack=actual_pipeline()
        for spec in cases():
            with self.subTest(spec=spec):
                args=fixture(torch,spec)
                cpu=list(args)
                # Actual CUDA ratio0 reference MUST be used in future GPU qual.
                # CPU integer div0 throws; at logical0 denominator-independent
                # count is provably0, so ratio1 exercises all remaining logic.
                if spec['ratio']==0:cpu[5]=1
                indices,lens=combine(*cpu)
                codes=args[0].reshape(spec['rows'],-1).tolist()
                pure=(codes,args[1].tolist(),args[2].tolist(),args[3].tolist(),*args[4:])
                limits=(min(2**31-1,args[-2]*spec['requests']),)
                if spec['rows']<=129:limits+=(-1,0,7)
                for limit in limits:
                    flat,ptr=pack(indices,lens,num_rows=limit)
                    expected,actualptr=fused(*pure,num_rows=limit)
                    self.assertEqual(ptr.tolist(),actualptr)
                    self.assertEqual(flat[:int(ptr[-1])].tolist(),expected)
                    self.assertEqual(len(expected),int(lens.sum()))

    def test_random_ragged_boundaries_and_invalid_nnz(self):
        rng=random.Random(1017)
        combine,pack=actual_pipeline()
        for _ in range(80):
            rows,requests=rng.randint(1,65),rng.choice((1,2,8))
            lengths=[0]*requests
            for row in range(rows):lengths[rng.randrange(requests)]+=1
            qsl=[97]
            for length in lengths:qsl.append(qsl[-1]+length)
            seq=[q+rng.randrange(256) for q in lengths]
            gather=[rng.randrange(s+1) for s in seq]
            ratio=rng.choice((0,1,2,4,16,128));topk=0 if ratio==0 else rng.choice((0,7,32))
            window=rng.choice((1,7,128,129));N=0 if ratio==0 else 32
            codes=[[rng.choice((-1,0,3,31,32,2**32+3,-2**32+3)) for col in range(6)] for row in range(rows)]
            args=(torch.tensor(codes,dtype=torch.int64),torch.tensor(qsl,dtype=torch.int32),
                  torch.tensor(seq,dtype=torch.int64),torch.tensor(gather,dtype=torch.int64),window,ratio or 1,topk,2**30,N)
            dense,lens=combine(*args)
            for limit in (-1,0,7,2**31-1):
                flat,ptr=pack(dense,lens,num_rows=limit)
                got,ip=fused(codes,qsl,seq,gather,window,ratio,topk,2**30,N,limit)
                self.assertEqual(ptr.tolist(),ip);self.assertEqual(flat[:int(ptr[-1])].tolist(),got)

    def test_jit_specialization_and_swa_division_guards(self):
        tree=ast.parse((Path(__file__).parent/'ragged.py').read_text())
        kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_ragged')
        d={k.arg:ast.literal_eval(k.value) for k in kernel.decorator_list[0].keywords}
        self.assertEqual(set(d['do_not_specialize']),{'M','N','LIMIT','ROWS'})
        self.assertEqual(set(d['do_not_specialize_on_alignment']),{'TOP','QSL','SEQ','GATHER','FLAT','INDPTR'})
        self.assertTrue(all(a.annotation is None for a in kernel.args.args if a.arg in d['do_not_specialize']))
        for name in ('_ragged','_primitive'):
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            branches=[n for n in ast.walk(fn) if isinstance(n,ast.If) and ast.unparse(n.test)=='LOGICAL == 0']
            self.assertTrue(branches)
            for branch in branches:
                self.assertFalse(any(isinstance(n,ast.BinOp) and isinstance(n.op,(ast.FloorDiv,ast.Mod)) for stmt in branch.body for n in ast.walk(stmt)))
        prod={(s['ratio'],s['requests']) for s in cases() if s.get('production_signature')}
        self.assertEqual(prod,{(r,n) for r in (0,1,2,4,16,128) for n in range(1,9)})

    def test_no_cuda_initialized(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
