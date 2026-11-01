"""CPU-only exact incumbent semantics and source contract checks."""
import ast
import hashlib
import inspect
from pathlib import Path
import random
import unittest

import torch
from qualification import cases, fixture
from semantics import reference, fused
from source_contract import check_original_file, check_reference_source

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'prefill-reference/original-rocm.py'
ARGS = ['topk_indices', 'query_start_loc', 'seq_lens', 'gather_lens',
        'window_size', 'compress_ratio', 'topk', 'M', 'N']


def incumbent():
    tree = ast.parse(SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'combine_topk_swa_indices')
    ns = {'torch': torch, '_SPARSE_PREFILL_TOPK_ALIGNMENT': 128}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns[fn.name]


class CPUContract(unittest.TestCase):
    def test_pinned_source_identity(self):
        self.assertTrue(check_original_file(SOURCE)['passed'])
        source = SOURCE.read_text()
        with self.assertRaises(ValueError):
            check_reference_source(source.replace('topk_values >= 0', 'topk_values > 0'))

    def test_actual_incumbent_signature(self):
        self.assertEqual(list(inspect.signature(incumbent()).parameters), ARGS)
        tree = ast.parse((Path(__file__).parent/'combiner.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'combine_topk_swa_indices')
        self.assertEqual([a.arg for a in fn.args.args], ARGS)

    def test_dynamic_layout_and_actual_warm_signatures(self):
        tree = ast.parse((Path(__file__).parent/'combiner.py').read_text())
        kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_combine')
        arguments = {a.arg:a for a in kernel.args.args}
        self.assertIsNone(arguments['M'].annotation)
        self.assertIsNone(arguments['N'].annotation)
        self.assertEqual(ast.literal_eval(kernel.decorator_list[0].keywords[0].value), ['M', 'N'])
        warmed = set()
        for spec in cases():
            if spec.get('production_signature'):
                args = fixture(torch, spec)
                warmed.add((spec['ratio'], spec['requests'], args[-2], args[-1]))
                self.assertTrue(all(t.dtype == torch.int32 for t in args[:4]))
        self.assertEqual(warmed, {(r,n,(262144+r-1)//r+128+2048,(262144+r-1)//r)
                                  for r in (1,4,16,128) for n in range(1,9)})

    def test_actual_torch_against_pure_oracles(self):
        original = incumbent()
        for spec in cases():
            with self.subTest(spec=spec):
                args = fixture(torch, spec)
                actual = original(*args)
                codes = args[0].reshape(spec['rows'], -1).tolist()
                pure_args = (codes, args[1].tolist(), args[2].tolist(), args[3].tolist(), *args[4:])
                expected = reference(*pure_args)
                self.assertEqual(actual[0].tolist(), expected[0])
                self.assertEqual(actual[1].tolist(), expected[1])
                self.assertEqual(fused(*pure_args), expected)

    def test_random_row_algebra(self):
        gen = random.Random(1017)
        for _ in range(80):
            rows, requests = gen.randint(1, 65), gen.choice((1, 2, 8))
            lengths = [0] * requests
            for row in range(rows): lengths[gen.randrange(requests)] += 1
            qsl = [gen.choice((0, 97, 1001))]
            for length in lengths: qsl.append(qsl[-1] + length)
            seq = [q + gen.randrange(256) for q in lengths]
            gather = [gen.randrange(s + 1) for s in seq]
            window, ratio, topk = gen.choice((1, 7, 128, 129)), gen.choice((1, 4, 16, 128)), gen.choice((0, 7, 32))
            codes = [[gen.choice((-1, 0, 3, 31, 32, 2**32 + 3, -2**32 + 3)) for _ in range(6)] for row in range(rows)]
            args = codes, qsl, seq, gather, window, ratio, topk, 2**30, 32
            self.assertEqual(reference(*args), fused(*args))

    def test_no_cache_or_mutation_or_cuda_init(self):
        text = (Path(__file__).parent/'combiner.py').read_text()
        tree = ast.parse(text)
        self.assertNotIn('setattr(', text)
        self.assertNotIn('synchronize(', text)
        self.assertNotIn('.item(', text)
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and n.func.attr in ('init', 'load_library') for n in ast.walk(tree)))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
