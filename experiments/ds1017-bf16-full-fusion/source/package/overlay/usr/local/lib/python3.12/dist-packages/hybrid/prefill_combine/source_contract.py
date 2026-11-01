"""Pure AST API/source identity check, safe before CUDA/model startup."""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap

SOURCE_SHA256 = 'a1cfd543dae0936eb991a7ce8094acc369854934b9482a94db822abd410413a0'
FUNCTION_AST_SHA256 = '42ca3e297b282ce0b350ebcb7967da6083dcd684195b6587790e474266c13bdd'
ARGS = ['topk_indices', 'query_start_loc', 'seq_lens', 'gather_lens',
        'window_size', 'compress_ratio', 'topk', 'M', 'N']


def check_reference_source(source):
    tree = ast.parse(textwrap.dedent(source))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == 'combine_topk_swa_indices')
    if [arg.arg for arg in fn.args.args] != ARGS:
        raise ValueError('Retained combiner API differs from pinned nine arguments')
    digest = hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest()
    if digest != FUNCTION_AST_SHA256:
        raise ValueError('Retained combiner arithmetic differs from pinned Torch function')
    return {'passed': True, 'function_ast_sha256': digest, 'args': ARGS}


def check_reference(reference):
    return check_reference_source(inspect.getsource(reference))


def check_original_file(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Original rocm.py bytes differ from pinned pre-integration file')
    result = check_reference_source(path.read_text())
    tree = ast.parse(path.read_text())
    alignment = [n.value.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == '_SPARSE_PREFILL_TOPK_ALIGNMENT'
                         for t in n.targets) and isinstance(n.value, ast.Constant)]
    if alignment != [128]:
        raise ValueError('Original combiner alignment is not 128')
    return dict(result, source_sha256=SOURCE_SHA256, alignment=128)
