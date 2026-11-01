"""Pure source/API guards against the fetched actual runtime; no CUDA imports."""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
import json
from collections import Counter

OPS_SHA256 = 'a36c0e13cf63a77c804a5d5d60ec2f6d3e62bec51de3afac4f28619f91f5f8fc'
CONFIG_SHA256 = '69c61d4fd3c6020357d46620d854d7060d8c197bc656a4c01a725d044dd8736c'
AST_SHA = {
    'combine_topk_swa_indices': '42ca3e297b282ce0b350ebcb7967da6083dcd684195b6587790e474266c13bdd',
    '_pack_dense_prefix_to_ragged_kernel': '90761841104be2bd6795b965c0f9abbee33a37321c14f48d9c4164555f8400ec',
    'build_ragged_indices_from_dense': '76717b16d20ed0f426f085b638da5434ba2908fdfb890e2e4520c7ab1fa9e232',
    '_sparse_attn_prefill_ragged_kernel': 'eb4b927b86df99ef1e9ef8872c44c49e8cba8fba21cc65e494f7477e7de1887e',
    'rocm_sparse_attn_prefill': '37020112b51109dfa0f5ac81ddecb74102a25d034992d381bc383af128db367d',
}


def check_function_source(source, name):
    node = next(n for n in ast.parse(textwrap.dedent(source)).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    if digest != AST_SHA[name]:
        raise ValueError('Actual retained function changed: ' + name)
    return digest


def check_ops_file(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != OPS_SHA256:
        raise ValueError('Actual sparse ops bytes differ from pinned source')
    return {name: check_function_source(path.read_text(), name) for name in AST_SHA if name != 'combine_topk_swa_indices'}


def check_references(combine, pack):
    return {name: check_function_source(inspect.getsource(fn), name) for fn, name in (
        (combine, 'combine_topk_swa_indices'), (pack, 'build_ragged_indices_from_dense'))}


def check_model_config(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != CONFIG_SHA256:
        raise ValueError('Actual model config bytes differ')
    c = json.loads(path.read_text())['text_config']
    counts = Counter(c['compress_ratios'][:c['num_hidden_layers']])
    if (c['num_hidden_layers'] != 40 or counts != {0:2, 1:20, 2:18}
            or c['index_topk'] != 512 or c['sliding_window'] != 128):
        raise ValueError('Actual model prefill domain changed')
    return dict(target_layers=40, ratio_counts=dict(counts), topk=512, window=128,
                actual_ratios=[0,1,2], draft_ratios=c['compress_ratios'][40:])
