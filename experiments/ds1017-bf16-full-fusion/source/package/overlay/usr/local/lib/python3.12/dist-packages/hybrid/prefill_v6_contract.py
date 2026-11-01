"""Exact installed source, metadata domain and dispatch audit; no CUDA imports."""
import ast,hashlib,importlib.util
from pathlib import Path
SITE='usr/local/lib/python3.12/dist-packages'
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def need(ok,message):
 if not ok:raise ValueError(message)
def audit(package,effective_site=None,config_path=None):
 p=Path(package);effective=p/'overlay'/SITE if effective_site is None else Path(effective_site)
 h=effective/'hybrid';r=h/'prefill_ragged';contract=load(r/'source_contract.py','ragged_source_contract')
 ops=(p/'prefill-reference/live-api/rocm_aiter_mla_sparse.py') if effective_site is None else effective/'vllm/v1/attention/ops/rocm_aiter_mla_sparse.py'
 config=Path(config_path) if config_path else p/'prefill-reference/live-api/model-config.json'
 operations=contract.check_ops_file(ops);model=contract.check_model_config(config)
 tree=ast.parse((effective/'vllm/models/deepseek_v4_1/amd/rocm.py').read_text())
 fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_forward_prefill');text=ast.unparse(fn)
 need('packed = ragged_candidate(*combine_args, num_rows=kv.numel() // q.shape[-1])' in text,'Actual prefill ragged call/physical KV bound differs')
 need('combined_indices, combined_lens = combine_topk_swa_indices(*combine_args)' in text,'Actual retained fallback missing')
 need('ragged_indices=packed[0], ragged_indptr=packed[1]' in text and '**ragged_kwargs' in text,'Actual same-attention ragged dispatch missing')
 contract.check_function_source((effective/'vllm/models/deepseek_v4_1/amd/rocm.py').read_text(),'combine_topk_swa_indices')
 startup=ast.unparse(ast.parse((h/'prefill_startup.py').read_text()))
 need("name not in RAGGED_PROFILES or ragged_ok" in startup,'Qualified all-rank ragged profile gate missing')
 need("rec['cases'] != 85" in startup and "rec['attention']['cases'] != 9" in startup,'Integer/actual-attention startup completeness gate missing')
 need("rocm.rocm_sparse_attn_prefill is not sparse_ops.rocm_sparse_attn_prefill" in startup,'Actual attention alias not audited')
 files=[]
 for name in ('ragged.py','qualification.py','source_contract.py','attention_qualification.py'):
  actual=r/name;fixture=p/'prefill-ragged-tests/prefill_v6_ragged'/name
  need(actual.read_bytes()==fixture.read_bytes(),'Ragged fixture/installed bytes differ: '+name)
  files.append(dict(file=name,sha256=hashlib.sha256(actual.read_bytes()).hexdigest()))
 return dict(passed=True,same_attention_source=operations,model=model,model_config_sha256=contract.CONFIG_SHA256,
             sparse_ops_sha256=contract.OPS_SHA256,component_files=files,engine_layer_coverage_verified=False,gpu_qualified=False)
