"""Isolated image ABI/import check. CUDA devices remain hidden; no kernels run."""
import ast,importlib,json
from pathlib import Path
import torch
import runpy
core=Path('/usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so')
for op in (b'fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert',b'fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert'):
    assert op in core.read_bytes(),'Required native writer missing from installed core binary'
for name in ['abc_contract','abc_runtime','abc_kernels','abc_sampler','abc_gate','abc_startup','abc_control','abc_policy','abc_greedy','abc_attention','abc_admission','abc_fidelity','abc_probe','abc_numeric','abc_lane_check','abc_kv_contract','abc_bf16_read','abc_bf16_io','abc_bf16_check']:
    importlib.import_module('hybrid.'+name)
import dsv41_fp8w_ext,dsv41_woa_ext
torch.ops.load_library('/opt/star-abc/lib/_dsv41_mm_v2try.abi3.so')
assert int(torch.ops._dsv41_mm_v2try.moe_workspace_size_v2_try(64,5120,640,6,384))>0
assert not torch.cuda.is_initialized(),'CPU build/import check initialized CUDA'
for p in Path('/opt/star-abc/overlay').rglob('*.py'):ast.parse(p.read_text())
runpy.run_path('/opt/star-abc/tools/arena_cpu_check.py',run_name='__main__')
print(json.dumps(dict(cpu_imports=True,native_abi_load=True,torch=torch.__version__,cuda_initialized=False,gpu_qualified=False,model_quality_verified=False)))

