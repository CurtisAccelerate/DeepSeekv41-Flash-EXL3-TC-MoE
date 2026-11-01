"""Exact-toolchain CPU checks during image build, with CUDA devices hidden."""
import importlib,json,os,runpy,unittest
from pathlib import Path
os.environ['STAR_PREFILL']='0'
runpy.run_path('/opt/star-abc/tools/cpu_check.py',run_name='__main__')
import torch
for name in ('prefill_contract','prefill_runtime','prefill_qualification','prefill_startup','prefill_control','prefill_api_contract','prefill_v5_contract','prefill_interop','prefill_dense','prefill_combine.combiner','prefill_v6_contract','prefill_ragged.ragged'):
    importlib.import_module('hybrid.'+name)
from hybrid.prefill_startup import prepare
from types import SimpleNamespace
assert prepare(SimpleNamespace())==dict(enabled=False,ready=False,default_profile='REFERENCE')
torch.ops.load_library('/opt/star-abc/lib/_dsv41_mm_try.abi3.so')
from prefill_workspace import size
queries=[]
for width in (512,640):
    for rows in (65,128,256,512,1024,2048):
        actual=int(torch.ops._dsv41_mm_try.moe_workspace_size_prefill_try(rows,5120,width,6,384))
        assert actual==size(rows,width),(rows,width,actual,size(rows,width))
        queries.append(dict(rows=rows,intermediate=width,elements=actual))
assert not torch.cuda.is_initialized(),'CPU ABI query initialized CUDA'
package=Path('/opt/star-prefill/package')
from hybrid.prefill_api_contract import audit,NATIVE_ARGS
api=audit(package,plugin_path=Path('/usr/local/lib/python3.12/dist-packages/vllm_exl3/exl3.py'))
from hybrid.prefill_v5_contract import audit as audit_v5
api['v5']=audit_v5(package,effective_site=Path('/usr/local/lib/python3.12/dist-packages'))
from hybrid.prefill_v6_contract import audit as audit_v6
api['v6']=audit_v6(package,effective_site=Path('/usr/local/lib/python3.12/dist-packages'))
from vllm.config.profiler import ProfilerConfig
from prefill_prepare_launch import PROFILER
assert ProfilerConfig(**PROFILER).max_iterations==1 and not ProfilerConfig(**PROFILER).ignore_frontend
from vllm.engine.arg_utils import get_kwargs
from vllm.config import ObservabilityConfig
from vllm.utils.argparse_utils import FlexibleArgumentParser
import ast,inspect
import vllm.engine.arg_utils as actual_arg_utils
# Full EngineArgs parser constructs an automatic DeviceConfig and therefore
# cannot run with CUDA deliberately hidden. Exercise the actual observability
# argument field/action and prove its real registration from installed source.
source=ast.parse(inspect.getsource(actual_arg_utils))
assert any(isinstance(n,ast.Call) and n.args and isinstance(n.args[0],ast.Constant)
           and n.args[0].value=='--jit-monitor-verbose'
           and any(k.arg is None and ast.unparse(k.value)=="observability_kwargs['jit_monitor_verbose']" for k in n.keywords)
           for n in ast.walk(source))
parser=FlexibleArgumentParser()
parser.add_argument('--jit-monitor-verbose',**get_kwargs(ObservabilityConfig)['jit_monitor_verbose'])
parsed=parser.parse_args(['--jit-monitor-verbose'])
assert parsed.jit_monitor_verbose is True
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_mxfp8_marlin_linear
from hybrid import abc_numeric
from hybrid.prefill_dense.source_contract import check_runtime
api['dense_runtime']=check_runtime(apply_mxfp8_marlin_linear,abc_numeric)
schema=torch.ops._dsv41_mm_try.moe_out_prefill_try._schemas['']
assert [a.name for a in schema.arguments]==NATIVE_ARGS,'Actual loaded native schema differs'
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext,get_forward_context
assert CUDAGraphMode.NONE.name=='NONE' and callable(get_forward_context)
assert 'cudagraph_runtime_mode' in getattr(ForwardContext,'__annotations__',{}) or 'cudagraph_runtime_mode' in getattr(ForwardContext,'__dataclass_fields__',{})
assert callable(torch.nn.Module.named_modules)
Path('/opt/star-prefill/actual-api-check.json').write_text(json.dumps(dict(source_audit=api,loaded_schema=str(schema),enum_none=CUDAGraphMode.NONE.name,cuda_initialized=torch.cuda.is_initialized()),indent=2))
counts={}
for name,directory,expected in [('baseline',package/'tests',53),('prefill',package/'prefill-tests',13),('actual-api',package/'prefill-api-tests',6),
                                ('combine',package/'prefill-combine-tests',6),('dense',package/'prefill-dense-tests',8),('ragged',package/'prefill-ragged-tests',10)]:
    suite=unittest.TestLoader().discover(str(directory))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and not result.skipped and result.testsRun==expected,(name,result.testsRun,result.skipped)
    counts[name]=result.testsRun
assert not torch.cuda.is_initialized(),'CPU test suite initialized CUDA'
receipt=dict(passed=True,tests=counts,native_host_queries=queries,torch=torch.__version__,torch_cuda=torch.version.cuda,
             cuda_initialized=False,gpu_qualified=False,engine_qualified=False,model_quality_verified=False)
Path('/opt/star-prefill/cpu-check.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
