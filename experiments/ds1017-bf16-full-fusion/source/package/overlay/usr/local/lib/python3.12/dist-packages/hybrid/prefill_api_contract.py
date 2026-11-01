"""Pure-AST API audit of the actual mounted plugin/hook; no torch import or GPU."""
import ast
import hashlib
import json
from pathlib import Path
import re

HOOK_SHA='0fd676958c111102c21d1d6dc4914175728050c42d00068ba5a6a9c460abdcf5'
PLUGIN_SHA='be41583daa448d6148c1711e52cc24992cfaf5420369f22ba45e43378ff57433'
SITE='usr/local/lib/python3.12/dist-packages'
HOOK_ARGS=['x2d','ids','weights','layer','expert_map','limit']
NATIVE_ARGS=['x','topk_ids','topk_weights','packed_ptrs','packed_meta','expert_map','intermediate_size','bits_x2','out','workspace','swiglu_limit']
TC_KEYS={'ops','ptrs','meta','emap','workspace','intermediate','bits_x2','v2','prefill','prefill_rows'}
def need(ok,message):
    if not ok:raise ValueError(message)
def read(path):return Path(path).read_text(encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def function(tree,name):return next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
def text(node):return ast.unparse(node)
def contains(tree,expression):
    expected=ast.dump(ast.parse(expression,mode='eval').body,include_attributes=False)
    return any(ast.dump(n,include_attributes=False)==expected for n in ast.walk(tree))


def check_plugin(source):
    tree=ast.parse(source)
    need(any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_tc_hook' for t in n.targets)
             and isinstance(n.value,ast.Call) and text(n.value.func)=='_tc_util.module_from_spec' for n in tree.body),
         'Plugin must define the actual module global _tc_hook')
    need(contains(tree,'"/opt/dsv41-tc-moe/exl3_tc_hook.py"'),'Plugin hook path changed')
    f=function(tree,'apply_exl3_fused_moe')
    calls=[n for n in ast.walk(f) if isinstance(n,ast.Call) and text(n.func)=='_tc_hook.try_apply']
    need(len(calls)==1 and [text(a) for a in calls[0].args]==HOOK_ARGS,'Plugin actual hook callsite/signature changed')
    return dict(hook_global='_tc_hook',hook_path='/opt/dsv41-tc-moe/exl3_tc_hook.py',call_args=HOOK_ARGS)


def check_hook(source):
    tree=ast.parse(source);f=function(tree,'try_apply')
    need([a.arg for a in f.args.args]==HOOK_ARGS and f.args.vararg is None and f.args.kwarg is None,'Actual hook signature differs')
    build=function(tree,'try_build')
    assignments={text(t):n.value for n in ast.walk(build) if isinstance(n,ast.Assign) for t in n.targets}
    need('tc' in assignments and isinstance(assignments['tc'],ast.Dict),'Actual TC state dictionary missing')
    keys={k.value for k in assignments['tc'].keys if isinstance(k,ast.Constant)}
    need(keys==TC_KEYS,'Actual TC keys differ')
    for name,shape in [('ptrs','(_EXPERTS, 3, 3)'),('meta','(_EXPERTS, 3, 4)')]:
        call=assignments[name]
        need(isinstance(call,ast.Call) and text(call.func)=='torch.empty' and text(call.args[0])==shape,
             'Actual TC '+name+' shape differs')
        need(any(k.arg=='dtype' and text(k.value)=='torch.int64' for k in call.keywords),'Actual TC '+name+' dtype differs')
    work=assignments['workspace']
    need(isinstance(work,ast.Call) and text(work.func)=='torch.empty' and any(k.arg=='dtype' and text(k.value)=='torch.float32' for k in work.keywords),'Actual workspace must be FP32')
    need(contains(build,'torch.arange(_EXPERTS, dtype=torch.int32, device=device)'),'Actual fallback expert map format differs')
    need(contains(build,'_SHARED_WORKSPACE.get(dev_key)') and '_SHARED_WORKSPACE[dev_key]' in text(build),'Actual shared workspace ownership differs')
    native=[n for n in ast.walk(f) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='fn']
    expected=['x2d.contiguous()','ids','w.contiguous()',"tc['ptrs']","tc['meta']",'emap',"tc['intermediate']","tc['bits_x2']",'float(limit or 0.0)','out',"tc['workspace']"]
    need(len(native)==1 and [text(a) for a in native[0].args]==expected,'Incumbent native ABI differs')
    constants={t.id:n.value.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) for t in n.targets if isinstance(t,ast.Name)}
    need(all(constants.get(k)==v for k,v in {'_HIDDEN':5120,'_TOPK':6,'_EXPERTS':384}.items()),'Actual hook geometry constants differ')
    return dict(signature=HOOK_ARGS,tc_keys=sorted(keys),ptrs=[384,3,3],meta=[384,3,4],pointer_dtype='int64',workspace_dtype='float32',emap='int32 fallback; retained CUDA int32/int64 accepted by native',incumbent_native_args=expected)


def check_startup(source):
    tree=ast.parse(source)
    hook=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='hook' for t in n.targets)]
    need(len(hook)==1 and isinstance(hook[0].value,ast.Attribute) and text(hook[0].value)=='exl3._tc_hook',
         'Startup hook lookup must use actual module global exl3._tc_hook, not a donor helper call')
    return dict(hook_lookup='exl3._tc_hook',donor_helper_not_called=True)


def check_runner(source,model_source,cumulative_source):
    tree=ast.parse(source);runner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GPUModelRunner')
    init=function(runner,'__init__');get=function(runner,'get_model')
    assigns={text(t):text(n.value) for n in ast.walk(init) if isinstance(n,ast.Assign) for t in n.targets}
    for field,value in [('self.vllm_config','vllm_config'),('self.device','device'),('self.compilation_config','vllm_config.compilation_config'),('self._w26_serving_seen','False')]:
        need(assigns.get(field)==value,'Actual runner initialization differs: '+field)
    need(contains(get,'self.model') and contains(get,'self.model.unwrap()'),'Actual runner get_model unwrapping differs')
    need(contains(tree,'self.compilation_config.max_cudagraph_capture_size'),'Actual capture-size config API missing')
    need('config.num_hidden_layers' in model_source and 'self.layers = make_layers(' in model_source,'Actual target layer construction differs')
    need('worker._w26_cumulative_roots=roots' in cumulative_source,'Actual worker retained-root API differs')
    return dict(get_model='returns unwrapped nn.Module',runner_fields=['vllm_config','device','compilation_config','_w26_serving_seen'],
                layers='constructed from config.num_hidden_layers; actual count40 remains a startup check, not proved by generic class source')


def check_candidate(runtime_source,native_source):
    tree=ast.parse(runtime_source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and text(n.func)=='self.ops.moe_out_prefill_try']
    need(len(calls)==1,'Candidate operator call missing/ambiguous')
    args=calls[0].args
    need(len(args)==11 and text(args[8])=='out' and text(args[9])=="tc['workspace']" and text(args[10])=='float(limit or 0.0)',
         'Candidate out/workspace/limit ABI differs')
    block=re.search(r'm\.def\(("moe_out_prefill_try.*?);',native_source,re.S)
    need(block is not None,'Native registration missing')
    schema=''.join(json.loads(s) for s in re.findall(r'"(?:\\.|[^"\\])*"',block.group(1)))
    fields=schema[schema.index('(')+1:schema.rindex(') ->')].split(',')
    names=[x.strip().split()[-1].split('=')[0] for x in fields]
    need(names==NATIVE_ARGS,'Compiled source native schema order differs')
    need('TORCH_LIBRARY(_dsv41_mm_try, m)' in native_source,'Native namespace differs')
    need('self.mode() != "NONE"' in runtime_source and 'cuda_stream' in runtime_source,'Eager/stream guard missing')
    return dict(namespace='_dsv41_mm_try',schema=schema,args=NATIVE_ARGS,runtime='eager NONE only, same prepared stream; serving-stream equality still needs live hit smoke')


def audit(package,hook_path=None,plugin_path=None):
    package=Path(package);site=package/'overlay'/SITE
    hook_path=Path(hook_path) if hook_path else package/'prefill-reference/live-api/exl3_tc_hook.py'
    plugin_path=Path(plugin_path) if plugin_path else site/'vllm_exl3/exl3.py'
    need(sha(hook_path)==HOOK_SHA,'Effective hook source SHA differs from actual mounted audit copy')
    need(sha(plugin_path)==PLUGIN_SHA,'Effective plugin source SHA differs from actual mounted audit copy')
    h=site/'hybrid'
    checks=dict(plugin=check_plugin(read(plugin_path)),hook=check_hook(read(hook_path)),
        startup=check_startup(read(h/'prefill_startup.py')),
        runner=check_runner(read(site/'vllm/v1/worker/gpu_model_runner.py'),read(site/'vllm/models/deepseek_v4_1/nvidia/model.py'),read(h/'cumulative_startup.py')),
        candidate=check_candidate(read(h/'prefill_runtime.py'),read(package/'prefill-native/moe_prefill_try.cu')))
    return dict(passed=True,checks=checks,hook_target='/opt/dsv41-tc-moe/exl3_tc_hook.py',hook_source=str(hook_path),hook_sha256=HOOK_SHA,
                plugin_source=str(plugin_path),plugin_sha256=PLUGIN_SHA,torch_imported=False,gpu_qualified=False,
                remaining=['actual layer metadata/count and per-rank workspace','144-case actual-weight parity','actual serving stream/eager mode hits','memory floor and model quality/performance'])
