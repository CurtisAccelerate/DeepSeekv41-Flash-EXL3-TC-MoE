"""CPU-only actual bind-mount/source/ABI check. No GPU request and no model launch."""
import argparse,json,subprocess,hashlib
from pathlib import Path

SCRIPT='''import json,runpy,hashlib,pathlib
import os
engram=pathlib.Path(os.environ['W9_ENGRAM_BACKING'])/'manifest.json'
assert engram.is_file(),'Required Engram resident generation is absent'
assert hashlib.sha256(engram.read_bytes()).hexdigest()==os.environ['W9_ENGRAM_MANIFEST_SHA256'],'Engram manifest differs'
import ctypes,mmap
buf=mmap.mmap(-1,4096);buf[0]=1
address=ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(buf)))
status=ctypes.c_int(-1);libc=ctypes.CDLL(None,use_errno=True)
# x86_64 move_pages query only: no migration, no model or shared-arena accesses.
result=libc.syscall(ctypes.c_long(279),ctypes.c_int(0),ctypes.c_ulong(1),ctypes.byref(address),ctypes.c_void_p(),ctypes.byref(status),ctypes.c_int(0))
assert result==0 and status.value>=0,('NUMA page query unavailable',ctypes.get_errno(),status.value)
print('STAR_ABC_NUMA_QUERY '+json.dumps(dict(query_available=True,node=status.value,migration=False)))
ns=runpy.run_path('/w26-preflight/preflight_launch.py')
manifest=json.loads(pathlib.Path('/w26-preflight/manifest.json').read_text())
preflight=ns['attest'](manifest)
assert preflight['passed'] and not preflight['cuda_imported']
pins=json.loads(pathlib.Path('/usr/local/lib/python3.12/dist-packages/hybrid/attention-imports.json').read_text())
for row in pins['files']:
 assert hashlib.sha256(pathlib.Path(row['target']).read_bytes()).hexdigest()==row['sha256'],row['target']
from hybrid.abc_startup import attest
from hybrid.prefill_api_contract import audit
api=audit(pathlib.Path('/opt/star-prefill/package'),hook_path=pathlib.Path('/opt/dsv41-tc-moe/exl3_tc_hook.py'),plugin_path=pathlib.Path('/usr/local/lib/python3.12/dist-packages/vllm_exl3/exl3.py'))
assert api['passed'],'Actual effective hook/plugin API audit failed'
from hybrid.prefill_v5_contract import audit as audit_v5
api['v5']=audit_v5(pathlib.Path('/opt/star-prefill/package'),effective_site=pathlib.Path('/usr/local/lib/python3.12/dist-packages'))
from hybrid.prefill_v6_contract import audit as audit_v6
api['v6']=audit_v6(pathlib.Path('/opt/star-prefill/package'),effective_site=pathlib.Path('/usr/local/lib/python3.12/dist-packages'),config_path=pathlib.Path('/model-view/config.json'))
print('STAR_PREFILL_EFFECTIVE_API '+json.dumps(api))
identity=attest()
runpy.run_path('/opt/star-abc/tools/cpu_check.py')
print('STAR_ABC_STAGED_CPU '+json.dumps(dict(preflight_passed=True,attention_import_pins_passed=True,abc_source_and_binary_attestation_passed=True,effective_prefill_api_passed=api['passed'],effective_hook_sha256=api['hook_sha256'],effective_plugin_sha256=api['plugin_sha256'],identity=identity,origin_modules=len(preflight['modules']),no_gpu_request=True,model_loaded=False,gpu_qualified=False,model_quality_verified=False)))
'''

def check(root):
    spec=root/'launch.json';s=json.loads(spec.read_text())
    if s.get('schema')!='star.abc.launch.v1' or s['name'] not in ('star-abc-test','star-abc-test-fp8','star-abc-test-bf16'):raise ValueError('Dedicated frozen launch required')
    cmd=['docker','-H','unix:///run/dsv41-private/docker.sock','run','--rm','--network','none','--env','CUDA_VISIBLE_DEVICES=','--entrypoint','python3']
    for cap in s['host_config'].get('CapAdd') or []:cmd+=['--cap-add',cap]
    for setting in s['host_config'].get('SecurityOpt') or []:cmd+=['--security-opt',setting]
    for k,v in s['env'].items():
        if k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES'):continue
        cmd+=['--env',k+'='+v]
    cmd+=['--env','NVIDIA_VISIBLE_DEVICES=void','--env','TORCHINDUCTOR_CACHE_DIR=/tmp/abc-cpu-inductor']
    for m in s['mounts']:
        cmd+=['--mount','type=bind,src='+m['Source']+',dst='+m['Target']+',readonly']
    cmd+=[s['image'],'-c',SCRIPT]
    r=subprocess.run(cmd,capture_output=True,text=True)
    (root/'staged-cpu-check.log').write_text(r.stdout+'\n'+r.stderr)
    if r.returncode:raise RuntimeError('Staged CPU check failed; inspect staged-cpu-check.log\n'+r.stderr[-5000:])
    marker='STAR_ABC_STAGED_CPU '
    receipt=json.loads(next(line[len(marker):] for line in r.stdout.splitlines() if line.startswith(marker)))
    receipt.update(launch_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),image=s['image'])
    (root/'staged-cpu-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);a=ap.parse_args();check(a.root)
