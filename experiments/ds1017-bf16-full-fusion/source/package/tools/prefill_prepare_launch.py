"""Freeze fresh code/state mounts from the pinned dc87 snapshot; never starts a model."""
from pathlib import Path
import copy,hashlib,json,shutil

DATA_TARGETS={'/engram-1017','/engram-resident','/engram-wkv-bf16.safetensors','/exl3-1017','/fp8-1017','/model-view','/mnt/nvme-new/dsv41-exl3-arena-cache'}
CACHE_TARGETS={'/root/.tilelang','/root/.triton','/root/.cache/torch','/root/.cache/vllm','/root/.nv/ComputeCache'}
SITE='/usr/local/lib/python3.12/dist-packages'
MODULES=('prefill_contract','prefill_runtime','prefill_qualification','prefill_startup','prefill_control','prefill_api_contract','prefill_v5_contract','prefill_interop',
         'prefill_combine','prefill_combine.combiner','prefill_combine.qualification','prefill_combine.source_contract',
         'prefill_dense','prefill_dense.dense','prefill_dense.qualification','prefill_dense.source_contract','prefill_v6_contract','prefill_ragged','prefill_ragged.ragged','prefill_ragged.qualification','prefill_ragged.attention_qualification','prefill_ragged.source_contract')

PROFILER=dict(profiler='torch',torch_profiler_dir='/r1out/profiles',torch_profiler_with_stack=False,
              torch_profiler_record_shapes=False,torch_profiler_with_memory=False,ignore_frontend=False,
              max_iterations=1,capture_torch_profiler=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def validate_quality_choice(path,dtype):
    choice=json.loads(path.read_text())
    if (choice.get('schema')!='star.prefill.cache-choice.v1' or choice.get('accuracy_first') is not True
            or choice.get('selected_dtype')!=dtype or not isinstance(choice.get('decision_note'),str) or not choice['decision_note'].strip()):
        raise ValueError('Explicit parent accuracy-first cache decision is required')
    evidence=choice.get('quality_evidence')
    if not isinstance(evidence,list) or not evidence:raise ValueError('Cache choice must name its measured quality evidence')
    for row in evidence:
        if set(row)!={'path','sha256'} or sha(Path(row['path']))!=row['sha256']:
            raise ValueError('Cache-choice evidence hash mismatch')
    return choice


def copy_source(source,dest):
    if dest.exists():return
    dest.parent.mkdir(parents=True,exist_ok=True)
    if source.is_dir():shutil.copytree(source,dest)
    elif source.is_file():shutil.copy2(source,dest)
    else:raise ValueError('Frozen source missing: '+str(source))


def package_source(package,target):
    if target.startswith('/w9/'):
        p=package/'host-overlay'/target[len('/w9/'):]
    else:p=package/'overlay'/target.lstrip('/')
    return p if p.exists() else None


def resolve(mounts,package,target):
    for m in sorted(mounts,key=lambda x:len(x['Target']),reverse=True):
        t=m['Target']
        if target==t:return Path(m['Source'])
        if target.startswith(t+'/'):return Path(m['Source'])/target[len(t)+1:]
    return None if package is None else package_source(package,target)


def verify_reference_pins(reference,package):
    """Do not silently bless drift while repinning the new controlled source changes."""
    pf=resolve(reference['mounts'],None,'/w26-preflight/manifest.json')
    imports=resolve(reference['mounts'],None,SITE+'/hybrid/attention-imports.json')
    if pf is None or imports is None:raise ValueError('Original preflight/import manifests missing')
    old=json.loads(pf.read_text());old_imports=json.loads(imports.read_text())
    baseline={r['target']:r['sha256'] for r in json.loads((package/'prefill-reference/image-manifest.json').read_text())['files']}
    rows=old['files']+old['data_files']+old_imports['files']
    for row in rows:
        actual=resolve(reference['mounts'],None,row['target'])
        if actual is not None and actual.is_file():
            if sha(actual)!=row['sha256']:raise ValueError('Original mounted source/data pin drifted: '+row['target'])
        elif baseline.get(row['target'])!=row['sha256']:
            raise ValueError('Original image-only pin is not covered by verified dc87 manifest: '+row['target'])
    return dict(passed=True,rows=len(rows))


def frozen_code_pins(mounts,package):
    """Pin all frozen code, including incumbent external native libraries and child overrides."""
    targets=set()
    for m in mounts:
        target=m['Target']
        if target in DATA_TARGETS or target in CACHE_TARGETS or target in ('/w9/state','/r1out'):continue
        p=Path(m['Source'])
        if p.is_file():targets.add(target)
        else:
            for child in p.rglob('*'):
                if child.is_file() and '__pycache__' not in child.parts and child.suffix not in ('.pyc','.pyo'):
                    targets.add(target+'/'+child.relative_to(p).as_posix())
    targets.discard('/w26-preflight/manifest.json')  # avoid a recursive self-hash
    return [dict(target=t,sha256=sha(resolve(mounts,package,t))) for t in sorted(targets)]


def freeze_mounts(reference,reference_root,root,package):
    """Only new-root writes; model/Engram backing remains the identical existing data source."""
    snapshot=root/'snapshot'
    if snapshot.exists():raise ValueError('Fresh snapshot required; existing stage is never overwritten')
    shutil.copytree(reference_root/'snapshot',snapshot)
    mounts=[]
    for i,old in enumerate(reference['mounts']):
        m=copy.deepcopy(old);source=Path(m['Source']);target=m['Target']
        if target=='/w9/state':
            source=root/'state';source.mkdir()
        elif target=='/r1out':
            source=root/'round-logs';source.mkdir()
        elif target in CACHE_TARGETS:
            source=root/'caches'/str(i);source.mkdir(parents=True)
        elif target in DATA_TARGETS:
            if not source.exists():raise ValueError('Retained data mount missing: '+str(source))
        else:
            if not m['ReadOnly']:raise ValueError('Unexpected writable source mount: '+target)
            if source.is_relative_to(reference_root):
                new=root/source.relative_to(reference_root)
            else:new=root/'frozen-external'/str(i)/source.name
            copy_source(source,new);source=new
        m['Source']=str(source);mounts.append(m)
    # Apply host directory overlays and every explicit child bind. Longest target wins.
    for m in mounts:
        p=package_source(package,m['Target'])
        if p is None:continue
        dest=Path(m['Source'])
        if not dest.is_relative_to(root):raise ValueError('Refusing an overlay write outside new stage: '+str(dest))
        if p.is_dir():shutil.copytree(p,dest,dirs_exist_ok=True)
        else:shutil.copy2(p,dest)
    return mounts


def freeze(root,package,reference,image,dtype,quality_choice,operator_sha,port=8097,qualify=False,_reference_root=None):
    if dtype not in ('bfloat16','fp8_ds_mla'):raise ValueError('Explicit supported cache dtype required')
    if type(port) is not int or not 1024<=port<=65535:raise ValueError('Explicit valid port required')
    if (root/'launch.json').exists():raise ValueError('Frozen launch already exists')
    choice=validate_quality_choice(quality_choice,dtype)
    reference_root=Path('/mnt/roms/star-abc-stage-21c95d921009') if _reference_root is None else _reference_root
    if sha(reference_root/'launch.json')!=sha(package/'prefill-reference/launch.json'):
        raise ValueError('Original dc87 launch receipt changed')
    reference_proof=verify_reference_pins(reference,package)
    s=copy.deepcopy(reference)
    s['mounts']=freeze_mounts(reference,reference_root,root,package)
    s.update(image=image,name='star-abc-test-'+('bf16' if dtype=='bfloat16' else 'fp8'),kv_cache_dtype=dtype,port=port)
    s['cmd'][s['cmd'].index('--kv-cache-dtype')+1]=dtype
    s['cmd'][s['cmd'].index('--port')+1]=str(port)
    if '--profiler-config' in s['cmd']:raise ValueError('Unexpected inherited profiler configuration')
    s['cmd']+=['--profiler-config',json.dumps(PROFILER,separators=(',',':')),'--jit-monitor-verbose']
    s['env'].update(STAR_ABC_KV_DTYPE=dtype,STAR_PREFILL='1' if qualify else '0',
                    STAR_PREFILL_SO='/opt/star-abc/lib/_dsv41_mm_try.abi3.so',STAR_PREFILL_SO_SHA256=operator_sha)
    imports=resolve(s['mounts'],package,SITE+'/hybrid/attention-imports.json')
    if imports is None or not imports.is_file():raise ValueError('Mounted import manifest missing')
    pins=json.loads(imports.read_text())
    for name in MODULES:
        module='hybrid.'+name;target=SITE+'/hybrid/'+name.replace('.','/')+'.py'
        if package_source(package,target) is None:target=target[:-3]+'/__init__.py'
        if module not in {row['module'] for row in pins['files']}:pins['files'].append(dict(module=module,target=target,sha256=''))
    for row in pins['files']:
        p=resolve(s['mounts'],package,row['target'])
        if p is None or not p.is_file():raise ValueError('Cannot resolve import pin '+row['target'])
        row['sha256']=sha(p)
    imports.write_text(json.dumps(pins,indent=2))
    pf=resolve(s['mounts'],package,'/w26-preflight/manifest.json')
    preflight=json.loads(pf.read_text())
    code_pins=frozen_code_pins(s['mounts'],package)
    existing={row['target'] for row in preflight['files']+preflight['data_files']}
    for row in code_pins:
        if row['target'] not in existing:preflight['data_files'].append(dict(row))
    for row in preflight['files']+preflight['data_files']:
        p=resolve(s['mounts'],package,row['target'])
        if p is None or not p.is_file():raise ValueError('Cannot resolve preflight pin '+row['target'])
        row['sha256']=sha(p)
    pf.write_text(json.dumps(preflight,indent=2))
    (root/'FROZEN-CODE-PINS.json').write_text(json.dumps(code_pins,indent=2))
    (root/'state/abc-policy.json').write_text(json.dumps(dict(schema='star.abc.policy.v1',armed=False,session=None,identity=None,epoch=0,profile='A',width=5,costs=None)))
    (root/'state/prefill-policy.json').write_text(json.dumps(dict(schema='star.prefill.policy.v1',armed=False,selected='REFERENCE',session=None,identity=None,epoch=0)))
    quality_dir=root/'cache-quality-choice';quality_dir.mkdir()
    shutil.copy2(quality_choice,quality_dir/'choice.json')
    for i,row in enumerate(choice['quality_evidence']):shutil.copy2(row['path'],quality_dir/(str(i)+'-'+Path(row['path']).name))
    s['notes']=['Prefill v2/full stage built from immutable dc87; A/B/C decode sources/native binaries retained.',
                'Cache was selected by the parent from measured quality evidence, not by this preparer.',
                'STAR_PREFILL defaults off. Opt-in prepares qualified REFERENCE/MOE_TRY on this one fixed cache boot.',
                'No candidate is selected at boot; all-rank numerical, memory and live engine qualification remain required.',
                'Model and Engram data sources retained; every code/state/cache source is frozen under this fresh stage.',
                'No current container is stopped or started by this preparer.']
    common=list(s['cmd']);common[common.index('--kv-cache-dtype')+1]='CACHE_AXIS'
    s['comparison_contract_sha256']=hashlib.sha256(json.dumps(dict(image=image,cmd=common,env={k:v for k,v in s['env'].items() if k!='STAR_ABC_KV_DTYPE'},host_config=s['host_config'],preflight=sha(pf)),sort_keys=True).encode()).hexdigest()
    (root/'launch.json').write_text(json.dumps(s,indent=2))
    attested=[]
    for row in preflight['files']+preflight['data_files']:
        assert sha(resolve(s['mounts'],package,row['target']))==row['sha256']
        attested.append(row)
    for target in ('/w9/integration/w9_exl_ab/worker_extension.py','/w9/resident-api/w9_resident_api/control.py'):
        assert sha(resolve(s['mounts'],package,target))==sha(package_source(package,target)),target
    receipt=dict(schema='star.prefill.staging.v2',root=str(root),image=image,kv_cache_dtype=dtype,launch_sha256=sha(root/'launch.json'),
                 preflight_sha256=sha(pf),attention_imports_sha256=sha(imports),cache_choice_sha256=sha(quality_choice),
                 operator_sha256=operator_sha,default_prefill_profile='REFERENCE',startup_opt_in=qualify,
                 original_reference_pins_verified=reference_proof,frozen_code_files=len(code_pins),
                 fresh_state=True,shared_state=False,source_child_mounts_verified=True,launched=False,gpu_qualified=False,model_quality_verified=False)
    (root/'STAGING-RECEIPT.json').write_text(json.dumps(receipt,indent=2))
    return receipt
