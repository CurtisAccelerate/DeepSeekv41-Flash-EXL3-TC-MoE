"""Freeze an isolated launch spec; never starts/stops a container or changes live code."""
from pathlib import Path
import argparse,json,hashlib,subprocess,shutil,os

SOCKET='unix:///run/w14-dsv41/docker.sock'
BASE='/mnt/nvme-new/w84-r2-image-20261026'
def inspect(name):return json.loads(subprocess.check_output(['docker','-H',SOCKET,'inspect',name]))[0]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare(root,image,kv_cache_dtype='fp8_ds_mla',port=8097):
    if type(port) is not int or not 1024<=port<=65535:raise ValueError('Valid explicit test port required')
    if kv_cache_dtype not in ('fp8_ds_mla','bfloat16'):raise ValueError('Explicit cache dtype required')
    root=root.resolve()
    if root.parent!=Path('/mnt/roms') or not root.name.startswith('star-abc-stage-'):raise ValueError('Use a dedicated /mnt/roms/star-abc-stage-* root')
    original=inspect('w84-r2c-resident')
    if original['Image']!='sha256:6aaf027853aad0d170260f74930a45c57511c2262cffb552fefc49edaaf79dc7':raise RuntimeError('Reference base image changed')
    image_id=inspect(image)['Id']
    snapshot=root/'snapshot'
    if snapshot.exists():raise RuntimeError('Snapshot already exists; do not overwrite a staged launch')
    snapshot.mkdir()
    original_engram=next(Path(m['Source']) for m in original['Mounts'] if m['Destination']=='/w9/engram-loader')
    shutil.copytree(original_engram,snapshot/'engram-loader')
    for name in ['integration','resident-api','rootfs','versions','launch-preflight']:
        shutil.copytree(Path(BASE)/name,snapshot/name,symlinks=True)
    package=root/'package';overlay=package/'overlay'
    mounts=[]
    for i,m in enumerate(original['Mounts']):
        source,target=Path(m['Source']),m['Destination']
        if target=='/w9/engram-loader':source=snapshot/'engram-loader'
        elif target=='/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py':source=snapshot/'engram-loader/engram.py'
        if str(source)==BASE or str(source).startswith(BASE+'/'):source=snapshot/source.relative_to(BASE)
        elif target=='/w9/state':source=root/'state';source.mkdir()
        elif target=='/r1out':source=root/'round-logs';source.mkdir()
        elif target in ['/root/.tilelang','/root/.triton','/root/.cache/torch','/root/.cache/vllm','/root/.nv/ComputeCache']:
            source=root/'caches'/str(i);source.mkdir(parents=True)
        candidate=overlay/target.lstrip('/')
        if target.startswith('/w9/integration/'):
            host=package/'host-overlay/integration'/target[len('/w9/integration/'):]
            if host.is_file():candidate=host
        elif target.startswith('/w9/resident-api/'):
            host=package/'host-overlay/resident-api'/target[len('/w9/resident-api/'):]
            if host.is_file():candidate=host
        if candidate.is_file():
            if not source.is_relative_to(root):
                source=root/'frozen-files'/str(i)/Path(target).name;source.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(candidate,source)
        mounts.append(dict(Type='bind',Source=str(source),Target=target,ReadOnly=not m['RW']))
    # Directory-mount members not shadowed by per-file mounts need the same overlay bytes.
    for sub in ['integration','resident-api','engram-loader']:
        for p in (package/'host-overlay'/sub).rglob('*'):
            if p.is_file():shutil.copyfile(p,snapshot/sub/p.relative_to(package/'host-overlay'/sub))
    def resolve(target):
        for m in sorted(mounts,key=lambda x:len(x['Target']),reverse=True):
            dest=m['Target']
            if target==dest:return Path(m['Source'])
            if target.startswith(dest+'/'):return Path(m['Source'])/target[len(dest)+1:]
        p=overlay/target.lstrip('/')
        return p if p.is_file() else None
    # Apply overlay source to every file bind (longest destination wins at runtime).
    for m in mounts:
        candidate=overlay/m['Target'].lstrip('/')
        if candidate.is_file():shutil.copyfile(candidate,Path(m['Source']))
    imports_target='/usr/local/lib/python3.12/dist-packages/hybrid/attention-imports.json'
    imports=resolve(imports_target)
    if imports is None or not imports.is_file():raise RuntimeError('Missing mounted attention import manifest')
    pins=json.loads(imports.read_text())
    for name in ('abc_kv_contract','abc_bf16_read','abc_bf16_io','abc_bf16_check'):
        module='hybrid.'+name;target='/usr/local/lib/python3.12/dist-packages/hybrid/'+name+'.py'
        if module not in {r['module'] for r in pins['files']}:
            pins['files'].append(dict(module=module,target=target,sha256=''))
    for row in pins['files']:
        p=resolve(row['target'])
        if p is None or not p.is_file():raise RuntimeError('Cannot resolve source pin '+row['target'])
        row['sha256']=sha(p)
    imports.write_text(json.dumps(pins,indent=2))
    preflight=snapshot/'launch-preflight/manifest.json';pf=json.loads(preflight.read_text())
    for row in pf['files']+pf['data_files']:
        p=resolve(row['target'])
        if p is None or not p.is_file():raise RuntimeError('Cannot resolve preflight pin '+row['target'])
        row['sha256']=sha(p)
    preflight.write_text(json.dumps(pf,indent=2))
    # Re-check both levels, including the preflight data-file hash of attention-imports.
    for row in pf['files']+pf['data_files']:
        assert sha(resolve(row['target']))==row['sha256'],row['target']
    env={e.split('=',1)[0]:e.split('=',1)[1] for e in original['Config']['Env']}
    env.update(STAR_ABC='1',STAR_ABC_PACK_MIB='4096',STAR_ABC_CAPTURE_MIB='2048',STAR_ABC_OPERATOR_CHECKS='1',DSV41_R1_ROUNDLOG='1')
    env['STAR_ABC_KV_DTYPE']=kv_cache_dtype
    # Disable noisy CUPTI by default; round/path records remain enabled. Trace is a separate experiment.
    env['DSV41_R1_TRACE_MODE']='off'
    args=list(original['Config']['Cmd'] or original['Args'])
    if '--kv-cache-dtype' in args:args[args.index('--kv-cache-dtype')+1]=kv_cache_dtype
    else:args+=['--kv-cache-dtype',kv_cache_dtype]
    i=args.index('--gpu-memory-utilization');args[i+1]='0.93'
    i=args.index('--compilation-config');cc=json.loads(args[i+1]);cc['cudagraph_capture_sizes']=[1,2,3,4,6,8,12,16,18,20,24,28,30,32,36,48];args[i+1]=json.dumps(cc)
    spec=dict(schema='star.abc.launch.v1',name='star-abc-test',image=image_id,source_image=original['Image'],entrypoint=original['Config'].get('Entrypoint') or [],env=env,cmd=args,mounts=mounts,host_config={k:original['HostConfig'].get(k) for k in ['IpcMode','ShmSize','CapAdd','SecurityOpt','Ulimits','DeviceRequests']},restart_policy='no',port=8096,
              notes=['Dedicated initial capture required; subsequent test selections do not load weights or recapture.','A/B/C share identical KV/power/cache policy. Staged gpu-memory-utilization .93 reserves experiment memory; actual KV capacity must pass startup.','No public-production quality approval is manufactured.','Live GNmac and the Hermes8095 relay remain untouched until a later explicit load.'])
    # Use a different API port and isolated journal. Model/Engram/registered host backing are the retained read-only data sources.
    i=spec['cmd'].index('--port');spec['cmd'][i+1]=str(port);spec['port']=port
    spec['kv_cache_dtype']=kv_cache_dtype
    spec['name']='star-abc-test-'+('bf16' if kv_cache_dtype=='bfloat16' else 'fp8')
    common_cmd=list(spec['cmd']);common_cmd[common_cmd.index('--kv-cache-dtype')+1]='CACHE_AXIS'
    common_env={k:v for k,v in env.items() if k!='STAR_ABC_KV_DTYPE'}
    spec['comparison_contract_sha256']=hashlib.sha256(json.dumps(dict(image=image_id,cmd=common_cmd,env=common_env,host_config=spec['host_config'],preflight=sha(preflight)),sort_keys=True).encode()).hexdigest()
    spec['notes']+=['Cache precision is fixed for this boot. Switching cache dtype requires another load; A/B/C remain switchable within the boot.','BF16 is an experimental read/gather twin with mandatory generated GPU and graph checks; CPU preflight does not qualify quality, capacity or speed.']
    (root/'launch.json').write_text(json.dumps(spec,indent=2))
    (root/'state/abc-policy.json').write_text(json.dumps(dict(schema='star.abc.policy.v1',armed=False,session=None,identity=None,epoch=0,profile='A',width=5,costs=None)))
    evidence=dict(kv_cache_dtype=kv_cache_dtype,image=image_id,root=str(root),launch_sha256=sha(root/'launch.json'),preflight_sha256=sha(preflight),attention_imports_sha256=sha(imports),state_isolated=True,shared_state=False,launched=False)
    (root/'STAGING-RECEIPT.json').write_text(json.dumps(evidence,indent=2));print(json.dumps(evidence,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--image',required=True);ap.add_argument('--port',type=int,default=8097);ap.add_argument('--kv-cache-dtype',choices=('fp8_ds_mla','bfloat16'),default='fp8_ds_mla');ap.add_argument('--reuse-resident-engram',action='store_true');a=ap.parse_args();prepare(a.root,a.image,a.kv_cache_dtype,a.port)
    if a.reuse_resident_engram:
        from bind_resident_engram import repair
        repair(a.root)

