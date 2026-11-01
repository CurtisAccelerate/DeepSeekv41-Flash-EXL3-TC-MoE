"""Finish an isolated launch using the existing resident Engram checkpoint bytes."""
from pathlib import Path
import json,subprocess,hashlib,shutil

def repair(root):
    root=Path(root);s=json.loads((root/'launch.json').read_text())
    resident=Path('/mnt/w14-engram-tmpfs-20261022')
    checkpoint=resident/'gnmac'
    if not (checkpoint/'model-00047-of-00048.safetensors').is_file():raise ValueError('Existing resident Engram unavailable')
    for m in s['mounts']:
        if m['Target']=='/engram-1017':m['Source']=str(checkpoint)
    prior=root/'snapshot/engram-prior-manifest.json'
    shutil.copyfile('/mnt/nvme-new/w14-single-load-20261022/engram/previous-manifest.json',prior)
    s['mounts'].append(dict(Type='bind',Source=str(prior),Target='/w9/engram-prior-manifest.json',ReadOnly=True))
    s['env']['W9_ENGRAM_BACKING']='/engram-resident/generation-abc-checkpoint-slices-20261101'
    cmd=['docker','-H','unix:///run/dsv41-private/docker.sock','run','--rm','--network=none','--entrypoint','python3']
    for k,v in s['env'].items():
        if k not in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES'):cmd+=['--env',k+'='+v]
    cmd+=['--env','CUDA_VISIBLE_DEVICES=','--env','NVIDIA_VISIBLE_DEVICES=void']
    for m in s['mounts']:cmd+=['--mount','type=bind,src='+m['Source']+',dst='+m['Target']+(',readonly' if m['ReadOnly'] else '')]
    cmd+=[s['image'],'/opt/star-abc/tools/prepare_engram_slices.py','--index',s['env']['STAR_DSV41_SELECTED_INDEX'],
          '--config',s['env']['W9_ENGRAM_CONFIG'],'--prior','/w9/engram-prior-manifest.json','--root',s['env']['W9_ENGRAM_BACKING']]
    result=json.loads(subprocess.check_output(cmd,text=True))
    s['env']['W9_ENGRAM_MANIFEST_SHA256']=result['manifest_sha256']
    pf_path=root/'snapshot/launch-preflight/manifest.json';pf=json.loads(pf_path.read_text())
    for target,path in [('/w9/engram-loader/backing.py',root/'snapshot/engram-loader/backing.py'),
                        ('/w9/engram-loader/resident_tables.py',root/'snapshot/engram-loader/resident_tables.py'),
                        ('/w9/engram-prior-manifest.json',prior),
                        (s['env']['W9_ENGRAM_BACKING']+'/manifest.json',resident/'generation-abc-checkpoint-slices-20261101/manifest.json')]:
        pf['data_files'].append(dict(target=target,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pf_path.write_text(json.dumps(pf,indent=2))
    common_cmd=list(s['cmd']);common_cmd[common_cmd.index('--kv-cache-dtype')+1]='CACHE_AXIS'
    common_env={k:v for k,v in s['env'].items() if k!='STAR_ABC_KV_DTYPE'}
    s['comparison_contract_sha256']=hashlib.sha256(json.dumps(dict(image=s['image'],cmd=common_cmd,env=common_env,host_config=s['host_config'],preflight=hashlib.sha256(pf_path.read_bytes()).hexdigest()),sort_keys=True).encode()).hexdigest()
    s['notes'].append('Prior resident generation was absent. Existing checkpoint slices reuse exactly the prior proven shard payload hashes; no data copied. Existing checkpoint NUMA placement is a new common condition, not the historical private-shard placement.')
    (root/'launch.json').write_text(json.dumps(s,indent=2))
    receipt=json.loads((root/'STAGING-RECEIPT.json').read_text());receipt.update(launch_sha256=hashlib.sha256((root/'launch.json').read_bytes()).hexdigest(),preflight_sha256=hashlib.sha256(pf_path.read_bytes()).hexdigest(),engram=result)
    (root/'STAGING-RECEIPT.json').write_text(json.dumps(receipt,indent=2));(root/'ENGRAM-REPAIR.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
