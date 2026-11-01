"""Publish metadata for exact old Engram shards in existing resident checkpoints.

No payload allocation/copy, CUDA import, page migration or checkpoint writes.
Actual attach retains full shard SHA256 verification against the prior proven
payloads. The prior NUMA allocation is gone; shared-checkpoint placement is
explicitly a new common boot condition for every A/B/C comparison arm.
"""
import argparse,json,os,hashlib
from pathlib import Path
from backing import canonical,digest,hash_file,config_identity,make_plan,ensure_tmpfs

PRIOR='b8b10e3b30831aae6248525c4d8a9573cc4df367a68373c91e93abbeb0985710'

def prepare(index,config,prior_path,root):
    if hash_file(prior_path)!=PRIOR:raise ValueError('Proven prior Engram receipt changed')
    prior=json.loads(Path(prior_path).read_text())
    if config_identity(config)!=prior['engram_config']:raise ValueError('Engram geometry changed')
    names={}
    old={}
    for e in prior['entries']:
        names.setdefault(e['layer'],{})[e['kind']]=e['checkpoint_key']
        old[e['layer'],e['kind'],e['rank']]=e
    plan=make_plan(index,config,names,prior['layout'],tp=4,dp=1)
    root=Path(root)
    root.mkdir(mode=0o700,exist_ok=True);ensure_tmpfs(root)
    for e in plan['entries']:
        ref=old[e['layer'],e['kind'],e['rank']]
        for key in ('checkpoint_key','shape','full_shape','row_start','row_end','nbytes'):
            if e[key]!=ref[key]:raise ValueError('Proven shard geometry differs: '+key)
        if e['kind']=='weight' and e['dtype']!=ref['dtype']:raise ValueError('Engram weight format changed')
        if e['kind']=='weight_scale_inv' and e['dtype'] not in ('U8','F8_E8M0'):raise ValueError('Unexpected scale encoding')
        filename=Path(e['source']).name
        alias=root/('source-'+filename)
        payload=Path('/engram-resident/gnmac')/filename
        if not payload.is_file() or payload.stat().st_ino!=Path(e['source']).stat().st_ino:
            raise ValueError('Read-only source and registration alias do not share the existing resident file')
        if not alias.exists():alias.symlink_to(payload)
        e.update(file=alias.name,payload_offset=e['offset'],payload_sha256=ref['payload_sha256'],prior_numa_node=ref.get('numa_node'))
    body=dict(plan,storage='tmpfs',payload_layout='checkpoint-slices-v1',prior_payload_manifest_sha256=PRIOR,
              payload_copied=False,numa_policy='existing resident checkpoint placement; shared across all comparison arms')
    manifest=dict(body,generation=digest(canonical(body)))
    if (root/'manifest.json').exists():
        if (root/'manifest.json').read_bytes()!=canonical(manifest):raise ValueError('Existing Engram metadata generation differs')
    else:
        (root/'manifest.json').write_bytes(canonical(manifest));os.chmod(root/'manifest.json',0o400)
    receipt=dict(root=str(root),manifest_sha256=hash_file(root/'manifest.json'),entries=len(plan['entries']),
        payload_bytes=sum(e['nbytes'] for e in plan['entries']),payload_copied=False,
        payload_hash_checked_now=False,full_payload_hash_mandatory_on_worker_attach=True,prior_payload_manifest_sha256=PRIOR)
    print(json.dumps(receipt,indent=2));return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--config',required=True);p.add_argument('--prior',required=True);p.add_argument('--root',required=True)
    a=p.parse_args();prepare(a.index,a.config,a.prior,a.root)
