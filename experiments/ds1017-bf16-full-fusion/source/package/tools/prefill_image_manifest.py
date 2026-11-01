"""Pin the new image and prove old native/math bytes remain unchanged. No torch."""
from pathlib import Path
import hashlib,json

root=Path('/opt/star-abc'); package=Path('/opt/star-prefill/package')
reference=json.loads(Path('/opt/star-prefill/reference/image-manifest.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def resolve(target):
    return root/'host-overlay'/target[len('/w9/'):] if target.startswith('/w9/') else Path(target)
allowed={
 '/usr/local/lib/python3.12/dist-packages/hybrid/abc_startup.py',
 '/usr/local/lib/python3.12/dist-packages/hybrid/abc_kernels.py',
 '/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/amd/rocm.py',
 '/w9/integration/w9_exl_ab/worker_extension.py',
 '/w9/resident-api/w9_resident_api/control.py',
}
files={};changed=[];preserved=[]
for row in reference['files']:
    target=row['target'];actual=sha(resolve(target))
    if actual!=row['sha256']:
        assert target in allowed,('Unintended baseline byte change',target)
        changed.append(target)
    else:preserved.append(target)
    files[target]=actual
assert set(changed)==allowed,('Expected integration set differs',changed)
for p in (package/'overlay').rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts:
        target='/'+p.relative_to(package/'overlay').as_posix()
        assert sha(Path(target))==sha(p),('Installed overlay differs',target)
        files[target]=sha(p)
native='/opt/star-abc/lib/_dsv41_mm_try.abi3.so';files[native]=sha(Path(native))
rows=[dict(target=t,sha256=s) for t,s in sorted(files.items())]
identity=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(root/'manifest.json').write_text(json.dumps(dict(schema='star.abc.image.v1',identity=identity,files=rows),indent=2))
receipt=dict(passed=True,identity=identity,files=len(rows),changed_baseline_targets=sorted(changed),preserved_baseline_targets=preserved,
             native_sha256=files[native],native_target=native,base_native_binaries_preserved=[t for t in preserved if t.endswith('.so')],
             gpu_qualified=False,engine_qualified=False,model_quality_verified=False)
Path('/opt/star-prefill/image-receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
