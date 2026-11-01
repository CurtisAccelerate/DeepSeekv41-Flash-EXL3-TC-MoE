"""Run during image build after compilation. No GPU operations."""
from pathlib import Path
import json,hashlib
root=Path('/opt/star-abc');pkg=Path('/usr/local/lib/python3.12/dist-packages');files=[]
for p in sorted((root/'overlay').rglob('*')):
    if p.is_file():
        target=Path('/')/p.relative_to(root/'overlay')
        files.append(dict(target=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
for p in sorted((root/'host-overlay').rglob('*.py')):
    rel=p.relative_to(root/'host-overlay')
    target='/w9/'+str(rel) if rel.parts[0] in ('integration','resident-api','engram-loader') else str(rel)
    files.append(dict(target=target,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
for pattern in ['dsv41_fp8w_ext*.so','dsv41_woa_ext*.so']:
    matches=list(pkg.glob(pattern))
    if len(matches)!=1:raise RuntimeError('Missing/ambiguous native extension '+pattern)
    p=matches[0];files.append(dict(target=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
p=root/'lib/_dsv41_mm_v2try.abi3.so';files.append(dict(target=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
identity=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(root/'manifest.json').write_text(json.dumps(dict(schema='star.abc.image.v1',identity=identity,files=files),indent=2))
print(json.dumps(dict(identity=identity,files=len(files))))

