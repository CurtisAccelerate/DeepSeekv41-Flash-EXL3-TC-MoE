"""CPU build check before copying any new runtime source. No torch import."""
from pathlib import Path
import hashlib,json

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
root=Path('/opt/star-abc')
reference=json.loads(Path('/opt/star-prefill/reference/image-manifest.json').read_text())
present=json.loads((root/'manifest.json').read_text())
assert present==reference,'Exact dc87 base manifest required'
for row in reference['files']:
    target=row['target']
    p=root/'host-overlay'/target[len('/w9/'):] if target.startswith('/w9/') else Path(target)
    assert p.is_file() and sha(p)==row['sha256'],('Base bytes differ',target)
Path('/opt/star-prefill/base-verification.json').write_text(json.dumps(dict(passed=True,files=len(reference['files']),image='sha256:dc87ecb3dfdca94ce12a958c09a26314affd5f1fc2df7eb00dd2b82d9203675b',gpu_touched=False),indent=2))
