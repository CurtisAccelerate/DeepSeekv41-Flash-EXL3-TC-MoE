"""Verify published source bytes and parse Python; privacy-only helper edits are documented."""
from pathlib import Path
import ast,hashlib,json
root=Path(__file__).resolve().parent
d=json.loads((root/'SOURCE-MANIFEST.json').read_text(encoding='utf-8'))
for r in d['files']:
 p=root/r['path'];assert p.is_file() and p.stat().st_size==r['bytes']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],r['path']
 if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
print(json.dumps(dict(passed=True,exported_files=len(d['files']),manifest_matches=True)))
