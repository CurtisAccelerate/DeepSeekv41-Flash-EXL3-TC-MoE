from pathlib import Path
import hashlib,json,shutil
shutil.copytree('/opt/star-release/publication/source/package','/opt/star-prefill/package',dirs_exist_ok=True)
root=Path('/opt/star-abc');manifest=json.loads((root/'manifest.json').read_text())
for row in manifest['files']:
    if not row['target'].startswith('/w9/'):continue
    source=root/'host-overlay'/row['target'][len('/w9/'):]
    data=source.read_bytes()
    assert hashlib.sha256(data).hexdigest()==row['sha256'],str(source)
    dest=Path(row['target']);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
print('Materialized pinned host-overlay files for standalone image CPU attestation; no model data copied.')
