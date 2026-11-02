"""Prepare a source-only release context outside the repository."""
from pathlib import Path
import argparse,shutil
ap=argparse.ArgumentParser();ap.add_argument('--experiment',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
source=a.experiment.resolve();out=a.out.resolve();assert source.is_dir() and not out.exists() and not out.is_relative_to(source)
out.mkdir();shutil.copytree(source,out/'publication',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('DEFAULT-RECIPE.json','balanced_default_client.py'):shutil.copyfile(source/'defaults'/name,out/name)
for name in ('Dockerfile','verify_release.py','materialize_host_overlay.py'):shutil.copyfile(source/'release'/name,out/name)
print(str(out))
