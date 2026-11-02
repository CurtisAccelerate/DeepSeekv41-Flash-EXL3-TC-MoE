"""Prepare the speed-default source/build context outside this repository."""
from pathlib import Path
import argparse,shutil
ap=argparse.ArgumentParser();ap.add_argument('--experiment',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
source=a.experiment.resolve();out=a.out.resolve();assert source.is_dir() and not out.exists() and not out.is_relative_to(source)
out.mkdir();shutil.copytree(source,out/'publication',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('DEFAULT-RECIPE.json','LANES.json','lanes.py','lane_entrypoint.py','lane_client.py','configure_launch.py','check_ready.py'):shutil.copyfile(source/'defaults'/name,out/name)
for name in ('Dockerfile','verify_release.py','materialize_host_overlay.py','test_lanes.py','README.md'):shutil.copyfile(source/'release/speed-default'/name,out/name)
print(str(out))
