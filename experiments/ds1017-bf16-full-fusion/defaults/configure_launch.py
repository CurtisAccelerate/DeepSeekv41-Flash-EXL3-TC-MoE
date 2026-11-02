"""Apply a lane to a fresh frozen launch spec. Never loads/stops a model."""
from pathlib import Path
import argparse,copy,json
from lanes import config,environment,validate_command,digest
def configure(spec,lane,image):
 s=copy.deepcopy(spec);c=config(lane)
 if s.get('schema')!='star.abc.launch.v1' or s.get('kv_cache_dtype')!='bfloat16':raise ValueError('Use a fresh frozen BF16 dedicated-stage spec')
 if not image.startswith('sha256:') or len(image)!=71:raise ValueError('Use the immutable new release image ID')
 oldentry=s.get('entrypoint') or []
 if '/opt/star-release/lane_entrypoint.py' in oldentry:raise ValueError('Already configured; start from the original fresh spec')
 command=[*oldentry,*s['cmd']];validate_command(command)
 s.update(image=image,entrypoint=['python3','/opt/star-release/lane_entrypoint.py','--lane',lane,'serve','--'],cmd=command,env=environment(lane,s['env']),release_lane=c,release_lane_contract_sha256=digest(c),requires_new_gpu_qualification=True)
 # This is a new declared comparison contract, not a reused old arm identity.
 if s.get('comparison_contract_sha256'):s['previous_comparison_contract_sha256']=s['comparison_contract_sha256']
 s['comparison_contract_sha256']=digest(dict(image=image,entrypoint=s['entrypoint'],cmd=s['cmd'],env=s['env'],lane=c))
 return s
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--spec',type=Path,required=True);ap.add_argument('--image',required=True);ap.add_argument('--lane',choices=('speed','bf16','quality'),default='speed');a=ap.parse_args()
 if (a.spec.parent/'state/status.json').exists():raise SystemExit('This stage has been started; preserve it and prepare a fresh stage')
 original=a.spec.read_bytes();s=configure(json.loads(original),a.lane,a.image)
 backup=a.spec.with_suffix('.before-lanes.json')
 if backup.exists():raise SystemExit('Backup already exists; refuse repeated mutation')
 backup.write_bytes(original);a.spec.write_text(json.dumps(s,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(prepared=True,lane=a.lane,capture_scope=s['env']['STAR_ABC_SCOPE'],launched=False,needs_mounted_source_preflight_and_gpu_qualification=True)))
if __name__=='__main__':main()
