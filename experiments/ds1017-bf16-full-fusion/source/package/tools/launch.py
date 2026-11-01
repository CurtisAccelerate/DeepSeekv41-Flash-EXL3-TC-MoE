"""Execute an already frozen dedicated test launch, only when the GPU owner has stopped."""
import argparse,json,subprocess,socket
from pathlib import Path
SOCKET='unix:///run/w14-dsv41/docker.sock'
def run(*a,**kw):return subprocess.run(['docker','-H',SOCKET,*a],check=True,**kw)
ap=argparse.ArgumentParser();ap.add_argument('spec',type=Path);ap.add_argument('--execute',action='store_true');a=ap.parse_args();s=json.loads(a.spec.read_text())
if not a.execute:print(json.dumps(s,indent=2));raise SystemExit(0)
from control_client import endpoint
endpoint(s)
with socket.socket() as probe:
    try:probe.bind(('0.0.0.0',s['port']))
    except OSError as exc:raise SystemExit('Test API port is occupied; existing listeners are preserved: '+str(exc))
for socket in [SOCKET,'unix:///run/gnmac-docker.sock']:
    ps=[json.loads(line) for line in subprocess.check_output(['docker','-H',socket,'ps','--format','json']).decode().splitlines() if line]
    if any(p.get('Names') in ('gnmac-tp4','w84-r2c-resident','star-abc-test','star-abc-test-fp8','star-abc-test-bf16') for p in ps):raise SystemExit('Existing GPU owner still running. This launcher never stops it.')
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
if processes:raise SystemExit('GPU compute processes are still present. This launcher never stops them.')
cmd=['run','-d','--name',s['name'],'--network','host','--ipc',s['host_config']['IpcMode'],'--gpus','all','--restart','no']
for cap in s['host_config'].get('CapAdd') or []:cmd+=['--cap-add',cap]
for setting in s['host_config'].get('SecurityOpt') or []:cmd+=['--security-opt',setting]
for limit in s['host_config'].get('Ulimits') or []:cmd+=['--ulimit',limit['Name']+'='+str(limit['Soft'])+':'+str(limit['Hard'])]
if s['entrypoint']:cmd+=['--entrypoint',s['entrypoint'][0]]
for k,v in s['env'].items():cmd+=['--env',k+'='+v]
for m in s['mounts']:
    cmd+=['--mount','type=bind,src='+m['Source']+',dst='+m['Target']+(',readonly' if m['ReadOnly'] else '')]
cmd+=[s['image'],*s['entrypoint'][1:],*s['cmd']]
run(*cmd)
