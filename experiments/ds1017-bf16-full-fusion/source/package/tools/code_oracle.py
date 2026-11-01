"""Execute generated snippets only in a bounded, networkless, GPU-less disposable container."""
import json,subprocess,uuid
from pathlib import Path

DRIVER='''import json,sys,math
d=json.load(sys.stdin);ns={};exec(compile(d['code'],'candidate','exec'),ns)
if d['case']=='moving_average':
 f=ns['moving_average']
 assert f([],1)==[] and f([1,2],3)==[] and f([1,2,3,4],2)==[1.5,2.5,3.5]
 assert f([-2,2,6],1)==[-2,2,6]
 for w in [0,-1,1.5,True]:
  try:f([1,2],w)
  except ValueError:pass
  else:raise AssertionError('invalid window accepted')
else:
 f=ns['binary_search'];assert f([],1)==-1 and f([1,3,5],4)==-1
 for xs,t in [([1],1),([1,2,3],1),([1,2,3],3),([1,2,2,2,3],2),([-9,-2,0,7],-2)]:
  i=f(xs,t);assert isinstance(i,int) and 0<=i<len(xs) and xs[i]==t
print(json.dumps({'pass_screen':True,'oracle':d['case']}))
'''

def check(case,code,root):
    if case not in ('moving_average','binary_search') or not isinstance(code,str):return dict(pass_screen=False,reason='missing code')
    image=json.loads((Path(root)/'launch.json').read_text())['image'];name='star-abc-oracle-'+uuid.uuid4().hex
    prefix=['docker','-H','unix:///run/dsv41-private/docker.sock']
    cmd=prefix+['run','--rm','-i','--name',name,'--network','none','--read-only','--tmpfs','/tmp:rw,size=32m','--memory','512m','--cpus','1','--pids-limit','32','--cap-drop','ALL','--security-opt','no-new-privileges','--env','CUDA_VISIBLE_DEVICES=','--entrypoint','python3',image,'-I','-S','-c',DRIVER]
    try:
        p=subprocess.run(cmd,input=json.dumps(dict(case=case,code=code)),capture_output=True,text=True,timeout=15)
        if p.returncode:return dict(pass_screen=False,reason='functional oracle failed',stderr=p.stderr[-3000:])
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (subprocess.TimeoutExpired,ValueError,IndexError) as exc:return dict(pass_screen=False,reason=type(exc).__name__)
    finally:
        # Exact generated task-owned container name only; never enumerate/delete other containers.
        subprocess.run(prefix+['rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
