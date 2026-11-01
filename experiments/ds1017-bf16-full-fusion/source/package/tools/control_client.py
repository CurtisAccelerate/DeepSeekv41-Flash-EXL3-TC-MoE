"""Only the dedicated staged journal; no messages to other agents or production controls."""
from pathlib import Path
import json,uuid,time,os
def endpoint(spec):
    port=spec.get('port')
    if type(port) is not int or not 1024<=port<=65535:raise ValueError('Explicit valid test port required')
    args=spec['cmd']
    if '--port' not in args or args[args.index('--port')+1]!=str(port):raise ValueError('Launch port metadata and engine arguments differ')
    return 'http://127.0.0.1:'+str(port)
class Client:
    def __init__(self,root):
        self.root=Path(root).resolve();self.spec=json.loads((self.root/'launch.json').read_text())
        if self.spec.get('schema')!='star.abc.launch.v1' or self.spec['name'] not in ('star-abc-test','star-abc-test-fp8','star-abc-test-bf16'):raise ValueError('Dedicated ABC stage required')
        self.base_url=endpoint(self.spec)
        m=next(m for m in self.spec['mounts'] if m['Target']=='/w9/state')
        self.state=Path(m['Source'])
        if self.state.parent!=self.root:raise ValueError('Shared/original control directory refused')
    def call(self,body,timeout=420):
        rid=uuid.uuid4().hex;p=self.state/'inbox'/(rid+'.json');p.parent.mkdir(exist_ok=True)
        tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(body));os.replace(tmp,p)
        response=self.state/'responses'/p.name;start=time.monotonic()
        while not response.exists():
            if time.monotonic()-start>timeout:raise TimeoutError('ABC control receipt timed out: '+rid)
            time.sleep(.1)
        obj=json.loads(response.read_text())
        if obj.get('ok') is not True:raise RuntimeError('ABC control rejected: '+obj.get('error','unknown'))
        return obj
    def headers(self,session):
        token=(self.state/'benchmark-token').read_text().strip()
        return {'Content-Type':'application/json','X-W9-Benchmark-Token':token,'X-W26-Benchmark-Session':session}

