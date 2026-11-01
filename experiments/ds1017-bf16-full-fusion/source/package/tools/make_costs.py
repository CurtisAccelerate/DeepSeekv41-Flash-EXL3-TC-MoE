"""Build local k3/k5 costs from steady consecutive rank-zero host cycle starts."""
import argparse,hashlib,json,statistics
from pathlib import Path

def build(paths,identity,min_samples=50,min_epochs=2):
    cells={(b,k):[] for b in range(1,9) for k in (3,5)};epochs={key:set() for key in cells};pins=[]
    def key(r):
        tag=r.get('abc',{});b=r.get('B');nd=r.get('nd',{});k={'C':5,'C3':3}.get(tag.get('profile'))
        if tag.get('identity')!=identity or not tag.get('session') or k is None or b not in range(1,9):return None
        if r.get('complete') is not True or r.get('decode')!=b or r.get('prefill') or r.get('unknown') or r.get('tokens')!=b*(k+1):return None
        if len(nd)!=b or set(nd.values())!={k} or r.get('T',{}).get('greedy')!=b:return None
        if any(r.get(f) for f in ('joined','resumed','finished','preempted')):return None
        return b,k,tag['session'],tag['epoch'],tuple(sorted(nd))
    for path in paths:
        raw=path.read_bytes();pins.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()));prev=None
        lines=[json.loads(line) for line in raw.splitlines() if line.strip()]
        headers=[r for r in lines if r.get('type')=='header']
        if len(headers)!=1 or headers[0].get('rank')!=0:raise ValueError('One complete rank-zero log per input file required')
        for r in lines:
            if r.get('type')!='round':continue
            a,z=key(prev or {}),key(r)
            if a is not None and a==z and r['it']==prev['it']+1:
                ms=(r['t_mono_ns']-prev['t_mono_ns'])/1e6
                if not 0<ms<10000:raise ValueError('Invalid cycle interval; do not clip unexplained stalls')
                cells[a[:2]].append(ms);epochs[a[:2]].add(a[2:4])
            prev=r
    missing=[str(key) for key,v in cells.items() if len(v)<min_samples or len(epochs[key])<min_epochs]
    if missing:raise ValueError('Insufficient steady local cycles/independent selection epochs: '+', '.join(missing))
    return dict(schema='star.abc.costs.v1',identity=identity,hardware='STAR-140W',source='measured_full_cycle',
        clock='rank0 consecutive complete steady execute_model start monotonic_ns; host + GPU pipeline, not isolated kernel time',
        milliseconds={str(b):{str(k):statistics.mean(cells[b,k]) for k in (3,5)} for b in range(1,9)},
        samples={str(b):{str(k):dict(count=len(cells[b,k]),epochs=len(epochs[b,k]),median_ms=statistics.median(cells[b,k]),max_ms=max(cells[b,k])) for k in (3,5)} for b in range(1,9)},evidence=pins,
        evidence_sha256=hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest(),production_approved=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--identity',required=True);ap.add_argument('--logs',required=True,nargs='+',type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
    a.out.write_text(json.dumps(build(a.logs,a.identity),indent=2))
