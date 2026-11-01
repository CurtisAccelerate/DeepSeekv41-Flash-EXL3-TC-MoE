"""Experimental per-B expert choice from paired full-wall speed and functional screens."""
import argparse,hashlib,json,statistics,math
from pathlib import Path

def identity(doc):
    ids={p['selection']['identity'] for p in doc['profiles']}
    if len(ids)!=1:raise ValueError('Mixed capture identities')
    return next(iter(ids))

def build(speed,quality,pins):
    ident=identity(speed)
    if speed.get('phase')!='speed' or quality.get('phase')!='quality' or identity(quality)!=ident or not quality.get('fast_screen_pass'):raise ValueError('Matched speed and passing built-in quality results required')
    for doc in (speed,quality):
        if doc.get('end_error') or not doc.get('end',{}).get('ended'):raise ValueError('Incomplete/failed test session')
        for p in doc['profiles']:
            if p.get('system_before',{}).get('hardware')!='STAR-140W' or p.get('system_after',{}).get('hardware')!='STAR-140W':raise ValueError('Missing measured hardware/power envelope')
    for name in ('B','B-LANE'):
        tested={c['users'] for p in quality['profiles'] if p['profile']==name for c in p['cells'] if c.get('screen',{}).get('pass_screen') and c.get('completed_without_cap')}
        if tested!=set(range(1,9)):raise ValueError('Every B1..8 target route requires its own concurrent quality screen')
    # Repeated A endpoints expose drift; refuse to calibrate if the full-wall rate moves >5%.
    refs=[p for p in speed['profiles'] if p['profile']=='A']
    if len(refs)<2:raise ValueError('Bracket the speed matrix with A controls')
    def rates(profile,b):
        grouped={}
        for p in speed['profiles']:
            if p['profile']!=profile:continue
            for c in p['cells']:
                if c['users']!=b:continue
                if c.get('coverage',{}).get('requested_B_observed_on_all_ranks') is not True:raise ValueError('Requested B target path was not observed on every rank')
                batches=c['batches']
                if len(batches)<3 or any(len(x['requests'])!=b or any(not r.get('completion_tokens') for r in x['requests']) for x in batches):raise ValueError('Three complete paired batches required')
                for x in batches:
                    v=x['e_tok_s']
                    if not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:raise ValueError('Invalid full-wall rate')
                grouped.setdefault(c['prompt'],[]).extend(x['e_tok_s'] for x in batches)
        return {p:statistics.median(v) for p,v in grouped.items()}
    choice={};comparisons={}
    for b in range(1,9):
        controls=[{c['prompt']:c['E'] for c in p['cells'] if c['users']==b} for p in (refs[0],refs[-1])]
        if set(controls[0])!={'count','code','prose'} or controls[0].keys()!=controls[1].keys():raise ValueError('Incomplete drift controls')
        if any(abs(controls[1][p]/controls[0][p]-1)>.05 for p in controls[0]):raise ValueError('A wall-rate drift exceeds 5%; rerun a balanced matrix')
        hybrid,lane=rates('B',b),rates('B-LANE',b)
        if set(hybrid)!={'count','code','prose'} or hybrid.keys()!=lane.keys():raise ValueError('Matched count/code/prose cells required')
        ratios={p:lane[p]/hybrid[p] for p in hybrid};gain=math.prod(ratios.values())**(1/len(ratios))
        # Conservative: no workload regresses >2%, aggregate gain exceeds 2%.
        choice[str(b)]='lane' if gain>1.02 and min(ratios.values())>=.98 else 'hybrid'
        comparisons[str(b)]=dict(lane_over_hybrid_wall_ratios=ratios,geomean=gain,selected=choice[str(b)])
    return dict(schema='star.abc.routes.v1',identity=ident,hardware='STAR-140W',source='matched_star_wall_and_quality',experts=choice,comparisons=comparisons,
        evidence=pins,evidence_sha256=hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest(),production_approved=False,
        limitation='Functional screen and workload-specific wall calibration only; not broad model-quality qualification')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--speed',required=True,type=Path);ap.add_argument('--quality',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
    pins=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (a.speed,a.quality)]
    a.out.write_text(json.dumps(build(json.loads(a.speed.read_text()),json.loads(a.quality.read_text()),pins),indent=2))
