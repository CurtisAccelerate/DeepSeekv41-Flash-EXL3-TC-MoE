"""Compare fixed-prefix decode loss between matched FP8/BF16 boots; no API calls."""
import argparse,json,math,statistics
from pathlib import Path
from precision_common import compare_teacher,compare_ids

def compare(left,right,profile,left_vectors,right_vectors):
    if left.get('mode')!='teacher' or right.get('mode')!='teacher':raise ValueError('Frozen teacher runs required')
    lb,rb=left['boot'],right['boot']
    if lb['image']!=rb['image']:raise ValueError('KV isolation requires the identical staged image')
    if not lb.get('comparison_contract_sha256') or lb['comparison_contract_sha256']!=rb.get('comparison_contract_sha256'):
        raise ValueError('Common launch configuration/attestation differs')
    if (lb['kv_cache_dtype'],rb['kv_cache_dtype'])!=('fp8_ds_mla','bfloat16'):
        raise ValueError('Left must be FP8, right BF16')
    def select(doc):
        arm=next((p for p in doc['profiles'] if p['profile']==profile),None)
        if arm is None:raise ValueError('Missing same kernel profile')
        cases={c['label']:c for c in arm['cases']}
        if len(cases)!=len(arm['cases']):raise ValueError('Duplicate case labels')
        return cases
    x,y=select(left),select(right)
    if set(x)!=set(y) or not x:raise ValueError('Teacher case sets differ or empty')
    report=dict(schema='star.abc.kv-compare.v1',left_boot=lb,right_boot=rb,profile=profile,cases={},model_quality_verified=False)
    all_rows=[];left_nll=[];right_nll=[];groups={}
    for label,a in x.items():
        b=y[label]
        if not a['coverage']['requested_B_observed_on_all_ranks'] or not b['coverage']['requested_B_observed_on_all_ranks']:
            raise ValueError('Actual private target bank not observed on all ranks')
        if not compare_ids(a,b)['exact']:raise ValueError('Forced teacher completion IDs differ')
        aa={r['index']:r for r in a['teacher']};bb={r['index']:r for r in b['teacher']}
        if len(aa)!=len(a['teacher']) or len(bb)!=len(b['teacher']) or set(aa)!=set(bb):raise ValueError('Missing/duplicate score positions')
        rows=[];ln=[];rn=[]
        for index in sorted(aa):
            u,v=aa[index],bb[index]
            r=compare_teacher(u,v,left_vectors,right_vectors)
            if u['scheduled_tokens'] not in (4,6) or v['scheduled_tokens'] not in (4,6):continue
            if not r['graph_match']:raise ValueError('Decode graph descriptors differ')
            if any(not math.isfinite(z['nll']) or z['nll']<0 for z in (u,v)):raise ValueError('Invalid raw token loss')
            rows.append(r);ln.append(u['nll']);rn.append(v['nll'])
        if not rows:raise ValueError('No matched small-row decode scores')
        report['cases'][label]=dict(tokens=len(rows),fp8_nll=statistics.mean(ln),bf16_nll=statistics.mean(rn),
            delta_nll_bf16_minus_fp8=statistics.mean(r['delta_nll'] for r in rows),winner_changes=sum(r['winner_changed'] for r in rows),rows=rows)
        # Gold and deliberately wrong continuations have opposite interpretations.
        # Preserve the frozen corpus labels instead of pooling them into a quality score.
        kind='gold' if label.endswith('-gold') else ('wrong' if label.endswith('-wrong') else 'unclassified')
        group=groups.setdefault(kind,dict(left=[],right=[],rows=[]))
        group['left']+=ln;group['right']+=rn;group['rows']+=rows
        all_rows+=rows;left_nll+=ln;right_nll+=rn
    report['aggregate']=dict(tokens=len(all_rows),fp8_nll=statistics.mean(left_nll),bf16_nll=statistics.mean(right_nll),
        fp8_perplexity=math.exp(statistics.mean(left_nll)),bf16_perplexity=math.exp(statistics.mean(right_nll)),
        delta_nll_bf16_minus_fp8=statistics.mean(r['delta_nll'] for r in all_rows),winner_changes=sum(r['winner_changed'] for r in all_rows))
    report['aggregate']['interpretation']='Mixed-corpus numerical diagnostic only; use reference_groups for gold/wrong quality directions.'
    report['reference_groups']={kind:dict(tokens=len(g['rows']),fp8_nll=statistics.mean(g['left']),bf16_nll=statistics.mean(g['right']),
        delta_nll_bf16_minus_fp8=statistics.mean(r['delta_nll'] for r in g['rows']),winner_changes=sum(r['winner_changed'] for r in g['rows'])) for kind,g in groups.items()}
    report['interpretation']='Paired loss on these frozen continuations; gold/wrong controls must be interpreted separately. Not a general model perplexity benchmark, tool-chain success rate, or timing result.'
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fp8',required=True,type=Path);ap.add_argument('--bf16',required=True,type=Path)
    ap.add_argument('--profile',default='A');ap.add_argument('--fp8-vectors',required=True,type=Path);ap.add_argument('--bf16-vectors',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();doc=compare(json.loads(a.fp8.read_text()),json.loads(a.bf16.read_text()),a.profile,a.fp8_vectors,a.bf16_vectors)
    if a.out.exists():raise ValueError('Refusing to overwrite prior comparison')
    a.out.write_text(json.dumps(doc,indent=2));print(json.dumps(doc['aggregate'],indent=2))

if __name__=='__main__':main()
