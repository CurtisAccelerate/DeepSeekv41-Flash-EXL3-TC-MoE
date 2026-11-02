"""Read-only all-rank startup readiness; no generated request or model restart."""
from pathlib import Path
import argparse,json,sys,urllib.request
from lanes import config
from lane_client import validate_ready
def check(root,lane='speed'):
 sys.path.insert(0,str(root/'package/tools'));from control_client import Client
 c=Client(root);status=c.call(dict(op='abc_status'))
 receipts=[json.loads((root/'round-logs'/f'abc-preparation-rank{i}.json').read_text()) for i in range(4)]
 identity=validate_ready(c.spec,status['abc'],receipts,lane)
 pf=c.call(dict(op='prefill_status'));rows=pf['prefill'];assert len(rows)==4
 for r in rows:
  assert r['ready'] and r['pending'] is None and config(lane)['prefill'] in r['available']
  assert r['receipt']['numerical']['cases']==144
  rag=r['receipt']['components']['ragged'];assert rag['passed'] and rag['cases']==85 and rag['attention']['passed'] and rag['attention']['cases']==9
 with urllib.request.urlopen(c.base_url+'/health',timeout=10) as response:assert response.status==200
 return dict(passed=True,lane=lane,capture_scope=c.spec['env']['STAR_ABC_SCOPE'],bank_identity=identity,ranks=4,actual_weight_row_cases_per_rank=96,sampler_ownership_cases_per_rank=32,prefill_cases_per_rank=144,integer_cases_per_rank=85,same_attention_cases_per_rank=9,health=200,generated_requests=0,model_quality_verified=False)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--lane',choices=('speed','bf16','quality'),default='speed');ap.add_argument('--out',type=Path);a=ap.parse_args();r=check(a.root,a.lane)
 if a.out:a.out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(r,indent=2))
if __name__=='__main__':main()
