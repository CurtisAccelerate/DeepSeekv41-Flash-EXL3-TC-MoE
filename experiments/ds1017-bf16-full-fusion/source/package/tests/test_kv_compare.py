import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from compare_kv_boots import compare

class KVCompareTests(unittest.TestCase):
    def fixture(self):
        boot=dict(image='same',comparison_contract_sha256='matched',kv_cache_dtype='fp8_ds_mla')
        scores=[dict(prompt_sha256='p',teacher_sha256='t',index=i,input_position=10+i,teacher_token_id=30+i,
            graph=[6,1,True,False,0],scheduled_tokens=6,nll=.2,winner=30+i,margin=.1) for i in range(2)]
        case=dict(label='fixture-short-gold',prompt_sha256='p',token_ids=[30,31],teacher=scores,coverage={'requested_B_observed_on_all_ranks':True})
        left=dict(mode='teacher',boot=boot,profiles=[dict(profile='A',cases=[case])])
        right=json.loads(json.dumps(left));right['boot']['kv_cache_dtype']='bfloat16'
        right['profiles'][0]['cases'][0]['teacher'][1]['nll']=.1
        return left,right
    def test_paired_loss_and_reference_group(self):
        a,b=self.fixture();r=compare(a,b,'A',Path('.'),Path('.'))
        self.assertAlmostEqual(r['reference_groups']['gold']['delta_nll_bf16_minus_fp8'],-.05)
        self.assertEqual(r['aggregate']['tokens'],2)
    def test_rejects_changed_engine_or_teacher_shape(self):
        for change in ('image','contract','prefix','missing','coverage'):
            a,b=self.fixture();case=b['profiles'][0]['cases'][0]
            if change=='image':b['boot']['image']='different'
            elif change=='contract':b['boot']['comparison_contract_sha256']='different'
            elif change=='prefix':case['teacher'][1]['teacher_sha256']='different'
            elif change=='missing':case['teacher'].pop()
            else:case['coverage']['requested_B_observed_on_all_ranks']=False
            with self.assertRaises(ValueError):compare(a,b,'A',Path('.'),Path('.'))
