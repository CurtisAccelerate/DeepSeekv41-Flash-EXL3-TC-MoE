import copy,json,sys,tempfile,unittest,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from make_costs import build as costs
from make_routes import build as routes
from run_matrix import finish_quality
from code_oracle import DRIVER

class MeasurementTests(unittest.TestCase):
    def test_full_cycle_costs_exclude_membership_changes_and_require_all_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'round-rank0.jsonl';rows=[dict(type='header',rank=0)];it=0;t=0
            for b in range(1,9):
                for name,k in [('C',5),('C3',3)]:
                    for epoch in (1,2):
                        for n in range(4):
                            it+=1;t+=25000000
                            rows.append(dict(type='round',complete=True,it=it,t_mono_ns=t,B=b,decode=b,prefill=0,unknown=0,tokens=b*(k+1),nd={str(i):k for i in range(b)},T={'greedy':b},joined=['new'] if n==0 else [],abc=dict(profile=name,identity='pin',session='s',epoch=epoch)))
            p.write_text('\n'.join(map(json.dumps,rows)))
            out=costs([p],'pin',min_samples=4)
            self.assertEqual(out['milliseconds']['8']['3'],25)
            self.assertEqual(out['samples']['1']['5']['count'],4)
            with self.assertRaises(ValueError):costs([p],'wrong',min_samples=4)
            with self.assertRaises(ValueError):costs([p],'pin',min_samples=5)
    def test_repeated_draft_comparisons_cannot_hide_a_failed_run(self):
        cell=dict(users=1,rep=0,case='facts',slot=0,response={'content':'ok'},screen={'pass_screen':True},completed_without_cap=True,coverage={'requested_B_observed_on_all_ranks':True})
        doc={'profiles':[dict(index=0,profile='B',cells=[copy.deepcopy(cell)]),dict(index=1,profile='C',cells=[copy.deepcopy(cell)]),dict(index=2,profile='C',cells=[copy.deepcopy(cell)])]}
        doc['profiles'][1]['cells'][0]['response']['content']='bad'
        finish_quality(doc)
        self.assertFalse(doc['fast_screen_pass']);self.assertFalse(doc['draft_isolation']['1:C']);self.assertTrue(doc['draft_isolation']['2:C'])
    def test_routes_choose_only_wall_improvements_without_workload_regression(self):
        def profile(name,rate):
            cells=[]
            for b in range(1,9):
                for prompt in ('count','code','prose'):
                    batches=[dict(requests=[{'completion_tokens':100}]*b,e_tok_s=rate) for _ in range(3)]
                    cells.append(dict(users=b,prompt=prompt,E=rate,batches=batches,coverage={'requested_B_observed_on_all_ranks':True}))
            return dict(profile=name,selection={'identity':'pin'},system_before={'hardware':'STAR-140W'},system_after={'hardware':'STAR-140W'},cells=cells)
        speed=dict(phase='speed',profiles=[profile('A',100),profile('B',110),profile('B-LANE',120),profile('A',100)],end={'ended':True})
        quality=dict(phase='quality',fast_screen_pass=True,profiles=[profile('B',0),profile('B-LANE',0)],end={'ended':True})
        for p in quality['profiles']:
            p['cells']=[dict(users=b,screen={'pass_screen':True},completed_without_cap=True) for b in range(1,9)]
        self.assertEqual(set(routes(speed,quality,[])['experts'].values()),{'lane'})
        speed['profiles'][2]['cells'][0]['batches'][0]['e_tok_s']=50
        speed['profiles'][2]['cells'][0]['batches'][1]['e_tok_s']=50
        self.assertEqual(routes(speed,quality,[])['experts']['1'],'hybrid')
        quality['profiles'][1]['cells'].pop()
        with self.assertRaises(ValueError):routes(speed,quality,[])
    def test_functional_oracle_distinguishes_plausible_wrong_code(self):
        # Trusted fixtures only; actual model-generated code uses the disposable Docker sandbox.
        good='def binary_search(xs,t):\n return xs.index(t) if t in xs else -1'
        bad='def binary_search(xs,t):\n return -1'
        for code,ok in [(good,True),(bad,False)]:
            p=subprocess.run([sys.executable,'-I','-S','-c',DRIVER],input=json.dumps(dict(case='binary_search',code=code)),text=True,capture_output=True)
            self.assertEqual(p.returncode==0,ok)

if __name__=='__main__':unittest.main()
