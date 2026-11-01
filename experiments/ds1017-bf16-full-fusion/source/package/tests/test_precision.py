import sys,json,tempfile,unittest,dataclasses
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'overlay/usr/local/lib/python3.12/dist-packages'));sys.path.insert(0,str(ROOT/'tools'))
import torch
from hybrid import abc_probe,abc_fidelity,abc_lane_check
from hybrid.abc_contract import PROFILES,Selection
from precision_common import compare_ids,parse_response,compare_teacher

class PrecisionTests(unittest.TestCase):
    def extra(self,ids=None):return dict(abc_teacher_ids=ids or [30,31,32,33,34,35,36],abc_probe_label='fixture',abc_teacher_prompt_sha256=abc_probe.ids_hash([10,20]))
    def test_teacher_contract_refuses_ambiguous_ids_and_wrong_prompt(self):
        for field,value in [('abc_teacher_ids',[True]),('abc_teacher_ids',[129280]),('abc_probe_label','../../outside'),('abc_teacher_prompt_sha256','wrong'),('abc_probe_full',2),('abc_probe_full','true'),('abc_probe_full',1.0)]:
            e=self.extra();e[field]=value
            with self.assertRaises(ValueError):abc_probe.validate(e,[10,20])
        self.assertIsNone(abc_probe.validate({},[10,20]))
    def test_teacher_vector_flag_transport_normalization(self):
        for value,want in [(True,True),(False,False),(1,True),(0,False)]:
            extra=self.extra();extra['abc_probe_full']=value
            self.assertIs(abc_probe.validate(extra,[10,20])['full'],want)
    def test_real_openai_request_teacher_vector_flag_boundary(self):
        try:from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        except ImportError:self.skipTest('Installed vLLM API schema required; mandatory in immutable-image tests')
        for value,want in [(True,True),(False,False)]:
            extra=self.extra();extra['abc_probe_full']=value
            req=ChatCompletionRequest(model='fixture',messages=[{'role':'user','content':'hi'}],vllm_xargs=extra)
            self.assertIs(abc_probe.validate(req.vllm_xargs,[10,20])['full'],want)
    def test_teacher_position_anchor_and_beyond_end_mapping(self):
        p=abc_probe.validate(self.extra(),[10,20])
        self.assertEqual(abc_probe.expected(p,1),(0,30));self.assertEqual(abc_probe.expected(p,2),(1,31))
        self.assertEqual(abc_probe.expected(p,8),(7,0))
    def test_same_visible_text_does_not_hide_different_numeric_tokenization(self):
        a=dict(prompt_sha256='same',token_ids=[1,2]);b=dict(prompt_sha256='same',token_ids=[3])
        self.assertFalse(compare_ids(a,b)['exact']);b['prompt_sha256']='different'
        with self.assertRaises(ValueError):compare_ids(a,b)
    def test_missing_numeric_ids_are_not_inferred_from_decoded_strings(self):
        with self.assertRaises(ValueError):parse_response(dict(choices=[dict(logprobs={'content':[{'token':'hello'}]},message={'content':'hello'})]))
    def test_oracle_only_replaces_comparison_ids_and_refuses_sampling(self):
        @dataclasses.dataclass
        class Metadata:
            draft_token_ids:object
            num_draft_tokens:object
            other:object
        m=Metadata(torch.arange(5),[5],object());bank=SimpleNamespace(hits=__import__('collections').Counter())
        with patch('hybrid.abc_runtime.current',return_value=PROFILES['B-ORACLE']),patch('hybrid.abc_runtime.BANK',bank):
            got=abc_fidelity.rejection(m,SimpleNamespace(all_greedy=True))
            self.assertTrue(bool((got.draft_token_ids==-1).all()));self.assertTrue(torch.equal(m.draft_token_ids,torch.arange(5)))
            self.assertIs(got.other,m.other)
            with self.assertRaises(RuntimeError):abc_fidelity.rejection(m,SimpleNamespace(all_greedy=False))
    def test_probe_scores_original_logits_then_forces_known_prefix(self):
        plan=abc_probe.validate(self.extra(),[10,20]);selection=Selection('pin',['A','B']);selection.selected='B';selection.session='s';selection.epoch=1
        bank=SimpleNamespace(selection=selection,worker=SimpleNamespace(rank=0),hits=__import__('collections').Counter())
        logits=torch.zeros(6,129280);logits[:,99]=3.
        inputs=torch.tensor([30,31,32,33,34,35]);positions=torch.arange(2,8)
        runner=SimpleNamespace(_abc_teacher_plans={'r':plan},input_batch=SimpleNamespace(req_ids=['r'],num_reqs=1,sampling_metadata=SimpleNamespace(all_greedy=True)),input_ids=SimpleNamespace(gpu=inputs),_get_positions=lambda indices:positions)
        desc=SimpleNamespace(num_tokens=6,num_reqs=1,uniform=True,has_lora=False,num_active_loras=0)
        with tempfile.TemporaryDirectory() as tmp,patch('hybrid.abc_runtime.BANK',bank),patch('hybrid.abc_probe.Path',return_value=Path(tmp)):
            abc_probe.observe(runner,SimpleNamespace(num_scheduled_tokens={'r':6}),logits,torch.arange(6),SimpleNamespace(num_draft_tokens=[5]),desc)
            records=[json.loads(l) for l in (Path(tmp)/'abc-teacher-fixture.jsonl').read_text().splitlines()]
        self.assertEqual([r['teacher_token_id'] for r in records],[31,32,33,34,35,36])
        self.assertEqual({r['winner'] for r in records},{99});self.assertTrue(all(r['nll']>0 for r in records))
        self.assertEqual(torch.argmax(logits,dim=1).tolist(),[31,32,33,34,35,36])
    def test_full_vector_kl_and_measured_margin_certificate(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);np.save(root/'a.npy',np.array([2.,0.,-1.]));np.save(root/'b.npy',np.array([2.01,0.,-1.]))
            a=dict(prompt_sha256='p',teacher_sha256='t',index=1,input_position=5,teacher_token_id=0,nll=.1,winner=0,margin=2.,graph=[6,1,True,False,0],full_vector='a.npy')
            b=dict(a,full_vector='b.npy',nll=.099)
            got=compare_teacher(a,b,root)
            self.assertGreaterEqual(got['kl_left_to_right'],0);self.assertTrue(got['argmax_stability_certificate_for_this_pair'])
            b['teacher_sha256']='different'
            with self.assertRaises(ValueError):compare_teacher(a,b,root)
    def test_independent_normalized_hadamard_is_orthogonal(self):
        x=torch.arange(256,dtype=torch.float64).reshape(2,128)/128
        self.assertTrue(torch.allclose(abc_lane_check.had128(abc_lane_check.had128(x)),x,atol=1e-12,rtol=0))

if __name__=='__main__':unittest.main()
