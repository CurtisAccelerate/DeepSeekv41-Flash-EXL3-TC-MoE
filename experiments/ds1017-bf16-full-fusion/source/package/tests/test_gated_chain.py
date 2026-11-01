import sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'overlay/usr/local/lib/python3.12/dist-packages'))
try:import torch
except ImportError:torch=None

@unittest.skipIf(torch is None,'Requires image PyTorch')
class GatedChainTests(unittest.TestCase):
    def test_capture_stream_is_reused_only_on_same_device(self):
        from hybrid import abc_sampler as S
        with patch.object(S,'_CAPTURE_STREAMS',{}),patch.object(torch.cuda,'Stream',side_effect=lambda **kw:object()) as factory:
            zero=S.capture_stream(torch.device('cuda',0))
            self.assertIs(S.capture_stream(torch.device('cuda',0)),zero)
            self.assertIsNot(S.capture_stream(torch.device('cuda',1)),zero)
            self.assertEqual(factory.call_count,2)
    def test_actual_chain_emits_int32_prefix_for_all_batches(self):
        from hybrid import abc_sampler as S
        for b in range(1,9):
            embedding=torch.nn.Embedding(16,4,dtype=torch.bfloat16)
            head=torch.nn.Linear(4,16,bias=False,dtype=torch.bfloat16)
            m=NS(model=NS(norm=torch.nn.Identity(),markov_head=NS(markov_w2=head)),
                 lm_head=head,logits_processor=lambda h,x,**kw:torch.nn.functional.linear(x,h.weight),
                 markov_embed=embedding,map_draft_to_target=lambda x:x,
                 compute_confidence=lambda h,e:torch.tensor([.9,.4,.2,.8,.8]).repeat(b))
            state=object.__new__(S.Integrated)
            state.p=NS(model=m);state.b=b;state.start=0;state.end=16
            state.fp8=False;state.fused=False;state.gate=True;state.nccl=None
            state.hidden=torch.zeros(b*5,4,dtype=torch.bfloat16);state.anchor=torch.zeros(b,dtype=torch.int64)
            with patch.object(S,'gather_pairs',lambda _,x:x[:,None,:].expand(-1,4,-1)):
                ids=state.chain()
            self.assertEqual(ids.shape,(b,5));self.assertEqual(state.keep.dtype,torch.int32)
            self.assertTrue(torch.equal(state.keep,torch.tensor([1,1,0,0,0],dtype=torch.int32).repeat(b,1)))
    def test_publish_accepts_chain_mask_and_preserves_anchor_slots(self):
        try:from hybrid import abc_gate as G
        except ImportError:self.skipTest('Requires image Triton import')
        from hybrid.abc_gate_check import confidence_keep
        for b in range(1,9):
            with torch.inference_mode():mask=torch.ones(50,dtype=torch.int32)
            index=torch.tensor([6*r+1+j for r in range(b) for j in range(5)])
            keep=confidence_keep(torch.tensor([.9,.4,.2,.8,.8]).repeat(b,1))
            with patch.object(G,'MASK',mask),patch.object(G,'INDICES',{b:index}):G.publish(keep,b)
            self.assertTrue(torch.equal(mask[index],keep.flatten()))
            self.assertTrue(bool((mask[torch.arange(b)*6]==1).all()))

if __name__=='__main__':unittest.main()
