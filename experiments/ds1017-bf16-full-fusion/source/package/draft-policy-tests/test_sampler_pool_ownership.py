import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from hybrid import abc_gate as G, abc_sampler as S

class SamplerPoolOwnership(unittest.TestCase):
 def run_case(self,mode):
  p=SimpleNamespace(device='cpu',_anchor_idx=torch.arange(8),input_ids=torch.arange(8,dtype=torch.long))
  saved=p.input_ids.clone();shared=torch.zeros((2,5),dtype=torch.long)
  old_mask,old_indices=G.MASK,G.INDICES
  G.MASK=torch.ones(50,dtype=torch.int32)
  G.INDICES={b:torch.tensor([6*r+1+j for r in range(b) for j in range(5)]) for b in (1,2)}
  states=[]
  class State:
   def __init__(self,b,gate):
    self.b=b;self.gate=gate;self.hidden=torch.zeros((b*5,16));self.calls=0
   def sample(self,hidden):
    self.calls+=1
    value=(hidden.float().sum().to(torch.long)+p.input_ids[:self.b].sum()+self.b)%129280
    out=torch.full((self.b,5),int(value),dtype=torch.long)
    if self.gate:
     self.keep=torch.ones((self.b,5),dtype=torch.int32)
     G.publish(self.keep,self.b)
     if mode=='bad_gate' and self.calls>1:G.MASK[G.INDICES[self.b]]=0
    if mode=='borrowed_output':
     shared[:self.b].copy_(out);out=shared[:self.b]
    if mode=='replay_corruption' and self.calls>1:out+=1
    return out
  states=[State(b,gate) for b in (1,2) for gate in (False,True)]
  try:
   with patch.object(torch.cuda,'synchronize'):
    try:return S.qualify_interleaved(p,states,lambda v:[v]*4)
    finally:self.assertTrue(torch.equal(saved,p.input_ids))
  finally:G.MASK,G.INDICES=old_mask,old_indices
 def test_serialized_outputs_survive_arbitrary_replay(self):
  r=self.run_case('good');self.assertTrue(r['passed']);self.assertEqual(r['cases'],8)
 def test_rejects_borrowed_output_storage(self):
  with self.assertRaisesRegex(RuntimeError,'ownership'):self.run_case('borrowed_output')
 def test_rejects_graph_replay_corruption(self):
  with self.assertRaisesRegex(RuntimeError,'ownership'):self.run_case('replay_corruption')
 def test_rejects_stale_gate_mask(self):
  with self.assertRaisesRegex(RuntimeError,'ownership'):self.run_case('bad_gate')

if __name__=='__main__':unittest.main()
