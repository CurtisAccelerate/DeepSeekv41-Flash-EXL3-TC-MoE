"""CPU-only admission, exception and selector checks of actual integrated Bank."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'overlay/usr/local/lib/python3.12/dist-packages'))
from hybrid.prefill_runtime import Bank
from hybrid.prefill_contract import FEATURES
def fixture():
 cuda=NS(capture=False,stream=7)
 cuda.is_current_stream_capturing=lambda:cuda.capture
 cuda.current_stream=lambda device:NS(cuda_stream=cuda.stream)
 worker=NS(rank=0,_abc_bank=NS(selection=NS(session='s')))
 calls=[];mode=['NONE']
 candidate=NS(supported=lambda *a,**k:True,combine_ragged=lambda *a,**k:calls.append((a,k)) or ('flat','ptr'))
 b=Bank(worker,None,NS(cuda=cuda,int32='int32'),None,{},7,lambda:mode[0],'a'*64,{},available=tuple(FEATURES),ragged=candidate)
 b.selection.prepare(dict(token='x',profile='MOE_RAGGED',session='s',identity='a'*64,epoch=1),'s');b.selection.commit('x')
 top=NS(dtype='int32',ndim=2,shape=(65,512),device='cuda:0',is_contiguous=lambda:True)
 meta=lambda n:NS(dtype='int32',numel=lambda:n)
 args=(top,meta(2),meta(1),meta(1),128,2,512,133248,131072)
 return b,args,cuda,worker,mode,calls
class Integration(unittest.TestCase):
 def test_actual_ratio2_and_swa0_production_signatures(self):
  b,a,_,_,_,calls=fixture();self.assertEqual(b.ragged_apply(*a,num_rows=133248),('flat','ptr'))
  swa=(*a[:5],0,0,2176,0);self.assertEqual(b.ragged_apply(*swa,num_rows=2176),('flat','ptr'))
  self.assertEqual(len(calls),2);self.assertEqual(sum(b.hits.values()),2)
 def test_default_session_and_unwarmed_fallback(self):
  b,a,_,w,_,calls=fixture();w._abc_bank.selection.session=None
  self.assertIsNone(b.ragged_apply(*a,num_rows=133248));w._abc_bank.selection.session='s'
  legacy=(*a[:5],4,*a[6:]);self.assertIsNone(b.ragged_apply(*legacy,num_rows=133248));self.assertFalse(calls)
 def test_capture_graph_and_stream_fallback(self):
  b,a,c,_,mode,calls=fixture();c.capture=True;self.assertIsNone(b.ragged_apply(*a,num_rows=133248));c.capture=False
  mode[0]='FULL';self.assertIsNone(b.ragged_apply(*a,num_rows=133248));mode[0]='NONE';c.stream=8
  self.assertIsNone(b.ragged_apply(*a,num_rows=133248));self.assertFalse(calls)
 def test_reentrancy_and_failure_clear_busy(self):
  b,a,_,_,_,_=fixture();b.in_call=True
  with self.assertRaises(RuntimeError):b.ragged_apply(*a,num_rows=133248)
  b.in_call=False
  def fail(*a,**k):raise ValueError('injected')
  b.ragged.combine_ragged=fail
  with self.assertRaises(ValueError):b.ragged_apply(*a,num_rows=133248)
  self.assertFalse(b.in_call)
