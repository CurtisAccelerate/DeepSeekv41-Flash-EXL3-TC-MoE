"""Metadata and opt-in behavior without torch, GPU, or model weights."""
import ctypes,os,unittest
from unittest.mock import patch
from types import SimpleNamespace
from ds1017_components import attention,native
from ds1017_components.attention_profiles import Profile
from ds1017_components.params import Params

class Tensor:
    def __init__(self,shape,dtype):self.shape=shape;self.dtype='torch.'+dtype;self.device='cuda:0';self.is_cuda=True
    def is_contiguous(self):return True
    def stride(self,axis):return self.shape[-1]
    def numel(self):
        n=1
        for v in self.shape:n*=v
        return n

def args():
    t=Tensor
    return [t((6,5120),'float16'),t((6,6),'int64'),t((6,6),'float16'),0,384,5120,
        *[t((384,),'int64') for _ in range(9)],None,None,None,
        3,3,3,False,True,0,10.,True,
        t((36,5120),'float16'),t((36,5120),'float16'),
        t((36,640),'float16'),t((36,640),'float16'),t((36,640),'float16'),
        t((36,5120),'float32'),t((495,),'int32'),t((6,5120),'float32'),None,None]

class Checks(unittest.TestCase):
    def test_defaults_off(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertIsNone(native.prepare_native('unused','unused','unused'))
            self.assertIsNone(attention.prepare_attention('unused'))
        self.assertEqual(native.dispatch(None,lambda *a:'fallback'),'fallback')
    def test_profile(self):
        p=Profile(8,8,False)
        self.assertEqual(p.workspace_bytes,(6*8*16*2+6*8*16*512)*4+6*16*512*2+4)
        for bad in ((True,8,False),(7,8,False),(8,8,True)):
            with self.assertRaises(ValueError):Profile(*bad)
    def test_abi(self):self.assertEqual(ctypes.sizeof(Params),344)
    def test_native_shapes(self):
        b=SimpleNamespace(cc=(8,0),device='cuda:0');a=args()
        with patch.dict(os.environ,{},clear=True):
            self.assertTrue(native.eligible(a,b))
            for bits in (2,3):
                a[18:21]=[bits]*3;self.assertTrue(native.eligible(a,b))
            a[18:21]=[5]*3;self.assertFalse(native.eligible(a,b))
            a=args();a[0].shape=(12,5120);self.assertFalse(native.eligible(a,b))
            a=args();a[15]=object();self.assertFalse(native.eligible(a,b))
            a=args();a[32].shape=(10,);self.assertFalse(native.eligible(a,b))
            a=args();a[0].is_cuda=False;self.assertFalse(native.eligible(a,b))
    def test_attention_fallback(self):
        self.assertEqual(attention.decode_during_capture(None,None,lambda **k:7),7)
        self.assertEqual(attention.decode_during_capture(object(),SimpleNamespace(layer_id=0),lambda **k:9,swa_only=True),9)

if __name__=='__main__':unittest.main()
