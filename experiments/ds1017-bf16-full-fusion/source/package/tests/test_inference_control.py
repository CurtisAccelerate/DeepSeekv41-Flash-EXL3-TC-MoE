"""Exercise the actual control mutation with a tensor created in inference mode."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'overlay/usr/local/lib/python3.12/dist-packages'))
try:
    import torch
    from hybrid import abc_gate
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'Requires image PyTorch/Triton runtime')
class InferenceControlTests(unittest.TestCase):
    def test_commit_and_end_reset_startup_inference_tensor(self):
        from hybrid.abc_contract import Selection
        from hybrid.abc_control import worker_invoke
        from unittest.mock import patch
        with torch.inference_mode():mask=torch.zeros(50,dtype=torch.int32)
        self.assertTrue(mask.is_inference())
        selection=Selection('test',['A','B','C'])
        bank=SimpleNamespace(selection=selection)
        worker=SimpleNamespace(rank=0,_abc_bank=bank)
        with patch.object(abc_gate,'MASK',mask),patch.object(torch.cuda,'synchronize'):
            for epoch,name in enumerate(('A','B','C','A'),1):
                selection.prepare('token',name,'session','test',epoch)
                self.assertFalse(torch.is_inference_mode_enabled())
                receipt=worker_invoke(worker,'commit','token')
                self.assertEqual(receipt['selected'],name)
                self.assertTrue(bool((mask==1).all()))
                with torch.inference_mode():mask.zero_()
            receipt=worker_invoke(worker,'end')
            self.assertEqual(receipt['selected'],'A')
            self.assertIsNone(receipt['session'])
            self.assertTrue(bool((mask==1).all()))

if __name__=='__main__':unittest.main()
