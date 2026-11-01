import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'overlay/usr/local/lib/python3.12/dist-packages'))
try:import torch
except ImportError:torch=None
from hybrid.abc_gate_check import prefix_exact

@unittest.skipIf(torch is None,'Requires image PyTorch runtime')
class GateOutputABITests(unittest.TestCase):
    def test_all_eight_batch_sizes_and_every_accepted_prefix(self):
        for b in range(1,9):
            full=torch.arange(b*6*4).reshape(b*6,4)
            reference=(full,full[5::6].clone())
            for m in range(6):
                keep=torch.arange(b*6)%6<=m
                changed=full.clone();changed[~keep]+=7
                candidate=(changed,changed[5::6].clone())
                self.assertTrue(prefix_exact(reference,candidate,keep))
                changed[0,0]+=1
                self.assertFalse(prefix_exact(reference,(changed,changed[5::6].clone()),keep))
    def test_wrong_sample_map_and_unknown_shape_rejected(self):
        full=torch.arange(48*4).reshape(48,4);keep=torch.ones(48,dtype=torch.bool)
        with self.assertRaises(ValueError):prefix_exact((full,full[5::6]),(full,full[4::6]),keep)
        with self.assertRaises(ValueError):prefix_exact((full,full),(full,full),keep)

if __name__=='__main__':unittest.main()
