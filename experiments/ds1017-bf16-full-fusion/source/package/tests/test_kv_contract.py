import os,sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'overlay/usr/local/lib/python3.12/dist-packages'))
import torch
from types import SimpleNamespace as N
from hybrid import abc_kv_contract as kv

class KVContractTests(unittest.TestCase):
    def test_allocator_contract_excludes_known_sharing_aliases_and_rejects_bad_extents(self):
        spec=N(page_size_bytes=131072)
        groups=[N(layer_names=['L'+str(i)],kv_cache_spec=spec) for i in range(6)]
        groups[0].layer_names.append('alias')
        alloc=[N(block_stride=131072,size=262144,offset=0,layer_stride=131072,layers=['L'+str(i)]) for i in range(6)]
        cfg=N(kv_cache_groups=groups,kv_cache_tensors=alloc,num_blocks=2)
        self.assertEqual(kv.declared_block_bytes(cfg,{'alias':'L0'}),131072)
        with self.assertRaises(ValueError):kv.declared_block_bytes(cfg,{})
        alloc[2].offset=131072
        with self.assertRaises(ValueError):kv.declared_block_bytes(cfg,{'alias':'L0'})
    def test_explicit_boot_choice_and_isolation(self):
        with patch.dict(os.environ,{'STAR_ABC':'1','STAR_ABC_KV_DTYPE':'bfloat16'}):self.assertEqual(kv.requested_dtype(),'bfloat16')
        with patch.dict(os.environ,{'STAR_ABC':'0','STAR_ABC_KV_DTYPE':'bfloat16'}):
            with self.assertRaises(ValueError):kv.requested_dtype()
        with patch.dict(os.environ,{'STAR_ABC':'1','STAR_ABC_KV_DTYPE':'fp16'}):
            with self.assertRaises(ValueError):kv.requested_dtype()
    def test_noncompact_bf16_view_byte_extent_and_page_boundary(self):
        raw=torch.empty(3*196608,dtype=torch.bfloat16)
        t=raw.as_strided((3,128,512),(196608,512,1),1024)
        self.assertTrue(kv.validate_view(t,128,t.device,393216,cuda=False))
        self.assertFalse(kv.validate_view(t,128,t.device,230400,cuda=False))
        bad=raw.as_strided((2,128,512),(196608,512,1),196608-512)
        self.assertFalse(kv.validate_view(bad,128,bad.device,393216,cuda=False))
        self.assertFalse(kv.validate_view(t.to(torch.float16),128,t.device,393216,cuda=False))
        # Production eligibility always requires CUDA, even after CPU fixture validation.
        self.assertFalse(kv.validate_view(t,128,t.device,393216))
    def test_probe_cache_storage_matches_selected_format(self):
        with patch.dict(os.environ,{'STAR_ABC':'1','STAR_ABC_KV_DTYPE':'fp8_ds_mla'}):
            _,main,extra=kv.probe_views(torch,'cpu')
            self.assertEqual(main.dtype,torch.uint8);self.assertEqual(extra[128].shape,(1,128,584))
        with patch.dict(os.environ,{'STAR_ABC':'1','STAR_ABC_KV_DTYPE':'bfloat16'}),patch.object(kv,'_geometry',{'block_bytes':393216}):
            _,main,extra=kv.probe_views(torch,'cpu')
            self.assertEqual(main.dtype,torch.bfloat16)
            self.assertTrue(kv.validate_view(extra[128],128,main.device,393216,cuda=False))
    def test_unconfigured_bf16_capture_probe_is_rejected(self):
        with patch.dict(os.environ,{'STAR_ABC':'1','STAR_ABC_KV_DTYPE':'bfloat16'}),patch.object(kv,'_geometry',None):
            with self.assertRaises(ValueError):kv.probe_views(torch,'cpu')

if __name__=='__main__':unittest.main()
