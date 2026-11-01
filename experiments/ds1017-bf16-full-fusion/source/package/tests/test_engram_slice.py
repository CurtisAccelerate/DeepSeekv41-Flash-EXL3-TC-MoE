import sys,unittest,tempfile,json,struct,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host-overlay/engram-loader'))
from backing import Mapping,header,source_identity,canonical,digest

class EngramSliceTests(unittest.TestCase):
    def fixture(self,root):
        data=b'0123456789abcdef'*512
        src=root/'checkpoint.safetensors';hdr=json.dumps({'table':{'dtype':'U8','shape':[len(data),1],'data_offsets':[0,len(data)]}}).encode()
        src.write_bytes(struct.pack('<Q',len(hdr))+hdr+data)
        (root/'payload').hardlink_to(src)
        _,base,hh=header(src);st=src.stat();offset=base+17;payload=data[17:4117]
        e=dict(layer='L',kind='weight',rank=0,source=str(src),header_sha256=hh,file='payload',offset=offset,payload_offset=offset,nbytes=len(payload),payload_sha256=digest(payload))
        e.update(dict(zip(('source_size','source_mtime_ns','source_ctime_ns','source_dev','source_ino'),source_identity(st))))
        body=dict(entries=[e],storage='tmpfs',payload_layout='checkpoint-slices-v1')
        manifest=dict(body,generation=digest(canonical(body)));p=root/'manifest.json';p.write_bytes(canonical(manifest))
        return p,digest(p.read_bytes()),src,payload
    def test_unaligned_slice_maps_exact_bytes_without_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,pin,_,payload=self.fixture(root)
            m=Mapping(root,pin,'L','weight',0,test_disk=True,mutable_shared=True)
            self.assertEqual(bytes(m.buffer),payload)
            self.assertGreater(m.registration_nbytes,len(payload))
            self.assertEqual(m.registration_address%4096,0);m.close()
    def test_corrupt_slice_rejected_against_prior_payload_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,pin,src,_=self.fixture(root)
            e=json.loads(p.read_text())['entries'][0]
            with src.open('r+b') as f:f.seek(e['payload_offset']+3);f.write(b'X')
            # Update selected-file stat identity only, leaving the trusted payload hash.
            d=json.loads(p.read_text());e=d['entries'][0]
            e.update(dict(zip(('source_size','source_mtime_ns','source_ctime_ns','source_dev','source_ino'),source_identity(src.stat()))))
            d['generation']=digest(canonical({k:v for k,v in d.items() if k!='generation'}));p.write_bytes(canonical(d))
            with self.assertRaisesRegex(ValueError,'payload corruption'):
                Mapping(root,digest(p.read_bytes()),'L','weight',0,test_disk=True,mutable_shared=True)
