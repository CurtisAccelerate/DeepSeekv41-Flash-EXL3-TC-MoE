"""Opt-in Engram binding. Import is CPU-only; attach explicitly registers CUDA.

Mappings and registrations deliberately live until worker process destruction.
Never unregister/rebind memory while a graph or UVA view might retain it.
"""
import ctypes as C
import os
from backing import Mapping, verify_selection

_LIVE = []

def enabled():
    return bool(os.environ.get('W9_ENGRAM_BACKING'))

class Registered:
    def __init__(self, mapping, dtype, *, allow_cuda=False):
        if not allow_cuda: raise RuntimeError('explicit CUDA registration authorization required')
        import torch
        self.mapping=mapping
        if not mapping.mutable_shared or os.environ.get('W9_ENGRAM_ALLOW_MUTABLE_SHARED')!='1':
            raise RuntimeError('explicit reviewed shared-RW registration mode required')
        # MAP_SHARED is deliberate: MAP_PRIVATE could trigger a full-table COW
        # duplicate during registration. This variant does not claim immutable pages.
        self.tensor=torch.frombuffer(mapping.buffer,dtype=dtype).reshape(mapping.entry['shape'])
        self.lib=C.CDLL('libcuda.so.1')
        context=C.c_void_p()
        self.call('cuCtxGetCurrent',C.byref(context))
        if not context.value: raise RuntimeError('register only inside initialized worker context')
        self.context=context.value
        # Explicit DEVICEMAP-only variant: CMP170HX attr113=0 cannot register
        # GPU-read-only host memory. The native pinned baseline is writable too.
        self.call('cuMemHostRegister_v2',C.c_void_p(getattr(mapping,'registration_address',self.tensor.data_ptr())),
                  C.c_size_t(getattr(mapping,'registration_nbytes',self.tensor.numel()*self.tensor.element_size())),C.c_uint(2))
        self.registered=True
        if not self.tensor.is_pinned():
            raise RuntimeError('external registration is not recognized by torch; refuse hidden pin copy')
        self.device_pointer=C.c_uint64()
        self.call('cuMemHostGetDevicePointer_v2',C.byref(self.device_pointer),
                  C.c_void_p(self.tensor.data_ptr()),C.c_uint(0))
    def call(self,name,*args):
        fn=getattr(self.lib,name);fn.restype=C.c_int
        result=fn(*args)
        if result: raise RuntimeError(f'{name} failed: CUDA {result}')

def attach_pair(layer, rank, tp, dp, row_start, row_end, dim, block_size):
    import torch
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    root=os.environ['W9_ENGRAM_BACKING']; expected=os.environ['W9_ENGRAM_MANIFEST_SHA256']
    pair=[]
    for kind,dtype,width in (('weight',torch.float8_e4m3fn,dim),
                             ('weight_scale_inv',torch.uint8,dim//block_size)):
        m=Mapping(root,expected,layer,kind,rank,mutable_shared=True)
        print('W9_ENGRAM_VERIFIED',layer,kind,rank,m.manifest.get('payload_layout','private-shards'),m.entry['nbytes'],m.entry['payload_sha256'],flush=True)
        index=os.environ['STAR_DSV41_SELECTED_INDEX']
        verify_selection(m.manifest,index,os.environ['W9_ENGRAM_CONFIG'])
        e=m.entry
        if (m.manifest['tp'],m.manifest['dp'],e['row_start'],e['row_end'],e['shape']) != (tp,dp,row_start,row_end,[row_end-row_start,width]):
            raise RuntimeError('Engram shard geometry differs from pinned generation')
        expected_dtype=('F8_E4M3',) if kind=='weight' else ('U8','F8_E8M0')
        if e['dtype'] not in expected_dtype:raise RuntimeError('Engram dtype mismatch')
        r=Registered(m,dtype,allow_cuda=True)
        # Guard the actual helper against its native fallback allocating/copying.
        view=get_accelerator_view_from_cpu_tensor(r.tensor)
        if view.data_ptr()!=r.device_pointer.value:raise RuntimeError('UVA helper copied backing')
        p=torch.nn.Parameter(r.tensor,requires_grad=False)
        p.engram_resident_entry=e
        p.engram_resident_owner=r
        p.engram_resident_view=view
        _LIVE.append(r);pair.append(p)
    return tuple(pair)

def validate_loaded(param, loaded_weight):
    """Checkpoint dispatch remains satisfied; values already byte-verified.

    safetensors get_tensor is a lazy source view; this hook never narrows/copies
    it. Manifest digest is supplied externally, not inferred from model names.
    """
    import torch
    e=param.engram_resident_entry
    if list(loaded_weight.shape)!=e['full_shape']: raise RuntimeError('checkpoint shape changed')
    legal=(torch.float8_e4m3fn,) if e['kind']=='weight' else (torch.uint8,torch.float8_e8m0fnu)
    if loaded_weight.dtype not in legal:raise RuntimeError('checkpoint dtype changed')
