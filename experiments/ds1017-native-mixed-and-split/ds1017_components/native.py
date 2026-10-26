"""Explicit startup-only native kernel adapter. No live graph mutation."""
import ctypes as C
import hashlib,os
from pathlib import Path
ROT='_ZN16exl3_moe_coop_ns24exl3_moe_coop_rot_kernelE13MoeCoopParams'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eligible(args,backend):
    """CPU metadata only. Unsupported conditions call the original extension."""
    if len(args)!=36:return False
    x,sel,rw,lo,hi,H,*_=args
    tables=args[6:15];bias=args[15:18];Kg,Ku,Kd,mcg,mul1,act,limit,gated=args[18:26]
    hadg,hadu,gug,guu,actout,dout,ctr,out=args[26:34]
    if backend.cc!=(8,0) or x.device!=backend.device or any(v is not None for v in (*bias,*args[34:36])):return False
    if not (lo==0 and hi==384 and H==5120 and Kg==Ku==Kd and Kg in (2,3)
            and not mcg and mul1 and act==0 and limit==10.0 and gated):return False
    if os.environ.get('EXL3_MOE_COOP_KSPLIT','1') not in ('','0','1'):return False
    if os.environ.get('EXL3_MOE_COOP_WIDE','1') not in ('','-1','1'):return False
    if os.environ.get('EXL3_MOE_COOP_DBG','0') not in ('','0'):return False
    tensors=(x,sel,rw,*tables,hadg,hadu,gug,guu,actout,dout,ctr,out)
    if not all(t.is_cuda and t.is_contiguous() and t.device==x.device for t in tensors):return False
    def dtype(t,name):return str(t.dtype)=='torch.'+name
    if not all(dtype(t,'float16') for t in (x,rw,hadg,hadu,gug,guu,actout)):return False
    if not all(dtype(t,'int64') for t in (sel,*tables)):return False
    if not dtype(ctr,'int32') or not all(dtype(t,'float32') for t in (dout,out)):return False
    if tuple(x.shape)!=(6,5120) or x.stride(0)!=5120:return False
    if tuple(sel.shape)!=(6,6) or tuple(rw.shape)!=(6,6) or tuple(out.shape)!=(6,5120):return False
    if any(tuple(t.shape)!=(384,) for t in tables):return False
    I=int(guu.shape[-1])
    if I not in (512,640):return False
    if any(t.shape[-1]!=5120 for t in (hadg,hadu,dout)):return False
    if any(t.shape[-1]!=I for t in (gug,guu,actout)):return False
    slots=min(t.numel()//w for t,w in ((hadg,5120),(hadu,5120),(gug,I),(guu,I),(actout,I),(dout,5120)))
    if slots!=36 or ctr.numel()<(36*(I//128)+6*40+72+3):return False
    return True

class Backend:
    def __init__(self,bundle,device,expected_sha256):
        import torch
        from .params import Params
        self.torch=torch;self.Params=Params;self.lib=C.CDLL('libcuda.so.1')
        self.device=device;prop=torch.cuda.get_device_properties(device);self.cc=(prop.major,prop.minor)
        if self.cc!=(8,0):raise RuntimeError('Native mixed preload requires SM80')
        current=C.c_void_p();self.call('cuCtxGetCurrent',C.byref(current))
        if not current.value:raise RuntimeError('Native mixed requires worker current context')
        self.context=current.value;self.modules={};self.functions={}
        for arm in ('native_mixed',):
            path=Path(bundle)/(arm+'.cubin')
            if sha(path)!=expected_sha256:raise RuntimeError('Native mixed cubin identity mismatch: '+arm)
            mod=C.c_void_p();self.call('cuModuleLoad',C.byref(mod),C.c_char_p(os.fsencode(path)))
            self.modules[arm]=mod;fn={}
            names={'rot':ROT}
            for k in (2,3):
                for stage in ('a','b'):names[f'{stage}{k}']=f'_ZN16exl3_moe_coop_ns22exl3_moe_coop_{stage}_kernelILi{k}ELi2ELb1ELb0EEEv13MoeCoopParams'
            for name,symbol in names.items():
                f=C.c_void_p();self.call('cuModuleGetFunction',C.byref(f),mod,C.c_char_p(symbol.encode()));fn[name]=f
            self.functions[arm]=fn
        # Modules intentionally retained until worker context destruction.
    def call(self,name,*args):
        f=getattr(self.lib,name);f.restype=C.c_int;r=f(*args)
        if r:raise RuntimeError(f'{name}: CUDA {r}')
    def synchronize(self):self.torch.cuda.synchronize(self.device)
    def free_bytes(self):
        self.synchronize();return int(self.torch.cuda.mem_get_info(self.device)[0])
    def memory_snapshot(self):
        self.synchronize()
        free,total=self.torch.cuda.mem_get_info(self.device)
        return dict(free=int(free),total=int(total),allocated=int(self.torch.cuda.memory_allocated(self.device)),reserved=int(self.torch.cuda.memory_reserved(self.device)))
    def launch(self,arm,args):
        current=C.c_void_p();self.call('cuCtxGetCurrent',C.byref(current))
        if current.value!=self.context:raise RuntimeError('Native mixed CUDA context changed')
        p=self.Params();x,sel,rw,lo,hi,H=args[:6]
        p.x=x.data_ptr();p.sel=sel.data_ptr();p.rw=rw.data_ptr();p.x_stride=x.stride(0)
        p.bsz=6;p.topk=6;p.H=p.Hi=p.Ho=p.H_out=5120;p.min_expert=lo;p.max_expert=hi;p.n_local=384
        for name,t in zip(('g_trellis','g_suh','g_svh','u_trellis','u_suh','u_svh','d_trellis','d_suh','d_svh'),args[6:15]):setattr(p,name,t.data_ptr())
        p.act=0;p.act_limit=10;p.gated=True;p.a_global=True;p.gu_f32=False;p.ksplit_a=p.ksplit_b=1;p.dbg=0
        for name,t in zip(('had_g','had_u','gu_g','gu_u','act_out','d_out'),args[26:32]):setattr(p,name,t.data_ptr())
        p.I=int(args[29].shape[-1]);p.slots_max=36;p.rows_max=6;p.ctr_a_len=36*(p.I//128);p.ctr_b_len=6*40
        p.ctr_a=args[32].data_ptr();p.ctr_b=p.ctr_a+p.ctr_a_len*4;p.runs=p.ctr_b+p.ctr_b_len*4
        p.out=args[33].data_ptr();p.out_stride=args[33].stride(-2)
        stream=C.c_void_p(self.torch.cuda.current_stream(self.device).cuda_stream)
        k=int(args[18]);stage=8192 if k==3 else 0
        fn=self.functions[arm]
        params=(C.c_void_p*1)(C.addressof(p))
        for name,grid,smem in (('rot',180,0),(f'a{k}',36*2*(p.I//128),10240+16384+stage),(f'b{k}',36*40,16384+stage+8192)):
            self.call('cuLaunchKernel',fn[name],*[C.c_uint(v) for v in (grid,1,1,512,1,1,smem)],stream,params,C.c_void_p())

def prepare_native(bundle,device,expected_sha256):
    """Load in an existing current context before capture; default off.

    expected_sha256 must come from the locally reviewed build artifact.
    Keep this returned object alive until every dependent graph is destroyed.
    """
    if os.environ.get('DS1017_NATIVE_MIXED','0')!='1':return None
    if len(expected_sha256)!=64:raise ValueError('Expected build SHA256')
    return Backend(bundle,device,expected_sha256)

def dispatch(backend,original,*args):
    if backend is None or not eligible(args,backend):return original(*args)
    backend.launch('native_mixed',args)
    return None
