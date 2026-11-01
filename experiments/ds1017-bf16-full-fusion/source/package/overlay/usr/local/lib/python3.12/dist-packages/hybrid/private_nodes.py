"""Explicit private-graph CUDA updates. Never imports/mutates W9 registries.

ABI definitions match attested node_versions/collective_args. Only six-row
host profile is supported; DMA/pair48 transformations are deliberately rejected.
CUDA driver errors propagate; only deliberate Rejected exceptions are expected.
"""
import ctypes as C
from functools import wraps

class Rejected(Exception):pass
class KernelParams(C.Structure):
    _fields_=[('func',C.c_void_p)]+[(k,C.c_uint) for k in ['gridDimX','gridDimY','gridDimZ','blockDimX','blockDimY','blockDimZ','sharedMemBytes']]+[('kernelParams',C.POINTER(C.c_void_p)),('extra',C.POINTER(C.c_void_p))]
TYPES=(C.c_uint64,C.c_uint64,C.c_uint64,C.c_uint64,C.c_int,C.c_int64,C.c_int64,C.c_uint64)
class Arguments:
    def __init__(self,values):
        assert len(values)==8
        self.scalars=tuple(t(v) for t,v in zip(TYPES,values))
        self.pointers=(C.c_void_p*8)(*[C.cast(C.byref(v),C.c_void_p) for v in self.scalars])
    def bind(self,p):p.kernelParams=C.cast(self.pointers,C.POINTER(C.c_void_p));p.extra=C.POINTER(C.c_void_p)();return p
class Driver:
    def __init__(self):self.lib=C.CDLL('libcuda.so.1')
    def call(self,name,*args):
        f=getattr(self.lib,name);f.restype=C.c_int;code=f(*args)
        if code:raise RuntimeError(name+': CUDA '+str(code))
    def synchronize(self):self.call('cuCtxSynchronize')
def nodes(driver,graph):
    raw=C.c_void_p(graph.raw_cuda_graph());n=C.c_size_t();driver.call('cuGraphGetNodes',raw,None,C.byref(n))
    buf=(C.c_void_p*n.value)();driver.call('cuGraphGetNodes',raw,buf,C.byref(n))
    for handle in buf:
        node=C.c_void_p(handle);kind=C.c_int();driver.call('cuGraphNodeGetType',node,C.byref(kind))
        if kind.value==4:raise Rejected('Child graphs require separate qualification')
        if kind.value==0:
            params=KernelParams();driver.call('cuGraphKernelNodeGetParams',node,C.byref(params));yield node,params
def read_values(p):
    if not p.kernelParams or p.extra:raise Rejected('Unexpected kernel parameter representation')
    return tuple(t.from_address(p.kernelParams[i]).value for i,t in enumerate(TYPES))
def pointer(value):return value.value if isinstance(value,C.c_void_p) else int(value)

def inventory(method):
    """A rejected or partial inventory can never become an update plan."""
    @wraps(method)
    def checked(self,*args,**kwargs):
        if self.changed or self.invalid:raise Rejected('Inventory transaction unavailable')
        self.check_handles()
        try:
            result=method(self,*args,**kwargs)
            keys=[(item[1].value,item[2].value) for item in self.items]
            if len(keys)!=len(set(keys)):raise Rejected('Duplicate executable node update')
            self.sealed=self.plan_binding()
            return result
        except BaseException:
            self.invalid=True
            raise
    return checked

class PrivateUpdates:
    def __init__(self,driver,graphs,owners):
        if not graphs or not owners:raise Rejected('Graph and module/collective lifetime roots required')
        self.driver=driver;self.graphs=tuple(graphs);self.owners=tuple(owners)
        self.signature=tuple((g.raw_cuda_graph(),g.raw_cuda_graph_exec()) for g in graphs)
        if any(not raw or not exe for raw,exe in self.signature) or len({x[1] for x in self.signature})!=len(self.signature):raise Rejected('Null or duplicate executable graph')
        self.items=[];self.changed=[];self.owned_arguments=[]
        self.invalid=False;self.sealed=None
    def plan_binding(self):
        arguments=tuple(tuple(bytes(v) for v in arg.scalars) if isinstance(arg,Arguments) else (bytes(arg[0]),bytes(arg[1])) for arg in self.owned_arguments)
        return (tuple((id(g),exe.value,node.value,bytes(old),bytes(new),label) for g,exe,node,old,new,label in self.items),arguments)
    def check_handles(self):
        if self.signature!=tuple((g.raw_cuda_graph(),g.raw_cuda_graph_exec()) for g in self.graphs):raise Rejected('Private graph handles changed')
    @inventory
    def host_six(self,registry,expected_count,*,allow_qualified_existing=False):
        owner=registry['owner'];funcs=registry['funcs'];mapping={pointer(fn):algo for algo,fn in funcs.items()};found=0
        self.owners+= (registry,)
        for graph in self.graphs:
            for node,p in nodes(self.driver,graph):
                if p.func not in mapping:continue
                values=read_values(p);algo=mapping[p.func]
                if values[2:5]!=(owner.dev_base,owner.seq.data_ptr(),owner.rank):raise Rejected('Wrong private collective identity')
                if values[5]*16!=61440 or values[6]!=owner.slot_bytes:raise Rejected('Only exact61440-byte collective supported')
                geometry=(p.gridDimX,p.gridDimY,p.gridDimZ,p.blockDimX,p.blockDimY,p.blockDimZ)
                if allow_qualified_existing and algo==1 and geometry==(1,1,1,512,1,1):
                    # A fresh capture can inherit the currently selected six
                    # profile. Count and attest it; never pretend an update was
                    # needed or alter the baseline/global collective registry.
                    found+=1;continue
                if geometry!=(1,1,1,1024,1,1) or algo!=0:raise Rejected('Expected stock or explicitly allowed qualified six-row geometry')
                args=Arguments(values);original=args.bind(KernelParams.from_buffer_copy(bytes(p)))
                candidate=args.bind(KernelParams.from_buffer_copy(bytes(p)));candidate.func=pointer(funcs[1]);candidate.blockDimX=512
                self.owned_arguments.append(args);self.items.append((graph,C.c_void_p(graph.raw_cuda_graph_exec()),node,original,candidate,'host_six'));found+=1
        if found!=expected_count:raise Rejected('Incomplete private collective coverage: '+str(found))
        return found
    @inventory
    def native(self,Params,source_functions,target_functions,expected_counts):
        if C.sizeof(Params)!=344:raise Rejected('Unexpected EXL Params ABI')
        if set(source_functions)!=set(expected_counts) or set(target_functions)!=set(expected_counts):raise Rejected('Native stage keys differ')
        for functions in (source_functions,target_functions):
            addresses=[pointer(v) for v in functions.values()]
            if not all(addresses) or len(set(addresses))!=len(addresses):raise Rejected('Null or aliased native functions')
        names={pointer(fn):name for name,fn in source_functions.items()};counts={k:0 for k in expected_counts}
        for graph in self.graphs:
            for node,p in nodes(self.driver,graph):
                if p.func not in names:continue
                name=names[p.func]
                if name not in counts or not p.kernelParams or p.extra or not p.kernelParams[0]:raise Rejected('EXL function or argument representation differs')
                raw=C.create_string_buffer(C.string_at(p.kernelParams[0],C.sizeof(Params)));arg=Params.from_buffer_copy(raw)
                if (arg.bsz,arg.topk,arg.H,arg.slots_max,arg.ksplit_a,arg.ksplit_b)!=(6,6,5120,36,1,1) or arg.I not in (512,640):raise Rejected('EXL shape differs')
                if (arg.Hi,arg.Ho,arg.H_out,arg.min_expert,arg.max_expert,arg.n_local,arg.rows_max)!=(5120,5120,5120,0,384,384,6):raise Rejected('EXL shard/layout differs')
                if arg.g_bias or arg.u_bias or arg.d_bias or arg.sh_out or arg.sh_gate_w or arg.dbg or arg.gu_f32 or not arg.gated or not arg.a_global or arg.act!=0 or arg.act_limit!=10:raise Rejected('EXL optional/fused semantics differ')
                grid=180 if name=='rot' else 72*(arg.I//128) if name.startswith('a') else 36*40
                smem=0 if name=='rot' else (26624 if name.startswith('a') else 24576)+(8192 if name.endswith('3') else 0)
                if (p.gridDimX,p.gridDimY,p.gridDimZ,p.blockDimX,p.blockDimY,p.blockDimZ,p.sharedMemBytes)!=(grid,1,1,512,1,1,smem):raise Rejected('EXL geometry/shared memory differs')
                pointers=(C.c_void_p*1)(C.addressof(raw));original=KernelParams.from_buffer_copy(bytes(p));original.kernelParams=C.cast(pointers,C.POINTER(C.c_void_p))
                candidate=KernelParams.from_buffer_copy(bytes(original));candidate.func=pointer(target_functions[name])
                self.owned_arguments.append((raw,pointers));self.items.append((graph,C.c_void_p(graph.raw_cuda_graph_exec()),node,original,candidate,name));counts[name]+=1
        if counts!=expected_counts or counts.get('rot')!=counts.get('a2',0)+counts.get('a3',0) or counts.get('a2')!=counts.get('b2') or counts.get('a3')!=counts.get('b3'):raise Rejected('Incomplete EXL stage triples/counts')
        return counts
    def apply(self,fail_after=None):
        self.check_handles()
        if self.invalid or self.sealed is None or self.sealed!=self.plan_binding():raise Rejected('Incomplete or mutated update inventory')
        if self.changed:raise Rejected('Transaction already applied')
        self.driver.synchronize()
        for item in self.items:
            graph,exe,node,original,candidate,label=item
            self.driver.call('cuGraphExecKernelNodeSetParams',exe,node,C.byref(candidate));self.changed.append(item)
            if fail_after is not None and len(self.changed)==fail_after:raise Rejected('Intentional private partial-update rejection')
        return len(self.changed)
    def rollback(self):
        self.check_handles();self.driver.synchronize()
        while self.changed:
            graph,exe,node,original,candidate,label=self.changed[-1]
            self.driver.call('cuGraphExecKernelNodeSetParams',exe,node,C.byref(original))
            self.changed.pop()
        self.driver.synchronize()
