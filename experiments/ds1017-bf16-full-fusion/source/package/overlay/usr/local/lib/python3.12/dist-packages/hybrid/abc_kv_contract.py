"""Explicit boot-only cache choice and checked block-interleaved BF16 arena.

The indexer retains its original quantized storage. BF16 here refers to the
attention SWA and compressed latent caches, not weights or indexer precision.
"""
import os

_geometry = None
_receipt = None

def requested_dtype():
    value=os.environ.get('STAR_ABC_KV_DTYPE','fp8_ds_mla')
    if value not in ('fp8_ds_mla','bfloat16'):
        raise ValueError('Unsupported explicit staged cache dtype')
    if value=='bfloat16' and os.environ.get('STAR_ABC')!='1':
        raise ValueError('BF16 twin is restricted to isolated STAR_ABC staging')
    return value

def validate_view(t,block,device,block_bytes,*,cuda=True):
    if (t is None or (cuda and not t.is_cuda) or t.device!=device
            or str(t.dtype)!='torch.bfloat16' or t.ndim!=3
            or t.shape[0]<1 or tuple(t.shape[1:])!=(block,512)
            or t.stride()!=(block_bytes//2,512,1) or block_bytes%2):return False
    start=t.storage_offset()*2
    # A layer page cannot cross the enclosing shared physical block.
    if start%block_bytes+block*1024>block_bytes:return False
    end=start+(t.shape[0]-1)*block_bytes+block*1024
    return end<=t.untyped_storage().nbytes()

def bf16_layout(t,block,device):
    return (_geometry is not None and requested_dtype()=='bfloat16'
            and validate_view(t,block,device,_geometry['block_bytes']))

def declared_block_bytes(cfg,shared):
    if len(cfg.kv_cache_groups)!=6:raise ValueError('Expected six retained cache groups')
    # Compute the same physical block extent from the declared per-layer specs,
    # then independently check every allocator record and actual storage view.
    specs={}
    totals=[]
    allocated={name for alloc in cfg.kv_cache_tensors for name in alloc.layers}
    for group in cfg.kv_cache_groups:
        spec=group.kv_cache_spec
        local={n:(spec.kv_cache_specs[n] if hasattr(spec,'kv_cache_specs') else spec)
               for n in group.layer_names if n in allocated}
        for n in group.layer_names:
            if n not in allocated and shared.get(n) not in allocated:
                raise ValueError('Unknown nonallocated cache layer')
        if set(local)&set(specs):raise ValueError('Layer occurs in multiple cache groups')
        specs.update(local);totals.append(sum(s.page_size_bytes for s in local.values()))
    block_bytes=max(totals)
    if block_bytes<128*1024 or block_bytes%512:raise ValueError('Unqualified BF16 arena extent')
    declared=set()
    for alloc in cfg.kv_cache_tensors:
        if alloc.block_stride!=block_bytes or alloc.size!=cfg.num_blocks*block_bytes:
            raise ValueError('Allocator is not the declared single block-interleaved arena')
        for i,name in enumerate(alloc.layers):
            if name not in specs or name in declared:raise ValueError('Allocator layer mapping differs')
            declared.add(name)
            offset=alloc.offset+i*alloc.layer_stride
            if not 0<=offset or offset+specs[name].page_size_bytes>block_bytes:
                raise ValueError('Layer page exceeds physical block')
    if declared!=set(specs):raise ValueError('Missing declared cache layers')
    return block_bytes

def configure(runner):
    global _geometry,_receipt
    if requested_dtype()!='bfloat16':raise ValueError('Not a BF16 boot')
    if runner.cache_config.cache_dtype!='bfloat16':raise ValueError('Engine cache dtype disagrees')
    cfg=runner.kv_cache_config
    block_bytes=declared_block_bytes(cfg,runner.shared_kv_cache_layers)
    # The shared arena also contains retained compressor state (including
    # FP32), not just latent/indexer caches. Match every view to its exact
    # declared spec and allocator offset rather than guessing from its dtype.
    from vllm.v1.kv_cache_interface import compute_layer_kv_cache_shape_bytes
    expected={};attention=set();state_dtypes=set()
    for alloc in cfg.kv_cache_tensors:
        for i,name in enumerate(alloc.layers):
            gid,group=next((j,g) for j,g in enumerate(cfg.kv_cache_groups) if name in g.layer_names)
            spec=group.kv_cache_spec
            if hasattr(spec,'kv_cache_specs'):spec=spec.kv_cache_specs[name]
            kernel_block=runner._kernel_block_sizes[gid]
            if getattr(spec,'storage_block_size',None) is not None:kernel_block=spec.storage_block_size
            shape=list(compute_layer_kv_cache_shape_bytes(spec,cfg.num_blocks,kernel_block))
            item=__import__('torch').empty(0,dtype=spec.dtype).element_size()
            if shape[-1]%item:raise ValueError('Spec content cannot be viewed in its declared dtype')
            shape[-1]//=item
            key=(alloc.offset+i*alloc.layer_stride,str(spec.dtype),tuple(shape))
            expected[key]=spec
            if getattr(spec,'cache_dtype_str',None)=='bfloat16':attention.add(key)
            else:state_dtypes.add(str(spec.dtype))
    caches=runner.kv_caches
    storages={t.untyped_storage().data_ptr():t.untyped_storage() for t in caches}
    if len(storages)!=1:raise ValueError('BF16 caches must share one physical arena')
    if next(iter(storages.values())).nbytes()!=cfg.num_blocks*block_bytes:
        raise ValueError('BF16 shared arena allocation extent differs')
    bf16_count=0;observed=set()
    for t in caches:
        if t.stride(0)*t.element_size()!=block_bytes:raise ValueError('Actual cache block stride differs')
        key=(t.storage_offset()*t.element_size(),str(t.dtype),tuple(t.shape))
        if key not in expected or t.stride(-1)!=1:raise ValueError('Actual arena view differs from its declared cache/state spec')
        observed.add(key)
        if key in attention:
            # The runner retains [B,H=1,N,C]; the layer binds its squeezed 3D view.
            if t.ndim!=4 or t.shape[1]!=1:raise ValueError('BF16 runner view must be [B,1,N,C]')
            latent=t.squeeze(1)
            if latent.shape[1] not in (32,64,128) or not validate_view(latent,latent.shape[1],runner.device,block_bytes):
                raise ValueError('Unqualified BF16 latent geometry')
            bf16_count+=1
    if observed!=set(expected):raise ValueError('A declared cache/state arena view is missing')
    if bf16_count<4:raise ValueError('Attention latent caches did not become BF16')
    geometry=dict(kv_cache_dtype='bfloat16',block_bytes=block_bytes,groups=6,
                  latent_row_bytes=1024,indexer_dtype='uint8',bf16_views=bf16_count,retained_state_dtypes=sorted(state_dtypes))
    if _geometry is not None and _geometry!=geometry:raise ValueError('Cache geometry changed after preparation')
    _geometry=geometry
    if _receipt is None:
        from .abc_bf16_check import qualify
        _receipt=qualify(runner.device,block_bytes)
    runner._abc_bf16_kv_receipt=dict(geometry=geometry,qualification=_receipt)
    return geometry

def probe_views(torch,device):
    if requested_dtype()=='bfloat16':
        if _geometry is None:raise ValueError('BF16 arena must be checked before capture probes')
        stride=_geometry['block_bytes']//2;width=512;dtype=torch.bfloat16
    else:stride=230400;width=584;dtype=torch.uint8
    raw=torch.zeros(2*stride,device=device,dtype=dtype)
    main=raw.as_strided((1,32,width),(stride,width,1),0)
    extras={b:raw.as_strided((1,b,width),(stride,width,1),stride) for b in (64,128)}
    return raw,main,extras
