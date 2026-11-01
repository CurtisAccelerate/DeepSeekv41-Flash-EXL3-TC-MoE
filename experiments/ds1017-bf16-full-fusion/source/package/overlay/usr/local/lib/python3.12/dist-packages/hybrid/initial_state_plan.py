"""Exact initial-only state view collector for the pinned T6 target dummy path.

SOURCE STAGED. Not called by current owner; not yet GPU/model qualified. This
collects explicit semantic/input metadata ranges, never model weights/full KV.
Unknown topology/builders fail rather than assuming null-slot safety.
"""
from .state_journal import require_initial_empty_state, ByteJournal, AttributeJournal, BindingJournal

PLAN_GPU_QUALIFIED=False

def collect(runner,*,phase,rows=6,reqs=1):
    if reqs<1 or rows!=6*reqs:raise ValueError('Journal covers exactly 6 rows per request')
    import torch
    from w9_exl_ab import collective_profiles as CP
    if getattr(runner,'_w26_serving_seen',True):raise ValueError('Startup-only guard missing or serving already began')
    if runner.is_multimodal_raw_input_only_model or runner.model_config.is_encoder_decoder or runner.enable_prompt_embeds or runner.uses_mrope or runner.uses_xdrope_dim:
        raise ValueError('Unqualified raw multimodal/encoder/prompt/position dummy branch')
    if not runner.supports_mm_inputs or getattr(runner.get_model(),'requires_raw_input_tokens',None) is not True:
        raise ValueError('Expected actual VL raw-token plus embedding input contract')
    p=runner.parallel_config
    if any(getattr(p,k,1)!=1 for k in ('pipeline_parallel_size','data_parallel_size','decode_context_parallel_size','prefill_context_parallel_size')) or p.use_ubatching:
        raise ValueError('Initial journal only supports TP4/PP1/DP1/CP1 without microbatches')
    if p.tensor_parallel_size!=4 or runner.max_num_tokens!=2048 or runner.max_model_len!=262144:
        raise ValueError('Pinned model capacity/topology differs')
    if runner.routed_experts_initialized:raise ValueError('Route capture state is outside this journal contract')
    tables=runner.input_batch.block_table.block_tables
    if len(tables)!=6:raise ValueError('Expected six retained cache groups')
    require_initial_empty_state(phase=phase,request_count=runner.input_batch.num_reqs,request_map_size=len(runner.requests),
        cpu_block_ids=[sum((b.block_table.cpu[r].tolist() for r in range(reqs)),[]) for b in tables],
        device_block_ids=[sum((b.block_table.gpu[r].tolist() for r in range(reqs)),[]) for b in tables])
    gpu=[];cpu=[];aliases=[];bindings=[]
    def bind(obj,field):
        value=getattr(obj,field)
        bindings.append((field,lambda obj=obj,field=field:getattr(obj,field),value))
        return value
    for field in ('input_batch','requests','kv_caches','attn_groups','optimistic_seq_lens_cpu','seq_lens','query_start_loc','query_pos','lookback_token_ids','positions','input_ids','_dsv41_pad_buf','_dsv41_pad_cpu'):
        bind(runner,field)
    bind(runner.input_batch,'block_table');bind(runner.input_batch.block_table,'block_tables')
    bind(runner.input_ids,'gpu')
    def add(name,tensor):
        if not isinstance(tensor,torch.Tensor) or not tensor.is_contiguous() or tensor.numel()==0:
            raise ValueError('Missing/noncontiguous initialized state: '+name)
        (gpu if tensor.is_cuda else cpu).append((name,tensor))
    def pair(name,buffer,part=None):
        bind(buffer,'cpu');bind(buffer,'gpu')
        add(name+'.cpu',buffer.cpu if part is None else buffer.cpu[part])
        add(name+'.gpu',buffer.gpu if part is None else buffer.gpu[part])
    # Every cache group shares one block-interleaved arena. Integer null-block
    # ownership is proved above; preserve ALL aliasing layer pages in block0.
    caches=runner.kv_caches
    from .abc_kv_contract import requested_dtype, configure
    page_bytes=230400
    if requested_dtype()=='bfloat16':
        page_bytes=configure(runner)['block_bytes']
    for i,tensor in enumerate(caches):
        bindings.append((f'kv_caches[{i}]',lambda i=i:runner.kv_caches[i],tensor))
    storages={t.untyped_storage().data_ptr():t.untyped_storage() for t in caches}
    if len(storages)!=1 or any(t.stride(0)*t.element_size()!=page_bytes for t in caches):
        raise ValueError('KV arena stride/alias topology differs')
    storage=next(iter(storages.values()))
    if storage.nbytes()!=runner.kv_cache_config.num_blocks*page_bytes:
        raise ValueError('KV storage extent differs from exact shared arena')
    null=torch.empty(0,dtype=torch.uint8,device=runner.device).set_(storage,0,(page_bytes,),(1,))
    add('kv.shared_null_block0',null)
    for i,b in enumerate(tables):
        bindings.append((f'block_table[{i}]',lambda i=i:runner.input_batch.block_table.block_tables[i],b))
        bind(b,'block_table');bind(b,'slot_mapping');bind(b.slot_mapping,'gpu')
        pair(f'block_table.{i}',b.block_table,slice(0,reqs))
        add(f'slot_mapping.{i}',b.slot_mapping.gpu[:rows])
    add('optimistic_seq_lens_cpu',runner.optimistic_seq_lens_cpu)
    add('seq_lens',runner.seq_lens)
    pair('query_start_loc',runner.query_start_loc)
    pair('query_pos',runner.query_pos,slice(0,rows))
    pair('lookback_token_ids',runner.lookback_token_ids)
    bind(runner,'num_decode_draft_tokens')
    pair('num_decode_draft_tokens',runner.num_decode_draft_tokens) # small conservative input-state identity/bytes guard
    # These buffers should not change in DCP1/nonrandom target dummy execution;
    # retain them as explicit pointer/input equality checks anyway.
    add('positions',runner.positions[:rows]);add('input_ids',runner.input_ids.gpu[:rows])
    bind(runner,'inputs_embeds');bind(runner.inputs_embeds,'gpu')
    add('inputs_embeds',runner.inputs_embeds.gpu[:rows]) # actual VL dummy input, including text-only decode
    for field in ('_dsv41_engram_prefix','_dsv41_engram_gid','_dsv41_pad_buf','_dsv41_pad_cpu'):
        if field not in runner.__dict__:raise ValueError('Baseline input initialization incomplete: '+field)
    add('padding',runner._dsv41_pad_buf[:rows]);add('padding.cpu',runner._dsv41_pad_cpu[:rows])
    seen=set();builder_names=[]
    for row,groups in enumerate(runner.attn_groups):
        bindings.append((f'attn_groups[{row}]',lambda row=row:runner.attn_groups[row],groups))
        for column,group in enumerate(groups):
            bindings.append((f'attn_groups[{row}][{column}]',lambda row=row,column=column:runner.attn_groups[row][column],group))
            b=group.get_metadata_builder(0)
            if id(b) in seen:continue
            seen.add(id(b));kind=type(b).__name__;prefix=f'builder{len(seen)}.{kind}';builder_names.append(kind)
            bindings.append((prefix,lambda group=group:group.get_metadata_builder(0),b))
            if kind=='DeepseekV4ROCMAiterSparseSWAMetadataBuilder':
                if (b.max_image_tokens,b.window_size,b.prefill_index_width,b.noncausal_index_width)!=(1024,128,1152,192) or b.decode_threshold<6:
                    raise ValueError('Unqualified SWA capability/decode geometry')
                # Capability is allowed ONLY with the separately enforced actual
                # initial causal T6 metadata contract. Preserve small vision state;
                # larger prefill arrays are bound and proved unreachable by source
                # and the exact-constructor fixture, not silently assumed inactive.
                for name in ('left_visible','right_visible','span_indptr','span_starts','span_ends'):
                    add(prefix+'.'+name,bind(b,name))
                for name in ('prefill_swa_indices','prefill_swa_lens','decode_swa_indices_noncausal'):
                    bind(b,name)
                if b.decode_swa_indices_noncausal is not None:
                    raise ValueError('Target builder unexpectedly initialized noncausal storage')
                for name in ('token_to_req_indices','decode_swa_indices','is_valid_token'):
                    add(prefix+'.'+name,bind(b,name)[:rows])
                for field in ('decode_swa_lens','decode_swa_ragged_indices_buffer','decode_swa_ragged_indptr_buffer'):bind(b,field)
                add(prefix+'.decode_swa_lens',b.decode_swa_lens) # tail is zeroed!
                add(prefix+'.ragged_indices',b.decode_swa_ragged_indices_buffer[:rows*b.window_size])
                add(prefix+'.ragged_indptr',b.decode_swa_ragged_indptr_buffer[:rows+1])
            elif kind=='DeepseekV4SparseMLAMetadataBuilder':
                bind(b,'req_id_per_token_buffer')
                add(prefix+'.req_id_per_token_buffer',b.req_id_per_token_buffer[:rows])
                if b.compress_ratio==2:add(prefix+'.compressed_slot_mapping_buffer',bind(b,'compressed_slot_mapping_buffer')[:rows])
                elif b.compress_ratio!=1:raise ValueError('Unexpected compression ratio')
            elif kind=='DeepseekV32IndexerMetadataBuilder':
                # Exact V4 backend on SM80: logical block128, states64/128,
                # kernel block128. The SM90-only reblocking/lazy allocation
                # branch is excluded before any private dummy executes.
                if (b.decode_threshold,b.num_speculative_tokens,b.dcp_world_size,b.pcp_world_size,b.kernel_block_size,b.kv_cache_spec.block_size)!=(6,5,1,1,128,128):
                    raise ValueError('Unqualified indexer classifier/CP/kernel geometry')
                if b.compress_ratio not in (1,2) or b.supports_varlen or not b.use_flattening:
                    raise ValueError('Unqualified indexer flattening/compression')
                if b.expanded_block_table_buffer.shape!=(2048,2048):
                    raise ValueError('Indexer table capacity differs from actual census/source')
                for name in ('offsets_buffer','decode_lens_buffer','per_req_decode_lens_buffer','decode_seq_lens_buffer','global_decode_seq_lens_buffer','decode_indices_buffer','arange_buffer','scheduler_metadata_buffer'):
                    add(prefix+'.'+name,bind(b,name))
                add(prefix+'.expanded_block_table_buffer',bind(b,'expanded_block_table_buffer')[:rows])
                if b.compress_ratio==2:
                    add(prefix+'.compressed_slot_mapping_buffer',bind(b,'compressed_slot_mapping_buffer')[:rows])
                    add(prefix+'.expanded_seq_lens_buffer',bind(b,'expanded_seq_lens_buffer'))
                optional=bind(b,'indexer_decode_block_table_buffer')
                if optional is not None:add(prefix+'.indexer_decode_block_table_buffer',optional[:rows])
            elif kind=='CompressorMetadataBuilder':
                for name in ('token_to_req_indices','slot_mapping_buffer'):add(prefix+'.'+name,bind(b,name)[:rows])
            else:raise ValueError('Unknown metadata builder: '+kind)
    from collections import Counter
    if Counter(builder_names)!=Counter({'DeepseekV4ROCMAiterSparseSWAMetadataBuilder':4,'DeepseekV4SparseMLAMetadataBuilder':2,'DeepseekV32IndexerMetadataBuilder':2,'CompressorMetadataBuilder':1}):
        raise ValueError('Builder census differs from actual loaded nine-builder topology')
    model=runner.get_model();engram_layers=0;hash_states=0;mtp=0
    bindings.append(('runner.get_model()',lambda:runner.get_model(),model))
    for name,m in model.named_modules():
        bindings.append(('module:'+name,lambda name=name:runner.get_model().get_submodule(name),m))
        if hasattr(m,'_dsv41_idx'):aliases.append((m,'_dsv41_shipped_x'))
        for field in ('_mtp_hidden_buffer','topk_indices_buffer','candidate_block_buffer'):
            value=getattr(m,field,None)
            if value is not None:
                bind(m,field)
                add(name+'.'+field,value[:rows]);mtp+=int(field=='_mtp_hidden_buffer')
        if type(m).__name__=='NgramHashState':
            if m._cache is None or m._kv_cache_ref is None or m._kv_cache_ref() is not m.swa_cache_module.kv_cache:
                raise ValueError('Engram hash-cache binding would allocate/rebind')
            add(name+'.hash_null_block',m._cache[:m.block_size]);hash_states+=1
            bind(m,'_cache');bind(m,'_kv_cache_ref');bind(m,'swa_cache_module');bind(m.swa_cache_module,'kv_cache')
        if hasattr(m,'staged_rows'):
            if getattr(m.embed_tokens,'dp_size',None)!=1:raise ValueError('Engram DP staging differs')
            add(name+'.staged_rows',m.staged_rows[:rows]);engram_layers+=1
            bind(m,'staged_rows')
    if (engram_layers,hash_states,mtp)!=(2,1,1):raise ValueError('Missing target Engram/MTP state census')
    excluded=[]
    for reg in CP.REG:
        owner=reg['owner'];seq=owner.seq
        excluded.append((str(seq.device),seq.data_ptr(),seq.data_ptr()+seq.numel()*seq.element_size()))
        excluded.append((str(seq.device),owner.dev_base,owner.dev_base+owner.seg_bytes))
    if not excluded:raise ValueError('Missing explicit host collective protocol exclusion')
    return dict(gpu=gpu,cpu=cpu,aliases=aliases,bindings=bindings,builders=builder_names,qualified=False,forbidden_device_ranges=excluded,
        exclusions='Communication tickets/host arena/PyNccl/events/registries/ownership are NOT restored; legitimate collective sequence advances remain monotonic',
        additional_gate='scratch/workspace nonsemantic proof and generated/model restore test still required')

class InitialJournal:
    def __init__(self,plan,budget_bytes=16*1024**2):
        import torch
        self.torch=torch
        self.gpu=ByteJournal(plan['gpu'],budget_bytes=budget_bytes,forbidden_device_ranges=plan['forbidden_device_ranges'])
        self.cpu=[(v,v.clone()) for _,v in plan['cpu']]
        self.attrs=AttributeJournal(plan['aliases'])
        self.bindings=BindingJournal(plan['bindings'])
        self.pointers=[(v,v.data_ptr(),tuple(v.shape),tuple(v.stride())) for _,v in plan['gpu']+plan['cpu']]
    def restore(self):
        self.torch.cuda.synchronize() # no outstanding consumer or H2D DMA
        self.gpu.restore()
        for view,copy in self.cpu:view.copy_(copy)
        self.attrs.restore();self.torch.cuda.synchronize()
    def verify(self):
        bindings_ok=self.bindings.verify()
        return self.gpu.verify() and self.attrs.verify() and bindings_ok and all(self.torch.equal(v,c) for v,c in self.cpu) and all(v.data_ptr()==ptr and tuple(v.shape)==shape and tuple(v.stride())==stride for v,ptr,shape,stride in self.pointers)
