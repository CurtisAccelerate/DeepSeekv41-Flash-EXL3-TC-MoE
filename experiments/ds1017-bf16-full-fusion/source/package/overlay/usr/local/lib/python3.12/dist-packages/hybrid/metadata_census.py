"""Read-only source-level state census. No tensor values, CUDA ops, or mutations.
Staged for future initial diagnostics; not installed into current owner.
"""
def tensor_metadata(value):
    if value is None:return None
    out={'class':type(value).__name__}
    for name in ('shape','dtype','device'):
        try:out[name]=str(getattr(value,name))
        except Exception:pass
    for name in ('stride','data_ptr','numel','element_size','is_contiguous'):
        try:out[name]=getattr(value,name)()
        except Exception:pass
    return out

from .generic_metadata import direct_metadata

def inspect_initial_metadata(runner):
    result={'scope':'generic direct tensor/paired-buffer and safe scalar metadata; no payload, weight/config recursion, or state mutation','errors':[],'builders':[],'module_counts':{},'predicates':{}}
    def value(label,fn):
        try:return fn()
        except Exception as exc:result['errors'].append({'field':label,'error':repr(exc)});return None
    result['direct_runner_metadata']=value('direct_runner_metadata',lambda:direct_metadata(runner))
    p=value('parallel_config',lambda:runner.parallel_config)
    if p is not None:result['parallel']={k:value(k,lambda k=k:getattr(p,k)) for k in ('tensor_parallel_size','pipeline_parallel_size','data_parallel_size','decode_context_parallel_size','prefill_context_parallel_size','use_ubatching')}
    result['runner']={k:value(k,lambda k=k:getattr(runner,k)) for k in ('max_num_tokens','max_model_len','routed_experts_initialized','_w26_serving_seen','supports_mm_inputs','is_multimodal_raw_input_only_model','enable_prompt_embeds','uses_mrope','uses_xdrope_dim')}
    result['is_encoder_decoder']=value('is_encoder_decoder',lambda:runner.model_config.is_encoder_decoder)
    result['initial_counts']={'requests':value('requests',lambda:len(runner.requests)),'num_reqs':value('num_reqs',lambda:runner.input_batch.num_reqs)}
    for k in ('optimistic_seq_lens_cpu','seq_lens','positions','_dsv41_pad_buf','_dsv41_pad_cpu'):
        result.setdefault('runner_buffers',{})[k]=tensor_metadata(value(k,lambda k=k:getattr(runner,k)))
    for k in ('query_start_loc','query_pos','lookback_token_ids','input_ids','inputs_embeds','num_decode_draft_tokens'):
        b=value(k,lambda k=k:getattr(runner,k));result.setdefault('paired_buffers',{})[k]={side:tensor_metadata(value(k+'.'+side,lambda side=side:getattr(b,side))) for side in ('cpu','gpu')}
    result['caches']=[tensor_metadata(t) for t in value('kv_caches',lambda:runner.kv_caches) or []]
    result['num_cache_blocks']=value('num_cache_blocks',lambda:runner.kv_cache_config.num_blocks)
    tables=value('block_tables',lambda:runner.input_batch.block_table.block_tables)
    result['block_tables']=[]
    for i,b in enumerate(tables or []):
        result['block_tables'].append({'index':i,'cpu':tensor_metadata(value('tablecpu',lambda:b.block_table.cpu)),'gpu':tensor_metadata(value('tablegpu',lambda:b.block_table.gpu)),'slot_mapping':tensor_metadata(value('slot',lambda:b.slot_mapping.gpu))})
    seen=set()
    for ri,groups in enumerate(value('attn_groups',lambda:runner.attn_groups) or []):
        for ci,g in enumerate(groups):
            b=value('builder',lambda:g.get_metadata_builder(0))
            if b is None or id(b) in seen:continue
            seen.add(id(b));row={'row':ri,'column':ci,'class':type(b).__name__,'layer_names':value('layer_names',lambda:list(g.layer_names))}
            row['direct_attributes']=value('builder_direct_metadata',lambda:direct_metadata(b))
            row['cache_spec_attributes']=value('cache_spec_metadata',lambda:direct_metadata(b.kv_cache_spec))
            row['kernel_block_size']=value('kernel_block_size',lambda:b.kernel_block_size)
            for k in ('max_image_tokens','decode_threshold','num_speculative_tokens','window_size','prefill_index_width','noncausal_index_width','compress_ratio','block_size','is_dspark'):
                if hasattr(b,k):row[k]=value(k,lambda k=k:getattr(b,k))
            names=('token_to_req_indices','decode_swa_indices','decode_swa_lens','is_valid_token','decode_swa_ragged_indices_buffer','decode_swa_ragged_indptr_buffer','prefill_swa_indices','prefill_swa_lens','left_visible','right_visible','span_indptr','span_starts','span_ends','decode_swa_indices_noncausal','req_id_per_token_buffer','compressed_slot_mapping_buffer','slot_mapping_buffer')
            row['buffers']={k:tensor_metadata(value(k,lambda k=k:getattr(b,k))) for k in names if hasattr(b,k)};result['builders'].append(row)
    model=value('get_model',lambda:runner.get_model());counts={'engram_layers':0,'hash_states':0,'mtp':0};result['special_modules']=[]
    if model is not None:
        for name,m in model.named_modules():
            kind=type(m).__name__;result['module_counts'][kind]=result['module_counts'].get(kind,0)+1
            fields=[k for k in ('_mtp_hidden_buffer','topk_indices_buffer','candidate_block_buffer','staged_rows') if getattr(m,k,None) is not None]
            hash_binding=None
            if kind=='NgramHashState':
                fields+=['_cache'];counts['hash_states']+=1
                hash_binding=value(name+'.hash_binding',lambda:m._cache is not None and m._kv_cache_ref is not None and m._kv_cache_ref() is m.swa_cache_module.kv_cache)
            counts['mtp']+=int('_mtp_hidden_buffer' in fields);counts['engram_layers']+=int('staged_rows' in fields)
            if fields:result['special_modules'].append({'name':name,'class':kind,'buffers':{k:tensor_metadata(getattr(m,k)) for k in fields},'embed_dp_size':getattr(getattr(m,'embed_tokens',None),'dp_size',None),'hash_cache_bound_to_current_kv':hash_binding,'direct_attributes':direct_metadata(m)})
    result['expected_counts_actual']=counts
    result['predicates']['capacity']=result['runner'].get('max_num_tokens')==2048 and result['runner'].get('max_model_len')==262144
    result['predicates']['six_cache_groups']=len(result['block_tables'])==6
    result['predicates']['initial_empty_cpu_counts']=result['initial_counts']=={'requests':0,'num_reqs':0}
    result['predicates']['never_served']=result['runner'].get('_w26_serving_seen') is False
    result['predicates']['engram_hash_mtp_counts']=counts=={'engram_layers':2,'hash_states':1,'mtp':1}
    result['not_verified']=['device block-table contents','exact storage alias extent','host collective forbidden ranges','dummy state writes/restoration','actual per-builder causal decode metadata','private capture/native/profile/numerical replay']
    return result
