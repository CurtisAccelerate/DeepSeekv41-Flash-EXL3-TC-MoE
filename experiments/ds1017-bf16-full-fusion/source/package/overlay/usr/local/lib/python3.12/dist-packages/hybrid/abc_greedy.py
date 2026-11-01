"""Exact greedy ID shortcut, guarded from penalties/logprobs/sampled requests."""
def logits(runner,h,so):
    from .abc_runtime import current,BANK
    p=current()
    if p is None or p.reference or getattr(so,'has_structured_output_requests',True):return None
    if getattr(runner,'_abc_teacher_plans',None):return None  # diagnostics need real target logits
    md=runner.input_batch.sampling_metadata
    if getattr(md,'logprob_token_ids',None) or getattr(runner,'num_prompt_logprobs',None):return None
    if not (md.all_greedy and md.max_num_logprobs is None and md.no_penalties and md.allowed_token_ids_mask is None and not md.bad_words_token_ids):return None
    for proc in md.logitsprocs.non_argmax_invariant:
        state=getattr(proc,'biases',None)
        if state is None:state=getattr(proc,'min_toks',None)
        if state is None or state:return None
    import torch
    lm=getattr(runner.model,'language_model',runner.model)
    lp,head=getattr(lm,'logits_processor',None),getattr(lm,'lm_head',None)
    if lp is None or head is None or getattr(head,'tp_size',1)<2 or not hasattr(lp,'get_top_tokens'):return None
    ids=lp.get_top_tokens(head,h).reshape(-1,1).to(torch.int64)
    out=torch.zeros(h.shape[0],lp.org_vocab_size,dtype=h.dtype,device=h.device)
    out.scatter_(1,ids,1.);out._abc_ids=ids.view(-1)
    BANK.hits['greedy_logits_ids']+=1
    return out

def verify(sampler,metadata,draft_probs,logits,sampling_metadata):
    ids=getattr(logits,'_abc_ids',None)
    if (ids is None or draft_probs is not None or not sampling_metadata.all_greedy or sampling_metadata.max_num_logprobs is not None or sampling_metadata.logprob_token_ids or sampler.synthetic_mode):return None
    import torch
    from vllm.v1.sample.rejection_sampler import rejection_greedy_sample_kernel,SamplerOutput
    b=len(metadata.num_draft_tokens)
    bonus=ids[metadata.bonus_logits_indices].to(torch.int32).unsqueeze(-1).contiguous()
    target=ids[metadata.target_logits_indices]
    out=torch.full((b,metadata.max_spec_len+1),-1,dtype=torch.int32,device=ids.device)
    rejection_greedy_sample_kernel[(b,)](out,metadata.cu_num_draft_tokens,metadata.draft_token_ids,target,bonus,None,metadata.max_spec_len,None,None,SYNTHETIC_MODE=False)
    return SamplerOutput(sampled_token_ids=out,logprobs_tensors=None)
