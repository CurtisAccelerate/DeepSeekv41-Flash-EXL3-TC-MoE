"""Greedy one-target-decision control at the SAME k5 target graph geometry."""
def rejection(metadata,sampling_metadata):
    from . import abc_runtime as R
    profile=R.current()
    if profile is None or not profile.oracle:return metadata
    if not sampling_metadata.all_greedy:raise RuntimeError('Forced-rejection controls require all-greedy requests')
    if not metadata.draft_token_ids.numel():return metadata
    if any(n not in (0,5) for n in metadata.num_draft_tokens):raise RuntimeError('Oracle requires k5 verification geometry')
    import dataclasses,torch
    R.BANK.hits['oracle_forced_rejection:'+profile.name]+=1
    # Only the comparison IDs change. Actual target inputs, graph shape, target
    # logits, positions, KV and normal rejection cleanup remain the engine's.
    return dataclasses.replace(metadata,draft_token_ids=torch.full_like(metadata.draft_token_ids,-1))
