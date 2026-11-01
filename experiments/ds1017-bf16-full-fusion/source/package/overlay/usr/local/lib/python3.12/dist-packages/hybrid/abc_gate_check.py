"""Accepted-prefix proof for GPUModelRunner's (all rows, last row per request) dummy ABI."""
def confidence_keep(conf):
    import torch
    if conf.ndim!=2 or conf.shape[1]!=5:raise ValueError('Confidence gate requires five draft probabilities')
    return (conf.cumprod(1)>=0.10).to(torch.int32).cumprod(1,dtype=torch.int32)

def prefix_exact(reference,candidate,keep,query_len=6):
    import torch
    if query_len!=6 or not isinstance(reference,tuple) or not isinstance(candidate,tuple) or len(reference)!=2 or len(candidate)!=2:
        raise ValueError('Unknown gated dummy output ABI')
    full,sampled=reference;other,other_sampled=candidate
    if (full.ndim!=2 or full.shape!=other.shape or full.shape[0]!=keep.numel()
        or full.shape[0]%query_len or keep.dtype!=torch.bool
        or sampled.shape!=full[query_len-1::query_len].shape or sampled.shape!=other_sampled.shape):
        raise ValueError('Unknown gated dummy output geometry')
    # _dummy_run returns hidden_states[cumsum(num_scheduled_tokens)-1].
    # Validate that map rather than silently dropping the per-request output.
    if not torch.equal(sampled,full[query_len-1::query_len]) or not torch.equal(other_sampled,other[query_len-1::query_len]):
        raise ValueError('Gated sampled-hidden row map differs')
    sampled_keep=keep[query_len-1::query_len]
    return bool(torch.equal(full[keep],other[keep]) and torch.equal(sampled[sampled_keep],other_sampled[sampled_keep]))
