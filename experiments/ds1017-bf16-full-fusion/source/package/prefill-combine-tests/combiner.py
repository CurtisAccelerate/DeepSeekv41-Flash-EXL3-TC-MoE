"""Stateless candidate only. Parent bank owns default-off admission/fallback.

No metadata cache, monkey patch, environment selector, CUDA initialization or
server actions. Importing torch/Triton does not initialize CUDA.
"""
import torch
from vllm.triton_utils import triton, tl

ALIGNMENT = 128
MAX_ROWS = 2048
MAX_REQUESTS = 8
MAX_WIDTH = 2048


@triton.jit(do_not_specialize=['M', 'N'],do_not_specialize_on_alignment=['TOP','QSL','SEQ','GATHER','OUT','LENS'])
def _combine(TOP, QSL, SEQ, GATHER, OUT, LENS,
             TOP_STRIDE: tl.constexpr, REQUESTS: tl.constexpr,
             LOGICAL: tl.constexpr, WINDOW: tl.constexpr,
             RATIO: tl.constexpr, M, N,
             WIDTH: tl.constexpr, QBLOCK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    # Dynamic layout widths never specialize the kernel by current context or
    # a future packed-gather plan. Widen BEFORE multiply/add; wrap only where
    # the retained Torch function explicitly converts its result to int32.
    m64 = M.to(tl.int64)
    n64 = N.to(tl.int64)
    j = tl.arange(0, QBLOCK)
    base = tl.load(QSL).to(tl.int64)
    ends = tl.load(QSL + j + 1, j < REQUESTS, other=0).to(tl.int64)
    absolute = base + row
    # rightmost boundary, including empty requests, is Torch repeat_interleave's
    # request for this row. Clamp malformed input only to keep loads in bounds;
    # malformed metadata is outside the admitted/reference domain.
    req = tl.minimum(tl.sum(((j < REQUESTS) & (absolute >= ends)).to(tl.int32)), REQUESTS - 1)
    start = tl.load(QSL + req).to(tl.int64)
    end = tl.load(QSL + req + 1).to(tl.int64)
    seq = tl.load(SEQ + req).to(tl.int64)
    gather = tl.load(GATHER + req).to(tl.int64)
    position = seq - (end - start) + absolute - start
    count = tl.maximum(0, tl.minimum((position + 1) // RATIO, LOGICAL))
    swa = tl.maximum(0, tl.minimum(position + 1, WINDOW))
    col = tl.arange(0, BLOCK)
    # Torch casts int64 codes to int32 before comparing them with N.
    value = tl.load(TOP + row * TOP_STRIDE + col, col < LOGICAL, other=-1).to(tl.int32)
    top_valid = (col < count) & (value >= 0) & (value.to(tl.int64) < n64)
    top_value = value + (m64 * req.to(tl.int64)).to(tl.int32)
    s = col.to(tl.int64) - count
    swa_valid = (s >= 0) & (s < swa)
    swa_value = (m64 * req.to(tl.int64) + n64 + s + position - swa + 1 - (seq - gather)).to(tl.int32)
    result = tl.where(swa_valid, swa_value, tl.where(top_valid, top_value, -1))
    tl.store(OUT + row * WIDTH + col, result, col < WIDTH)
    tl.store(LENS + row, (count + swa).to(tl.int32))


def supported(topk_indices, query_start_loc, seq_lens, gather_lens,
              window_size, compress_ratio, topk, M, N):
    """Structural admission only; caller owns valid immutable request metadata.

    Metadata domain: nondecreasing QSL, QSL[-1]-QSL[0]==rows, 0<=query<=seq,
    0<=gather<=seq<=262144; integer arithmetic cannot overflow int64. Empty
    requests are valid. Capture, other devices/layouts fall back in outer bank.
    """
    arrays = (topk_indices, query_start_loc, seq_lens, gather_lens)
    if any(not isinstance(t, torch.Tensor) for t in arrays):
        return False
    if not topk_indices.is_cuda or any(t.device != topk_indices.device for t in arrays):
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    if any(t.dtype not in (torch.int32, torch.int64) for t in arrays):
        return False
    if topk_indices.ndim not in (2, 3) or not 1 <= topk_indices.shape[0] <= MAX_ROWS:
        return False
    if any(t.ndim != 1 or not t.is_contiguous() for t in arrays[1:]):
        return False
    requests = seq_lens.numel()
    if not 1 <= requests <= MAX_REQUESTS or gather_lens.numel() != requests or query_start_loc.numel() != requests + 1:
        return False
    if any(type(v) is not int for v in (window_size, compress_ratio, topk, M, N)):
        return False
    width = ((topk + window_size + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    return (0 <= window_size <= 512 and compress_ratio in (1, 4, 16, 128)
            and 0 <= topk <= 1024 and 0 <= N <= M <= 2**30
            and 0 < width <= MAX_WIDTH)


def combine_topk_swa_indices(topk_indices, query_start_loc, seq_lens, gather_lens,
                             window_size, compress_ratio, topk, M, N):
    """Same nine positional arguments and two int32 outputs as incumbent.

    Call supported first in the bank; unsupported calls keep incumbent. This
    direct entry raises on unsupported shape rather than silently changing path.
    """
    if not supported(topk_indices, query_start_loc, seq_lens, gather_lens,
                     window_size, compress_ratio, topk, M, N):
        raise ValueError('Unqualified fused prefill combiner shape')
    flat = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    rows = flat.shape[0]
    logical = min(topk, flat.shape[1])
    width = ((topk + window_size + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    out = torch.empty((rows, width), dtype=torch.int32, device=flat.device)
    lens = torch.empty((rows,), dtype=torch.int32, device=flat.device)
    _combine[(rows,)](flat, query_start_loc, seq_lens, gather_lens, out, lens,
                      flat.stride(0), seq_lens.numel(), logical, window_size,
                      compress_ratio, M, N, width,
                      triton.next_power_of_2(seq_lens.numel()),
                      triton.next_power_of_2(width), num_warps=4)
    return out, lens
