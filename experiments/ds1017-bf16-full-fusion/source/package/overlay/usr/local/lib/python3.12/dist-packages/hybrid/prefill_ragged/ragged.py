"""Fused integer combine+ragged pack; standalone/default-off outer-bank candidate."""
import torch
from vllm.triton_utils import triton, tl

ALIGNMENT = 128


@triton.jit
def _primitive(v, RATIO: tl.constexpr, LOGICAL: tl.constexpr, WINDOW: tl.constexpr):
    # All products are int64. Integer cancellation retains exact prefixes even
    # when a short chunk attends a large history.
    if LOGICAL == 0:
        compressed = v * 0
    else:
        z = tl.minimum(v, RATIO * LOGICAL)
        q = z // RATIO
        rem = z % RATIO
        compressed = RATIO * q * (q - 1) // 2 + q * (rem + 1) + (v - z) * LOGICAL
    z_swa = tl.minimum(v, WINDOW)
    return compressed + z_swa * (z_swa + 1) // 2 + (v - z_swa) * WINDOW


@triton.jit(do_not_specialize=['M', 'N', 'LIMIT', 'ROWS'],
            do_not_specialize_on_alignment=['TOP', 'QSL', 'SEQ', 'GATHER', 'FLAT', 'INDPTR'])
def _ragged(TOP, QSL, SEQ, GATHER, FLAT, INDPTR, M, N, LIMIT, ROWS,
            TOP_STRIDE: tl.constexpr, REQUESTS: tl.constexpr,
            LOGICAL: tl.constexpr, WINDOW: tl.constexpr, RATIO: tl.constexpr,
            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, 8)
    base = tl.load(QSL).to(tl.int64)
    starts = tl.load(QSL + j, j < REQUESTS, other=0).to(tl.int64)
    ends = tl.load(QSL + j + 1, j < REQUESTS, other=0).to(tl.int64)
    sequences = tl.load(SEQ + j, j < REQUESTS, other=0).to(tl.int64)
    prefixes = sequences - (ends - starts)
    absolute = base + row
    req = tl.minimum(tl.sum(((j < REQUESTS) & (absolute >= ends)).to(tl.int32)), REQUESTS - 1)
    totals = _primitive(sequences, RATIO, LOGICAL, WINDOW) - _primitive(prefixes, RATIO, LOGICAL, WINDOW)
    preceding = tl.sum(tl.where((j < req) & (j < REQUESTS), totals, 0))
    start = tl.load(QSL + req).to(tl.int64)
    end = tl.load(QSL + req + 1).to(tl.int64)
    seq = tl.load(SEQ + req).to(tl.int64)
    gather = tl.load(GATHER + req).to(tl.int64)
    prefix = seq - (end - start)
    offset = absolute - start
    before = preceding + _primitive(prefix + offset, RATIO, LOGICAL, WINDOW) - _primitive(prefix, RATIO, LOGICAL, WINDOW)
    position = prefix + offset
    if LOGICAL == 0:
        count = tl.full((), 0, tl.int64)
    else:
        count = tl.maximum(0, tl.minimum((position + 1) // RATIO, LOGICAL))
    swa = tl.maximum(0, tl.minimum(position + 1, WINDOW))
    length = count + swa
    col = tl.arange(0, BLOCK)
    if LOGICAL == 0:
        value = tl.full((BLOCK,), -1, tl.int32)
    else:
        value = tl.load(TOP + row * TOP_STRIDE + col, col < count, other=-1).to(tl.int32)
    m64, n64 = M.to(tl.int64), N.to(tl.int64)
    valid_top = (col < count) & (value >= 0) & (value.to(tl.int64) < n64)
    top_value = value + (m64 * req.to(tl.int64)).to(tl.int32)
    s = col.to(tl.int64) - count
    swa_value = (m64 * req.to(tl.int64) + n64 + s + position - swa + 1 - (seq - gather)).to(tl.int32)
    value = tl.where(col < count, tl.where(valid_top, top_value, -1), swa_value)
    # Match the ORIGINAL pack kernel AFTER int32 offset wrapping. Invalid slots
    # remain present in NNZ/indptr; removing them would change attention order.
    value = tl.where((LIMIT < 0) | ((value >= 0) & (value.to(tl.int64) < LIMIT.to(tl.int64))), value, -1)
    tl.store(FLAT + before + col, value, col < length)
    tl.store(INDPTR + row, before.to(tl.int32))
    if row == ROWS - 1:
        tl.store(INDPTR + ROWS, (before + length).to(tl.int32))


def supported(topk_indices, query_start_loc, seq_lens, gather_lens,
              window_size, compress_ratio, topk, M, N, num_rows=-1):
    arrays = (topk_indices, query_start_loc, seq_lens, gather_lens)
    if any(not isinstance(t, torch.Tensor) for t in arrays): return False
    if not topk_indices.is_cuda or any(t.device != topk_indices.device for t in arrays): return False
    if torch.cuda.is_current_stream_capturing(): return False
    if any(t.dtype not in (torch.int32, torch.int64) for t in arrays): return False
    if topk_indices.ndim not in (2, 3) or not 1 <= topk_indices.shape[0] <= 2048: return False
    if any(t.ndim != 1 or not t.is_contiguous() for t in arrays[1:]): return False
    requests = seq_lens.numel()
    if not 1 <= requests <= 8 or gather_lens.numel() != requests or query_start_loc.numel() != requests + 1: return False
    if any(type(v) is not int for v in (window_size, compress_ratio, topk, M, N, num_rows)): return False
    width = ((topk + window_size + 127) // 128) * 128
    ratio_ok = compress_ratio in (1, 2, 4, 16, 128) or (compress_ratio == 0 and topk == 0 and N == 0)
    return (0 <= window_size <= 512 and ratio_ok
            and 0 <= topk <= 1024 and 0 <= N <= M <= 2**30
            and -1 <= num_rows <= 2**31-1 and 0 < width <= 2048)


def combine_ragged(topk_indices, query_start_loc, seq_lens, gather_lens,
                   window_size, compress_ratio, topk, M, N, num_rows=-1):
    """Valid metadata domain matches V5; returns full-capacity flat + indptr.

    No metadata cache; no host length readback. Original builder also leaves
    flat capacity beyond indptr[-1] uninitialized. Outer bank must admit only
    qualified/warmed serving signatures and retain original fallback.
    """
    if not supported(topk_indices, query_start_loc, seq_lens, gather_lens,
                     window_size, compress_ratio, topk, M, N, num_rows):
        raise ValueError('Unqualified fused ragged prefill shape')
    top = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    rows = top.shape[0]
    width = ((topk + window_size + 127) // 128) * 128
    logical = min(topk, top.shape[1])
    flat = torch.empty((rows * width,), dtype=torch.int32, device=top.device)
    ptr = torch.empty((rows + 1,), dtype=torch.int32, device=top.device)
    _ragged[(rows,)](top, query_start_loc, seq_lens, gather_lens, flat, ptr,
                     M, N, num_rows, rows, top.stride(0), seq_lens.numel(),
                     logical, window_size, compress_ratio, triton.next_power_of_2(width), num_warps=4)
    return flat, ptr
