"""Pure integer closed forms and row algebra; no torch/CUDA imports."""
def i32(x):
    return (x + 2**31) % 2**32 - 2**31


def floor_prefix(v, ratio, cap):
    """sum(min(u//ratio,cap) for u in range(1,v+1)), v>=0."""
    if cap == 0:
        return 0
    z = min(v, ratio * cap)
    q, rem = divmod(z, ratio)
    return ratio * q * (q - 1) // 2 + q * (rem + 1) + (v - z) * cap


def linear_prefix(v, cap):
    """sum(min(u,cap) for u in range(1,v+1)), v>=0."""
    z = min(v, cap)
    return z * (z + 1) // 2 + (v - z) * cap


def combined_prefix(v, ratio, logical, window):
    return floor_prefix(v, ratio, logical) + linear_prefix(v, window)


def fused(codes, qsl, seq, gather, window, ratio, topk, M, N, num_rows=-1):
    flat, ptr = [], []
    for row, code in enumerate(codes):
        absolute = qsl[0] + row
        req = min(sum(absolute >= end for end in qsl[1:]), len(seq) - 1)
        logical = min(topk, len(code))
        primitive = lambda v: combined_prefix(v, ratio, logical, window)
        starts = [s - (b - a) for s, a, b in zip(seq, qsl, qsl[1:])]
        before = sum(primitive(seq[r]) - primitive(starts[r]) for r in range(req))
        offset = absolute - qsl[req]
        before += primitive(starts[req] + offset) - primitive(starts[req])
        assert before == len(flat)
        ptr.append(before)
        position = starts[req] + offset
        count = 0 if logical == 0 else max(0, min((position + 1) // ratio, logical))
        swa = max(0, min(position + 1, window))
        for col in range(count + swa):
            if col < count:
                v = i32(code[col])
                v = i32(v + i32(M * req)) if 0 <= v < N else -1
            else:
                v = i32(M * req + N + col - count + position - swa + 1 - (seq[req] - gather[req]))
            if num_rows >= 0 and not 0 <= v < num_rows:
                v = -1
            flat.append(v)
    ptr.append(len(flat))
    return flat, ptr
