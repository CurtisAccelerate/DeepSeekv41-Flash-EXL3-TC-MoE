"""Pure Python incumbent oracle and independent fused row algebra; no imports."""
def i32(x):
    return (x + 2**31) % 2**32 - 2**31


def reference(codes, qsl, seq, gather, window, ratio, topk, M, N):
    width = (topk + window + 127) // 128 * 128
    result, lens = [], []
    for req, (lo, hi) in enumerate(zip(qsl, qsl[1:])):
        for offset in range(hi - lo):
            row = lo - qsl[0] + offset
            position = seq[req] - (hi - lo) + offset
            logical = min(topk, len(codes[row]))
            count = max(0, min((position + 1) // ratio, logical))
            swa = max(0, min(position + 1, window))
            output = [-1] * width
            for col in range(logical):
                v = i32(codes[row][col])
                if col < count and 0 <= v < N:
                    output[col] = i32(v + i32(M * req))
            for col in range(swa):
                output[count + col] = i32(M * req + N + col + position - swa + 1 - (seq[req] - gather[req]))
            result.append(output)
            lens.append(count + swa)
    return result, lens


def fused(codes, qsl, seq, gather, window, ratio, topk, M, N):
    width = (topk + window + 127) // 128 * 128
    result, lens = [], []
    for row, code in enumerate(codes):
        absolute = qsl[0] + row
        req = min(sum(absolute >= end for end in qsl[1:]), len(seq) - 1)
        position = seq[req] - (qsl[req + 1] - qsl[req]) + absolute - qsl[req]
        logical = min(topk, len(code))
        count = max(0, min((position + 1) // ratio, logical))
        swa = max(0, min(position + 1, window))
        output = []
        for col in range(width):
            v = i32(code[col]) if col < logical else -1
            value = i32(v + i32(M * req)) if col < count and 0 <= v < N else -1
            s = col - count
            if 0 <= s < swa:
                value = i32(M * req + N + s + position - swa + 1 - (seq[req] - gather[req]))
            output.append(value)
        result.append(output)
        lens.append(count + swa)
    return result, lens
