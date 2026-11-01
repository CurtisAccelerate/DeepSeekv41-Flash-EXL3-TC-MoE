"""Explicit startup-only parity check; caller passes torch and retained reference.

Nothing here initializes CUDA or runs qualification on import. Parent owns
all-rank admission, publication, stream/capture rules and default-off bank.
"""
import hashlib
import inspect
import time


def cases():
    items = []
    for rows in (1, 7, 65, 513, 1024, 2048):
        for requests in (1, 2, 8):
            k = len(items)
            items.append(dict(rows=rows, requests=requests, ratio=(1, 4, 16, 128)[k % 4],
                              window=128, topk=(512, 128, 7)[k % 3],
                              input_width=(512, 128, 6)[k % 3],
                              base=(0, 37, 1001)[k % 3],
                              dtype='int64' if k % 2 else 'int32',
                              metadata_dtype='int32' if k % 3 else 'int64',
                              layout=('2d', '3d', 'strided')[k % 3], lead_empty=k % 2 == 0))
    # Logical width clamp, no top-k, zero physical width, small/odd windows,
    # mixed metadata dtypes and int32 offset wrapping. All outputs <=5MiB.
    for ratio in (1, 4, 16, 128):
        for width, topk, window in ((0, 7, 1), (8, 0, 7), (6, 512, 129)):
            items.append(dict(rows=65, requests=8, ratio=ratio, window=window,
                              topk=topk, input_width=width, base=193,
                              dtype='int64', metadata_dtype='int32',
                              layout='3d' if width % 2 == 0 else '2d',
                              lead_empty=True, large_offset=ratio == 128))
    # Warm ACTUAL model input signatures, not merely semantic corner cases.
    # Rows do not specialize this kernel; requests and ratio currently do.
    # Top-k buffer is int32 in pinned model.py:598-603; ordinary metadata is
    # int32 too. M/N are dynamic/non-specializing but all real pairs are tested.
    for ratio in (1, 4, 16, 128):
        for requests in range(1, 9):
            items.append(dict(rows=65, requests=requests, ratio=ratio, window=128,
                              topk=512, input_width=512, base=0, dtype='int32',
                              metadata_dtype='int32', layout='2d', lead_empty=False,
                              production_signature=True))
    return items


def fixture(torch, spec, device='cpu'):
    rows, requests = spec['rows'], spec['requests']
    lengths = [0] * requests
    active = list(range(1 if spec['lead_empty'] and requests > 1 else 0, requests))
    for row in range(rows):
        lengths[active[row % len(active)]] += 1
    qsl = [spec['base']]
    for count in lengths:
        qsl.append(qsl[-1] + count)
    prefixes = (0, 1, 127, 128, 129, 16384, 130000, 0)
    seq = [q + prefixes[r] for r, q in enumerate(lengths)]
    gather = [q + min(prefixes[r], spec['window'] - 1) for r, q in enumerate(lengths)]
    N = (262144 + spec['ratio'] - 1) // spec['ratio']
    M = N + spec['window'] + 2048
    if spec.get('large_offset'):
        M, N = 2**30, 131072
    values = torch.tensor([-1, 0, 1, N - 1, N, N + 1, 2**32 + 3,
                           2**31 + 7, -2**32 + 3, -2**31 - 1], dtype=torch.int64)
    width = spec['input_width']
    ids = torch.arange(rows * width, dtype=torch.int64).reshape(rows, width)
    codes = values[ids % values.numel()].to(getattr(torch, spec['dtype']))
    if spec['layout'] == '3d':
        codes = codes.reshape(rows, 2 if width else 1, width // 2 if width else 0)
    elif spec['layout'] == 'strided':
        storage = torch.empty((rows, width * 2), dtype=codes.dtype)
        storage[:, ::2] = codes
        codes = storage[:, ::2]
    # Moving the backing storage before constructing the strided view preserves
    # its noncontiguous layout on CUDA too.
    if spec['layout'] == 'strided':
        codes = storage.to(device)[:, ::2]
    else:
        codes = codes.to(device)
    dtype = getattr(torch, spec['metadata_dtype'])
    qsl = torch.tensor(qsl, dtype=dtype, device=device)
    seq = torch.tensor(seq, dtype=dtype, device=device)
    # Exercise mixed integer metadata promotion as well.
    gather_dtype = torch.int64 if spec['base'] else dtype
    gather = torch.tensor(gather, dtype=gather_dtype, device=device)
    return (codes, qsl, seq, gather, spec['window'], spec['ratio'], spec['topk'], M, N)


def qualify(torch, reference, candidate, device, reserve_bytes=128 * 1024**2):
    """Require both int32 arrays bitwise equal on each explicit GPU fixture.

    Memory reserve is headroom, not a new allocation. A CUDA exception is never
    converted into an unsupported/fallback result. Caller classifies failures.
    """
    if not callable(reference) or not callable(candidate):
        raise TypeError('Retained reference and direct candidate must be passed')
    # Retained function identity, not a fabricated reference or bank wrapper.
    from .source_contract import check_reference
    source_contract = check_reference(reference)
    if torch.cuda.is_current_stream_capturing():
        raise ValueError('Startup qualification must be eager')
    started = time.monotonic()
    records = []
    initial = int(torch.cuda.mem_get_info(device)[0])
    minimum = initial
    for spec in cases():
        torch.cuda.synchronize(device)
        free = int(torch.cuda.mem_get_info(device)[0]); minimum = min(minimum, free)
        if free - reserve_bytes < 1024**3:
            raise ValueError('Combiner fixture cannot retain 1GiB floor plus 128MiB reserve')
        args = fixture(torch, spec, device)
        expected = reference(*args)
        actual = candidate(*args)
        torch.cuda.synchronize(device)
        shape = ((spec['rows'], ((spec['topk'] + spec['window'] + 127) // 128) * 128),
                 (spec['rows'],))
        equal = all(a.dtype == b.dtype == torch.int32 and tuple(a.shape) == tuple(b.shape) == s
                    and torch.equal(a, b) for a, b, s in zip(expected, actual, shape))
        record = dict(spec, bitwise_equal=bool(equal))
        records.append(record)
        if not equal:
            raise ValueError('Fused combiner parity failure: ' + str(record))
        if expected[0].numel() * expected[0].element_size() > 5 * 1024**2:
            raise ValueError('Fixture exceeds declared 5MiB output envelope')
        del args, expected, actual
    torch.cuda.synchronize(device)
    final = int(torch.cuda.mem_get_info(device)[0]); minimum = min(minimum, final)
    if final < 1024**3:
        raise ValueError('Combiner retained allocation violates 1GiB floor')
    return dict(passed=True, cases=len(records), records=records,
                initial_free=initial, final_free=final, min_sampled_free=minimum,
                seconds=time.monotonic() - started,
                reference_source_sha256=hashlib.sha256(inspect.getsource(reference).encode()).hexdigest(),
                source_contract=source_contract,
                model_quality_verified=False, performance_measured=False,
                stream=int(torch.cuda.current_stream(device).cuda_stream))
