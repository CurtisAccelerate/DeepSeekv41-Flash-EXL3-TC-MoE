"""Actual-weight bounded numerical checks and exact-M timings; no torch import."""
import gc
import statistics
import time
from .dense import ROWS


def guard(torch, device, reserve=128 * 1024**2):
    free = int(torch.cuda.mem_get_info(device)[0])
    if free - reserve < 1024**3:
        raise ValueError('Dense fixture cannot retain 1GiB floor plus 128MiB reserve')
    return free


def within_bound(torch, actual, ref, absolute_sum, K):
    eps32 = torch.finfo(torch.float32).eps
    gamma = (2 * K + 64) * eps32 / (1 - (2 * K + 64) * eps32)
    bound = torch.finfo(torch.bfloat16).eps * ref.abs() + gamma * absolute_sum + torch.finfo(torch.bfloat16).tiny
    fraction = (actual.double() - ref).abs() / bound
    if not bool((fraction <= 1).all()):
        raise ValueError('Dense operator exceeded existing FP32/BF16 forward-error bound')
    return float(fraction.max())


def time_pair(torch, runner, b, x):
    calls = {'marlin': lambda: runner.reference(b['layer'], x),
             'cublas_dequant_mm': lambda: runner._compute(b, x)}
    for call in calls.values():
        output = call(); del output
    torch.cuda.synchronize(runner.device)
    values = {key: [] for key in calls}
    for rep in range(3):
        for key in (list(calls) if rep % 2 == 0 else list(reversed(calls))):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record(); output = calls[key](); end.record()
            end.synchronize()
            values[key].append(float(begin.elapsed_time(end)))
            del output
    median = {key: statistics.median(value) for key, value in values.items()}
    wins = median['cublas_dequant_mm'] <= median['marlin'] * .98
    return dict(samples_ms=values, median_ms=median, tuned_candidate=bool(wins),
                required_speedup_percent=2, timing='CUDA events; dequant+mm included; warm1/reps3')


def fixture(torch, runner, b, rows, kind):
    """Clear all fixture tensor references even when numerical checks reject."""
    N, K = b['N'], b['K']
    x = baseline = actual = r = cols = sample_base = sample_actual = xx = ww = ref = absolute_sum = None
    try:
        gen = torch.Generator(device=runner.device).manual_seed(1017)
        x = torch.randn((rows, K), dtype=torch.bfloat16, device=runner.device, generator=gen) * .125
        if kind == 'alternating':
            x[:, 1::2] = -x[:, ::2]
        baseline = runner.reference(b['layer'], x)
        actual = runner._compute(b, x)
        torch.cuda.synchronize(runner.device)
        if not bool(torch.isfinite(baseline).all() and torch.isfinite(actual).all()):
            raise ValueError('Nonfinite full dense qualification output')
        r = torch.tensor([0, rows // 3, 2 * rows // 3, rows - 1], dtype=torch.long, device=runner.device)
        cols = torch.cat((torch.arange(min(32, N), device=runner.device),
                          torch.arange(max(32, N - 32), N, device=runner.device)))
        sample_base = baseline.index_select(0, r).index_select(1, cols)
        sample_actual = actual.index_select(0, r).index_select(1, cols)
        # Keep only 4x64 BF16 samples before any FP64 allocations.
        baseline = actual = None
        xx = x.index_select(0, r).double()
        ww = b['wb'].index_select(0, cols).double()
        ref = xx @ ww.t()
        absolute_sum = xx.abs() @ ww.abs().t()
        record = dict(N=N, K=K, rows=rows, input=kind, sample_rows=4, sample_columns=len(cols),
                      marlin_max_bound_fraction=within_bound(torch, sample_base, ref, absolute_sum, K),
                      cublas_max_bound_fraction=within_bound(torch, sample_actual, ref, absolute_sum, K),
                      full_outputs_finite=True, source_weight_bits_exact=True,
                      reference='FP64 dot of source-bit-exact BF16 weight; existing gamma bound')
        free = int(torch.cuda.mem_get_info(runner.device)[0])
        if free < 1024**3:
            raise ValueError('Dense numerical fixture crossed 1GiB free floor')
        sample_base = sample_actual = xx = ww = ref = absolute_sum = r = cols = None
        timing = time_pair(torch, runner, b, x) if kind == 'normal' else None
        return record, timing, free
    finally:
        x = baseline = actual = r = cols = sample_base = sample_actual = xx = ww = ref = absolute_sum = None
        gc.collect(); torch.cuda.empty_cache()


def qualify(torch, runner, device):
    started = time.monotonic()
    initial = guard(torch, device)
    minimum = initial
    representatives = {}
    for b in runner.bindings.values():
        representatives.setdefault((b['N'], b['K']), b)
    records = []
    timings = []
    for (N, K), b in sorted(representatives.items()):
        for rows in ROWS:
            for kind in ('normal', 'alternating'):
                minimum = min(minimum, guard(torch, device))
                record, timing, free = fixture(torch, runner, b, rows, kind)
                minimum = min(minimum, free)
                records.append(record)
                if timing is not None:
                    timing.update(N=N, K=K, rows=rows)
                    timings.append(timing)
            runner.qualified.add((N, K, rows))
            if timings[-1]['tuned_candidate']:
                runner.tuned.add((N, K, rows))
    final = guard(torch, device, reserve=0)
    return dict(passed=True, cases=len(records), records=records, timings=timings,
                qualified_rows=list(ROWS), tuned_keys=[list(k) for k in sorted(runner.tuned)],
                initial_free=initial, final_free=final, min_sampled_free=min(minimum, final),
                seconds=time.monotonic() - started, gpu_qualified=True,
                performance_measured='operator CUDA events only; engine/wall benchmark still required')
