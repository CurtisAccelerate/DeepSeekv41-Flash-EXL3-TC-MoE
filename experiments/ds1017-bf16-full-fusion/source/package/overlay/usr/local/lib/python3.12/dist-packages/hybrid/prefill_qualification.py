"""Startup-only actual-weight operator check. Must run on every rank; not run by CPU staging."""
import gc
import time
from .prefill_contract import SUPPORTED


ROWS = (65, 128, 256, 512, 1024, 2048)
STYLES = ("uniform", "skewed", "six_hot")


def qualify(torch, original, adapter, layers, device, reserve_bytes=128*1024**2):
    if {key for key in layers} != SUPPORTED:
        raise ValueError("Every K2/K3 I512/I640 shape needs actual-weight qualification")
    records = []
    initial_free = int(torch.cuda.mem_get_info(device)[0])
    min_free = initial_free
    started = time.monotonic()
    for key, layer in sorted(layers.items()):
        tc = layer._exl3_tc
        for rows in ROWS:
            for style in STYLES:
                for seed in (1017, 40468):
                    torch.cuda.synchronize(device)
                    free = int(torch.cuda.mem_get_info(device)[0]); min_free = min(min_free, free)
                    if free - reserve_bytes < 1024**3:
                        raise ValueError("Prefill fixtures cannot retain the 1GiB startup free floor")
                    gen = torch.Generator(device=device).manual_seed(seed)
                    x = torch.randn((rows, 5120), device=device, dtype=torch.bfloat16, generator=gen) * .125
                    slot = torch.arange(rows*6, device=device, dtype=torch.int64).view(rows, 6)
                    if style == "uniform": ids = slot % 384
                    elif style == "skewed": ids = (slot // 6 % 8 * 6 + slot % 6) % 384
                    else: ids = slot % 6
                    weights = torch.rand((rows,6), device=device, dtype=torch.float32, generator=gen)
                    weights /= weights.sum(-1, keepdim=True)
                    # Nonzero limit exercises the activation-clamp ABI as well as unclamped output.
                    limit = 0.0 if seed == 1017 else 7.0
                    reference = original(x, ids, weights, layer, None, limit)
                    if reference is None:
                        raise ValueError("Incumbent rejected a declared qualification shape")
                    candidate = adapter.apply(x, ids, weights, tc, None, limit)
                    torch.cuda.synchronize(device)
                    exact = bool(torch.equal(reference.view(torch.int16), candidate.view(torch.int16)))
                    finite = bool(torch.isfinite(reference).all().item() and torch.isfinite(candidate).all().item())
                    records.append(dict(intermediate=key[0],bits_x2=key[1],rows=rows,style=style,seed=seed,
                                        limit=limit,bitwise_equal=exact,finite=finite))
                    if not exact or not finite:
                        raise ValueError("Prefill candidate failed actual-weight operator parity: " + str(records[-1]))
                    del x, ids, slot, weights, reference, candidate
                    gc.collect();torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    final_free = int(torch.cuda.mem_get_info(device)[0]);min_free=min(min_free,final_free)
    if final_free < 1024**3:
        raise ValueError("Prefill retained allocation violates startup free floor")
    return dict(passed=True,cases=len(records),records=records,seconds=time.monotonic()-started,
                initial_free=initial_free,final_free=final_free,min_sampled_free=min_free,
                coverage="Four actual-weight shapes, six row counts, three routing distributions, two seeds/clamp modes",
                model_quality_verified=False,performance_measured=False)
