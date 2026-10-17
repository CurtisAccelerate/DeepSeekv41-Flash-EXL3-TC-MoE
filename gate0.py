# SPDX-License-Identifier: Apache-2.0
"""Gate 0 for the DS V4.1 tensor-core EXL3 MoE, one GPU, no server.

Incumbent: production exl3-tp4-align.py apply_exl3_fused_moe with
VLLM_EXL3_COOP=1 and VLLM_EXL3_MOE_KERNEL=exllamav3. Weights are the
/model-view tensors, sharded by that file's shard_exl3_col / shard_exl3_row
for chunk 0 (I=640) and chunk 2 (I=512). The FP64 reference is Morrowmake's
core_weight / apply_fp64 / moe_fp64 plus the DS SwiGLU clamp. Activations are
BF16 (created as BF16, never cast from FP16) and routed by the layer's own
gate. Row counts are prefixes of one 4096-row bank, which is the handoff's
slice/join rule: a row depends only on itself.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

LAYERS = (0, 10, 18, 22, 30, 39)
CHUNKS = ((0, 640), (2, 512))
DECODE_ROWS = (1, 6, 12, 18, 24, 36, 42, 48, 64)
PREFILL_ROWS = (65, 128, 512, 1024, 2048)
GO_DECODE = (6, 12, 36, 48)
GO_PREFILL = (1024, 2048)
SPEED_SHAPES = {(0, 0), (0, 2), (18, 0)}  # layer, chunk rank
HIDDEN = 5120
TOPK = 6
EXPERTS = 384
LIMIT = 10.0
BANK = 4096
MODEL = os.environ.get("GATE0_MODEL", "/model-view")
PLUGIN = os.environ.get("GATE0_PLUGIN", "/plugin/exl3-tp4-align.py")
SO = os.environ.get(
    "GATE0_SO",
    "/work/vllm_exl3/tc_moe/_dsv41_tc_moe_C.abi3.so",
)
OUT_DIR = os.environ.get("GATE0_OUT", "/work")


def log(msg: str) -> None:
    print(msg, flush=True)


def load_plugin():
    spec = importlib.util.spec_from_file_location("exl3_prod", PLUGIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lane_index() -> tuple[np.ndarray, np.ndarray]:
    rows = np.empty(256, np.int64)
    cols = np.empty(256, np.int64)
    for lane in range(32):
        q = lane % 4
        for c, column in enumerate((lane // 4, lane // 4 + 8)):
            for r, row in enumerate((2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9)):
                rows[lane * 8 + c * 4 + r] = row
                cols[lane * 8 + c * 4 + r] = column
    return rows, cols


_ROWS, _COLS = lane_index()
_LIN = torch.from_numpy((_ROWS * 16 + _COLS).astype(np.int64))


def codebook_lut(device: torch.device) -> torch.Tensor:
    states = np.arange(65536, dtype=np.uint64)
    mask = np.uint64(0xFFFFFFFF)
    product = (states * np.uint64(0x83DCD12D)) & mask
    total = sum(
        (product >> np.uint64(shift)) & np.uint64(0xFF) for shift in (0, 8, 16, 24)
    ).astype(np.float64)
    scale = float(np.array(0x1EEE, np.uint16).view(np.float16))
    bias = float(np.array(0xC931, np.uint16).view(np.float16))
    exact = ((1024.0 + total) * scale + bias).astype(np.float16).astype(np.float64)
    return torch.from_numpy(np.ascontiguousarray(exact)).to(device)


def hadamard(device: torch.device) -> torch.Tensor:
    index = np.arange(128)
    signs = np.vectorize(lambda v: bin(int(v)).count("1") & 1)(index[:, None] & index)
    h = (1.0 - 2.0 * signs) / np.sqrt(128.0)
    return torch.from_numpy(np.ascontiguousarray(h)).to(device=device, dtype=torch.float64)


def core_weight(trellis: torch.Tensor, bits_x2: int, lut: torch.Tensor, lin: torch.Tensor):
    """Morrowmake core_weight: float64 [padded_in, padded_out] on trellis.device."""
    device = trellis.device
    k_tiles, n_tiles, halves = trellis.shape
    h = trellis.to(torch.int64) & 0xFFFF
    words = h[..., 0::2] | (h[..., 1::2] << 16)
    shifts = torch.arange(31, -1, -1, device=device)
    ring = ((words.unsqueeze(-1) >> shifts) & 1).reshape(*words.shape[:-1], -1)
    ends = torch.arange(1, 257, device=device) * bits_x2 // 2
    window = (ends[:, None] - 16 + torch.arange(16, device=device)) % ring.shape[-1]
    states = torch.zeros(*ring.shape[:-1], 256, dtype=torch.int64, device=device)
    for bit in range(16):
        states |= ring[..., window[:, bit]] << (15 - bit)
    vals = lut[states.reshape(-1)].reshape(-1, 256)
    tiles = torch.empty(vals.shape[0], 256, dtype=torch.float64, device=device)
    tiles[:, lin] = vals
    tiles = tiles.reshape(k_tiles, n_tiles, 16, 16)
    return tiles.permute(0, 2, 1, 3).reshape(k_tiles * 16, n_tiles * 16)


def block_hadamard(v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return (v.reshape(v.shape[0], -1, 128) @ h).reshape(v.shape)


def apply_fp64(x, core, suh, svh, h):
    t = block_hadamard(x.double() * suh[: x.shape[1]].double(), h)
    return block_hadamard(t @ core, h) * svh.double()


class Fp64Pool:
    """Morrowmake _Fp64Pool. BF16 rounding ties are forgiven."""

    def __init__(self):
        self.n = 0
        self.c_sum = self.i_sum = 0.0
        self.c_max = self.i_max = 0.0
        self.parts = []

    def add(self, cand, inc, ref, name):
        cand = cand.detach().to(dtype=torch.bfloat16).cpu().contiguous()
        inc = inc.detach().to(dtype=torch.bfloat16).cpu().contiguous()
        ref = ref.detach().double().cpu().contiguous()
        if not torch.isfinite(cand).all():
            raise AssertionError(f"{name}: candidate is not finite")
        if not torch.isfinite(inc).all():
            raise AssertionError(f"{name}: incumbent is not finite")
        ec = (cand.double() - ref).abs()
        ei = (inc.double() - ref).abs()
        exact = ei == 0
        if not bool((ec[exact] == 0).all()):
            raise AssertionError(f"{name}: inexact where the incumbent is exact")
        self.n += ec.numel()
        self.c_sum += float(ec.sum())
        self.i_sum += float(ei.sum())
        self.c_max = max(self.c_max, float(ec.max()))
        self.i_max = max(self.i_max, float(ei.max()))
        self.parts.append((ec, tie_mask(cand, inc, ref)))

    def ratios(self):
        if self.i_sum == 0.0:
            mean_ratio = 1.0 if self.c_sum == 0.0 else float("inf")
        else:
            mean_ratio = self.c_sum / self.i_sum
        if self.i_max == 0.0:
            max_ratio = 1.0 if self.c_max == 0.0 else float("inf")
        else:
            max_ratio = self.c_max / self.i_max
        if max_ratio > 1.25 and self.i_max > 0.0:
            budget = 1.25 * self.i_max
            over = sum(int((e > budget).sum()) for e, _ in self.parts)
            forgiven = sum(int(((e > budget) & t).sum()) for e, t in self.parts)
            if over == forgiven and forgiven <= 1e-3 * self.n:
                rest = max(float(torch.where(t, 0.0, e).max()) for e, t in self.parts)
                max_ratio = rest / self.i_max
        return mean_ratio, max_ratio

    def check(self, name):
        mean_ratio, max_ratio = self.ratios()
        if mean_ratio > 1.10:
            raise AssertionError(f"{name}: mean error x{mean_ratio:.3f} of the incumbent")
        if max_ratio > 1.25:
            raise AssertionError(f"{name}: max error x{max_ratio:.3f} of the incumbent")
        return mean_ratio, max_ratio


def tie_mask(cand, inc, ref):
    ci = cand.view(torch.int16).to(torch.int32)
    ii = inc.view(torch.int16).to(torch.int32)
    adjacent = ((ci - ii).abs() == 1) & ((ci < 0) == (ii < 0))
    c, i = cand.double(), inc.double()
    return adjacent & ((ref - 0.5 * (c + i)).abs() <= (c - i).abs() / 16)


def load_cpu(layer: int):
    from safetensors import safe_open

    index = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    groups: dict[str, list[str]] = {}
    keys = []
    for e in range(EXPERTS):
        for proj in ("w1", "w2", "w3"):
            for suf in ("trellis", "suh", "svh", "mul1"):
                key = f"layers.{layer}.ffn.experts.{e}.{proj}.{suf}"
                keys.append(key)
                groups.setdefault(weight_map[key], []).append(key)
    gate_key = f"layers.{layer}.ffn.gate.weight"
    bias_key = f"layers.{layer}.ffn.gate.bias"
    for key in (gate_key, bias_key):
        groups.setdefault(weight_map[key], []).append(key)
    tensors = {}
    for fname, names in groups.items():
        with safe_open(os.path.join(MODEL, fname), framework="pt", device="cpu") as handle:
            for name in names:
                tensors[name] = handle.get_tensor(name)
    return tensors


def shard_pack(prod, tensors, layer, expert, proj, tp_rank):
    key = f"layers.{layer}.ffn.experts.{expert}.{proj}"
    trellis = tensors[key + ".trellis"]
    suh = tensors[key + ".suh"]
    svh = tensors[key + ".svh"]
    mul1 = tensors[key + ".mul1"]
    col = proj in ("w1", "w3")
    shard = prod.shard_exl3_col if col else prod.shard_exl3_row
    return (
        shard(trellis, "trellis", tp_rank, 4, aligned=True),
        shard(suh, "suh", tp_rank, 4, aligned=True),
        shard(svh, "svh", tp_rank, 4, aligned=True),
        mul1.contiguous(),
    )


def build_experts(prod, tensors, layer, tp_rank, intermediate, device):
    inners = []
    bits = None
    for e in range(EXPERTS):
        pack = {}
        for proj, name in (("w1", "gate"), ("w3", "up"), ("w2", "down")):
            trellis, suh, svh, mul1 = shard_pack(prod, tensors, layer, e, proj, tp_rank)
            bx2 = int(trellis.shape[-1] // 8)
            if bits is None:
                bits = bx2
            elif bits != bx2:
                raise RuntimeError(f"layer {layer} mixed bits_x2 {bits} and {bx2}")
            linear = prod.make_linear_exl3(
                trellis.to(device),
                suh.to(device),
                svh.to(device),
                mul1=mul1.to(device),
                out_dtype=torch.float16,
            )
            pack[name] = linear
        inners.append(pack)
    if bits not in (4, 6):
        raise RuntimeError(f"layer {layer} bits_x2 {bits}")
    k = bits // 2
    layer_mod = torch.nn.Module()
    layer_mod._exl3_hidden_size = HIDDEN
    layer_mod._exl3_intermediate_local = intermediate
    layer_mod._exl3_bits = k
    layer_mod._exl3_k = k
    layer_mod._exl3_mixed_k = False
    layer_mod._exl3_codebook_flags = (False, True) * 3
    layer_mod.w13_suh = inners[0]["gate"].suh
    prod.build_exl3_fused_state(layer_mod, inners)
    ptrs = torch.empty(EXPERTS, 3, 3, dtype=torch.int64, device=device)
    meta = torch.empty(EXPERTS, 3, 4, dtype=torch.int64, device=device)
    for e, pack in enumerate(inners):
        for p, name in enumerate(("gate", "up", "down")):
            linear = pack[name]
            ptrs[e, p, 0] = linear.trellis.data_ptr()
            ptrs[e, p, 1] = linear.suh.data_ptr()
            ptrs[e, p, 2] = linear.svh.data_ptr()
            if p < 2:
                meta[e, p] = torch.tensor([HIDDEN, intermediate, bits, 2], device=device)
            else:
                meta[e, p] = torch.tensor([intermediate, HIDDEN, bits, 2], device=device)
    emap = torch.arange(EXPERTS, dtype=torch.int32, device=device)
    gate = tensors[f"layers.{layer}.ffn.gate.weight"]
    bias = tensors[f"layers.{layer}.ffn.gate.bias"]
    return {
        "inners": inners,
        "layer": layer_mod,
        "ptrs": ptrs,
        "meta": meta,
        "emap": emap,
        "bits": bits,
        "k": k,
        "gate": gate,
        "bias": bias,
    }


def route_bank(built, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(20261016 + int(built["bits"]) * 100 + int(built["layer"]._exl3_intermediate_local))
    x = torch.randn(BANK, HIDDEN, generator=g).to(torch.bfloat16)
    x[:32].mul_(40)
    weight = built["gate"]
    bias = built["bias"].float()
    if weight.shape == (EXPERTS, HIDDEN):
        scores = x.float() @ weight.float().T
    elif weight.shape == (HIDDEN, EXPERTS):
        scores = x.float() @ weight.float()
    else:
        raise RuntimeError(f"gate.weight shape {tuple(weight.shape)}")
    scores = scores + bias.reshape(1, -1)
    values, ids = torch.topk(scores, TOPK, dim=-1)
    weights = torch.softmax(values, dim=-1).to(torch.float32)
    return (
        x.to(device).contiguous(),
        ids.to(device=device, dtype=torch.long).contiguous(),
        weights.to(device).contiguous(),
    )


def tc_call(ops, which, built, x, ids, weights, out, workspace):
    fn = ops.moe_out_v2 if which == "v2" else ops.moe_out_prefill
    out.zero_()
    fn(
        x,
        ids,
        weights,
        built["ptrs"],
        built["meta"],
        built["emap"],
        int(built["layer"]._exl3_intermediate_local),
        int(built["bits"]),
        LIMIT,
        out,
        workspace,
    )
    return out


def workspace_for(ops, which, rows, intermediate):
    fn = ops.moe_workspace_size_v2 if which == "v2" else ops.moe_workspace_size_prefill
    return int(fn(rows, HIDDEN, intermediate, TOPK, EXPERTS))


def incumbent(prod, built, x, ids, weights):
    return prod.apply_exl3_fused_moe(
        x, ids, weights, built["layer"], built["inners"], None, LIMIT
    )


def path_of(ids):
    """Same dispatch the production plugin uses, without wrapping the binding."""
    rows = int(ids.shape[0])
    topk = int(ids.shape[1])
    if rows * topk <= 256:
        return "coop"
    flat = ids.reshape(-1)
    valid = (flat >= 0) & (flat < EXPERTS)
    if rows > 256 and valid.any():
        counts = torch.bincount(flat[valid].to(torch.long), minlength=EXPERTS)
        if bool((counts > 256).any()):
            return "fat"
    return "stock"


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def median_ms(graph, repeats, warmup):
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(repeats):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def median_ms_eager(fn, repeats, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def reference(built, x, ids, weights, lut, h, lin, device):
    """FP64 routed output with the SwiGLU clamp. One expert at a time."""
    inter = int(built["layer"]._exl3_intermediate_local)
    bits = int(built["bits"])
    out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float64, device=device)
    routes: dict[int, list[tuple[int, int, float]]] = {}
    id_cpu = ids.cpu().tolist()
    w_cpu = weights.double().cpu().tolist()
    for row, (row_ids, row_w) in enumerate(zip(id_cpu, w_cpu)):
        for g, w in zip(row_ids, row_w):
            if 0 <= g < EXPERTS:
                routes.setdefault(int(g), []).append((row, w))
    peak_gate = 0.0
    t0 = time.perf_counter()
    for n_done, (local, items) in enumerate(routes.items(), start=1):
        rows = torch.tensor([row for row, _ in items], device=device, dtype=torch.long)
        xx = x.index_select(0, rows)
        parts = []
        for name in ("gate", "up", "down"):
            linear = built["inners"][local][name]
            parts.append(
                (
                    core_weight(linear.trellis, bits, lut, lin),
                    linear.suh.detach(),
                    linear.svh.detach(),
                )
            )
        gv = apply_fp64(xx, *parts[0], h)[:, :inter]
        uv = apply_fp64(xx, *parts[1], h)[:, :inter]
        peak_gate = max(peak_gate, float(gv.abs().max()))
        gv = torch.clamp(gv, max=LIMIT)
        uv = torch.clamp(uv, min=-LIMIT, max=LIMIT)
        act = gv / (1.0 + torch.exp(-gv)) * uv
        y = apply_fp64(act, *parts[2], h)[:, :HIDDEN]
        scale = torch.tensor([w for _, w in items], dtype=torch.float64, device=device).unsqueeze(-1)
        out.index_add_(0, rows, y * scale)
        del parts, gv, uv, act, y
        if n_done == 1 or n_done % 64 == 0:
            log(f"    fp64 experts {n_done}/{len(routes)} {time.perf_counter() - t0:.1f}s")
    return out, peak_gate


def bitwise(a, b, name):
    if not torch.equal(a.view(torch.int16), b.view(torch.int16)):
        raise AssertionError(name)


def run_pool(prod, ops, ext, built, bank, lut, h, lin, device, speed: bool):
    x_all, ids_all, w_all = bank
    inter = int(built["layer"]._exl3_intermediate_local)
    bits = int(built["bits"])
    pool = Fp64Pool()
    checks = []
    ref, peak = reference(built, x_all[:2048], ids_all[:2048], w_all[:2048], lut, h, lin, device)
    if peak <= LIMIT:
        raise AssertionError(f"clamp rows did not exceed {LIMIT}: peak |gate| {peak:.4f}")
    checks.append(("clamp", "pass", f"peak |gate| {peak:.4f}"))

    def take(rows):
        return x_all[:rows], ids_all[:rows], w_all[:rows], ref[:rows]

    for rows in DECODE_ROWS:
        x, ids, weights, gold = take(rows)
        ws = torch.empty(workspace_for(ops, "v2", rows, inter), dtype=torch.float32, device=device)
        out = torch.empty_like(x)
        cand = tc_call(ops, "v2", built, x, ids, weights, out, ws).clone()
        inc = incumbent(prod, built, x, ids, weights)
        pool.add(cand, inc, gold, f"decode {rows}")
        repeats = [tc_call(ops, "v2", built, x, ids, weights, out, ws).clone() for _ in range(3)]
        bitwise(repeats[0], repeats[1], f"decode {rows} repeat 1")
        bitwise(repeats[1], repeats[2], f"decode {rows} repeat 2")
        del ws, out, cand, inc
    for rows in PREFILL_ROWS:
        x, ids, weights, gold = take(rows)
        ws = torch.empty(workspace_for(ops, "prefill", rows, inter), dtype=torch.float32, device=device)
        out = torch.empty_like(x)
        cand = tc_call(ops, "prefill", built, x, ids, weights, out, ws).clone()
        inc = incumbent(prod, built, x, ids, weights)
        pool.add(cand, inc, gold, f"prefill {rows}")
        repeats = [tc_call(ops, "prefill", built, x, ids, weights, out, ws).clone() for _ in range(3)]
        bitwise(repeats[0], repeats[1], f"prefill {rows} repeat 1")
        bitwise(repeats[1], repeats[2], f"prefill {rows} repeat 2")
        del ws, out, cand, inc
    checks.append(("determinism", "pass", "three repeats bitwise identical at every row count"))

    x, ids, weights, _gold = take(2048)
    os.environ["VLLM_EXL3_TC_PREFILL_ROWS"] = "512"
    ws = torch.empty(workspace_for(ops, "prefill", 512, inter), dtype=torch.float32, device=device)
    out = torch.empty_like(x)
    small = tc_call(ops, "prefill", built, x, ids, weights, out, ws).clone()
    os.environ["VLLM_EXL3_TC_PREFILL_ROWS"] = "2048"
    ws = torch.empty(workspace_for(ops, "prefill", 2048, inter), dtype=torch.float32, device=device)
    full = tc_call(ops, "prefill", built, x, ids, weights, out, ws).clone()
    bitwise(small, full, "R=512 vs R=2048")
    checks.append(("subchunk", "pass", "R=512 bitwise equals R=2048 at 2048 rows"))

    x, ids, weights, _gold = take(36)
    eager_ws = torch.empty(workspace_for(ops, "v2", 36, inter), dtype=torch.float32, device=device)
    eager_out = torch.empty_like(x)
    eager = tc_call(ops, "v2", built, x, ids, weights, eager_out, eager_ws).clone()
    graph_out = torch.empty_like(x)
    graph_ws = torch.empty_like(eager_ws)
    before = torch.cuda.memory_allocated()

    def graph_body():
        tc_call(ops, "v2", built, x, ids, weights, graph_out, graph_ws)

    graph = capture(graph_body)
    allocated = torch.cuda.memory_allocated()
    graph.replay()
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    bitwise(graph_out, eager, "graph replay vs eager")
    if after != allocated:
        raise AssertionError(f"graph replay allocated {after - allocated} bytes")
    checks.append(
        (
            "graph",
            "pass",
            f"36-row replay bitwise eager; replay alloc delta 0 (capture delta {allocated - before})",
        )
    )

    dead = ids.clone()
    dead.fill_(-1)
    out = torch.empty_like(x)
    ws = torch.empty_like(eager_ws)
    tc_call(ops, "v2", built, x, dead, weights, out, ws)
    if not torch.equal(out, torch.zeros_like(out)):
        raise AssertionError("non-local decode rows are not exact 0")
    x2048, _, w2048 = x_all[:2048], ids_all[:2048], w_all[:2048]
    deadp = ids_all[:2048].clone()
    deadp.fill_(-1)
    outp = torch.empty_like(x2048)
    wsp = torch.empty(workspace_for(ops, "prefill", 2048, inter), dtype=torch.float32, device=device)
    tc_call(ops, "prefill", built, x2048, deadp, w2048, outp, wsp)
    if not torch.equal(outp, torch.zeros_like(outp)):
        raise AssertionError("non-local prefill rows are not exact 0")
    checks.append(("nonlocal", "pass", "ids -1 are exact 0 for v2 and prefill"))

    mean_ratio, max_ratio = pool.check(f"I={inter} bits_x2={bits}")
    checks.append(("fp64", "pass", f"mean x{mean_ratio:.4f} max x{max_ratio:.4f}"))

    speed_rows = []
    if speed:
        for rows in DECODE_ROWS:
            xx, ii, ww, _g = take(rows)
            v2_out = torch.empty_like(xx)
            v2_ws = torch.empty(workspace_for(ops, "v2", rows, inter), dtype=torch.float32, device=device)
            which = path_of(ii)

            def v2_body(xx=xx, ii=ii, ww=ww, v2_out=v2_out, v2_ws=v2_ws):
                tc_call(ops, "v2", built, xx, ii, ww, v2_out, v2_ws)

            holder = {}

            def inc_body(xx=xx, ii=ii, ww=ww):
                holder["y"] = incumbent(prod, built, xx, ii, ww)

            v2_ms = median_ms(capture(v2_body), 200, 20)
            inc_ms = median_ms(capture(inc_body), 200, 20)
            speed_rows.append(
                {"rows": rows, "phase": "decode", "v2_ms": v2_ms, "inc_ms": inc_ms, "path": which}
            )
            log(f"  speed decode {rows} v2 {v2_ms:.4f} inc {inc_ms:.4f} {which}")
        for rows in PREFILL_ROWS:
            xx, ii, ww, _g = take(rows)
            v2_out = torch.empty_like(xx)
            v2_ws = torch.empty(
                workspace_for(ops, "prefill", rows, inter), dtype=torch.float32, device=device
            )
            which = path_of(ii)

            def v2_eager(xx=xx, ii=ii, ww=ww, v2_out=v2_out, v2_ws=v2_ws):
                tc_call(ops, "prefill", built, xx, ii, ww, v2_out, v2_ws)

            def inc_eager(xx=xx, ii=ii, ww=ww):
                return incumbent(prod, built, xx, ii, ww)

            v2_ms = median_ms_eager(v2_eager, 200, 20)
            inc_ms = median_ms_eager(inc_eager, 200, 20)
            speed_rows.append(
                {"rows": rows, "phase": "prefill", "v2_ms": v2_ms, "inc_ms": inc_ms, "path": which}
            )
            log(f"  speed prefill {rows} v2 {v2_ms:.4f} inc {inc_ms:.4f} {which}")
    return {
        "checks": checks,
        "mean_ratio": mean_ratio,
        "max_ratio": max_ratio,
        "speed": speed_rows,
        "pass": True,
    }


def write_report(pools, error):
    decode_go = prefill_go = False
    lines = [
        "# Gate 0: tensor-core EXL3 MoE",
        "",
        "One GPU, no production server. Incumbent is `apply_exl3_fused_moe` from",
        "`exl3-tp4-align.py` with `VLLM_EXL3_COOP=1` and `VLLM_EXL3_MOE_KERNEL=exllamav3`.",
        "Activations are BF16 created as BF16 and routed by each layer's own gate.",
        "The 4-GPU eager trace server was not started.",
        "",
        "## Correctness",
        "",
        "| Pool | FP64 mean | FP64 max | Result | Notes |",
        "|---|---:|---:|---|---|",
    ]
    for pool in pools:
        name = (
            f"L{pool['layer']} chunk {pool['chunk']} I={pool['intermediate']} "
            f"K{pool['bits'] // 2}"
        )
        if pool.get("pass"):
            notes = "; ".join(f"{c[0]} {c[1]}" for c in pool["checks"])
            lines.append(
                f"| {name} | {pool['mean_ratio']:.4f} | {pool['max_ratio']:.4f} | pass | {notes} |"
            )
        else:
            lines.append(f"| {name} |  |  | fail | {pool.get('error', '')} |")
    lines += ["", "## Speed", "", "Median of 200 CUDA-event samples after 20 warm-ups.",
              "Decode is inside captured graphs. Prefill is eager, as production leaves it.",
              "",
              "| Shape | Rows | Phase | v2 ms | incumbent ms | path | v2/inc |",
              "|---|---:|---|---:|---:|---|---:|"]
    speed_index = {}
    for pool in pools:
        shape = (pool["intermediate"], pool["bits"])
        for row in pool.get("speed") or []:
            ratio = row["v2_ms"] / row["inc_ms"] if row["inc_ms"] else float("inf")
            lines.append(
                f"| I={shape[0]} K{shape[1] // 2} | {row['rows']} | {row['phase']} | "
                f"{row['v2_ms']:.4f} | {row['inc_ms']:.4f} | {row['path']} | {ratio:.4f} |"
            )
            speed_index[(shape, row["phase"], row["rows"])] = ratio
    needed = []
    for inter, bits, layer, chunk in ((640, 6, 0, 0), (512, 6, 0, 2), (640, 4, 18, 0)):
        for rows in GO_DECODE:
            needed.append(((inter, bits), "decode", rows))
        for rows in GO_PREFILL:
            needed.append(((inter, bits), "prefill", rows))
    decode_go = all(
        speed_index.get((shape, "decode", rows), 99) < 1.0
        for shape, _phase, rows in needed
        if _phase == "decode"
    )
    prefill_go = all(
        speed_index.get((shape, "prefill", rows), 99) < 1.0
        for shape, phase, rows in needed
        if phase == "prefill"
    )
    correct = all(pool.get("pass") for pool in pools) and len(pools) == len(LAYERS) * len(CHUNKS)
    if not correct:
        decode_go = False
        prefill_go = False
    if decode_go and prefill_go:
        decision = "decode go and prefill go"
    elif decode_go:
        decision = "decode go"
    elif prefill_go:
        decision = "prefill go"
    else:
        decision = "stop"
    lines += ["", "## Decision", "", f"**{decision}**", ""]
    if error:
        lines += ["Runner error:", "", "```", error, "```", ""]
    if not any(pool.get("bits") == 4 for pool in pools):
        lines.append(
            "No pool has bits_x2 4. The loaded trellis last dimension is 48, so these "
            "weights are K3 on the measured layers, including 18 and 22. The I=640 K2 "
            "shape is not in the checkpoint, and the three-shape rule is not met."
        )
    lines.append(
        "A flag is not promoted from this run. Acceptance A/B was not run."
    )
    text = "\n".join(lines) + "\n"
    path = os.path.join(OUT_DIR, "gate0-report.md")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    with open(os.path.join(OUT_DIR, "gate0-report.json"), "w", encoding="utf-8") as handle:
        json.dump({"pools": pools, "decision": decision, "error": error}, handle, indent=2)
    log(f"WROTE {path}")
    log(f"DECISION {decision}")
    return decision


def main():
    log(f"torch {torch.__version__} cuda {torch.version.cuda} device {torch.cuda.get_device_name(0)}")
    torch.ops.load_library(SO)
    ops = torch.ops._dsv41_tc_moe_C
    prod = load_plugin()
    import exllamav3_ext

    device = torch.device("cuda:0")
    lut = codebook_lut(device)
    h = hadamard(device)
    lin = _LIN.to(device)
    pools = []
    error = ""
    try:
        for layer in LAYERS:
            log(f"loading layer {layer}")
            tensors = load_cpu(layer)
            for chunk, intermediate in CHUNKS:
                log(f"layer {layer} chunk {chunk} I={intermediate}")
                built = build_experts(prod, tensors, layer, chunk, intermediate, device)
                if int(built["layer"]._exl3_intermediate_local) != intermediate:
                    raise RuntimeError("intermediate mismatch")
                gate = built["inners"][0]["gate"]
                if int(gate.svh.numel()) != intermediate or int(gate.suh.numel()) != HIDDEN:
                    raise RuntimeError(
                        f"gate shape suh {gate.suh.numel()} svh {gate.svh.numel()}"
                    )
                bank = route_bank(built, device)
                try:
                    result = run_pool(
                        prod,
                        ops,
                        exllamav3_ext,
                        built,
                        bank,
                        lut,
                        h,
                        lin,
                        device,
                        speed=(layer, chunk) in SPEED_SHAPES,
                    )
                except Exception as exc:
                    result = {
                        "pass": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "checks": [],
                        "speed": [],
                    }
                    log("POOL FAIL " + result["error"])
                    traceback.print_exc()
                result.update(
                    {
                        "layer": layer,
                        "chunk": chunk,
                        "intermediate": intermediate,
                        "bits": int(built["bits"]),
                    }
                )
                pools.append(result)
                write_report(pools, error)
                del built, bank
                torch.cuda.empty_cache()
            del tensors
    except Exception:
        error = traceback.format_exc()
        log(error)
    write_report(pools, error)
    return 1 if error else 0


if __name__ == "__main__":
    sys.exit(main())
