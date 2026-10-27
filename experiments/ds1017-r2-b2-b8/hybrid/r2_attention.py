"""Rows-generic compressed split attention for R2 private FULL captures (rows = 6B).

Same Triton partial/reduce kernels, pinned sources and layer scope as
w26_attention_split; only the row count, the per-row indptr capacity and the split
profile are per-B. Runs only inside an R2 private capture (try_decode reaches it
through w26_dual_graph.active_candidate_context, which the R2 runtime sets).
"""
from . import w26_attention_split as base
from .r2_contract import SPLITS, WARPS, attention_workspace_bytes

EXPECTED_LAYERS = base.EXPECTED_LAYERS


def eligibility(layer, k, device, rows):
    """w26_attention_split.eligibility with (rows,16,512) q/out and rows+1 indptr."""
    lid = getattr(layer, "layer_id", None)
    ratio = getattr(layer, "compress_ratio", None)
    if lid not in EXPECTED_LAYERS or ratio != (2 if lid < 20 else 1):
        return "layer_or_compression"
    if k["swa_only"] or k["topk_indices"] is not None:
        return "not_compressed_ragged"
    for t in (k["q"], k["output"]):
        if (not t.is_cuda or t.device != device or str(t.dtype) != "torch.bfloat16"
                or tuple(t.shape) != (rows, 16, 512) or t.stride() != (8192, 512, 1)):
            return "query_or_output"
    if (k["head_dim"], k["nope_head_dim"], k["rope_head_dim"]) != (512, 448, 64) or k["scale"] != 512 ** -.5:
        return "dimensions_scale"
    if not base._layout(k["swa_k_cache"], 32, device) or not base._layout(k["kv_cache"], 128 // ratio, device):
        return "cache_layout"
    for name in ("swa_ragged_indices", "topk_ragged_indices"):
        if not base._vector(k[name], device):
            return "indices"
    for name in ("swa_ragged_indptr", "topk_ragged_indptr"):
        if not base._vector(k[name], device, rows + 1):
            return "indptr"
    if k.get("adaptive_splits", False):
        return "adaptive_splits"
    sink = k["attn_sink"]
    if sink is not None and (not sink.is_cuda or sink.device != device or sink.ndim != 1 or sink.numel() < 16
                             or not sink.is_contiguous() or str(sink.dtype) != "torch.float32"):
        return "sink"
    return None


class RowsProfile:
    """(splits, warps, direct) for one row count; outside the frozen B1 Profile matrix."""
    __slots__ = ("rows", "splits", "warps", "direct")

    def __init__(self, rows, splits, warps, direct):
        if rows % 6 or not 12 <= rows <= 48:
            raise ValueError("R2 attention rows must be 6B for B in 2..8")
        if splits not in SPLITS or warps not in WARPS or type(direct) is not bool:
            raise ValueError("R2 attention split outside the sweep matrix")
        self.rows, self.splits, self.warps, self.direct = rows, splits, warps, direct

    def record(self):
        return dict(rows=self.rows, splits=self.splits, warps=self.warps, direct=self.direct)

    @property
    def version(self):
        return f"r2-compressed-t6-rows{self.rows}-split{self.splits}-corew{self.warps}-direct{int(self.direct)}-v1"

    @property
    def workspace_bytes(self):
        return attention_workspace_bytes(self.rows // 6, (self.splits, self.warps, self.direct))

    def __eq__(self, other):
        return type(other) is RowsProfile and self.record() == other.record()

    def __hash__(self):
        return hash(tuple(self.record().values()))


class AttentionStateRows(base.AttentionState):
    """One workspace per R2 graph; never shared with the B1 banks or another B."""

    def __init__(self, device, profile):
        if type(profile) is not RowsProfile:
            raise ValueError("Expected RowsProfile")
        self.profile = profile
        self.rows = profile.rows
        import hashlib
        import inspect
        from pathlib import Path
        import torch
        import vllm.v1.attention.ops.rocm_aiter_mla_sparse as op
        import vllm.v1.attention.ops.fp8_sm80 as fp8
        self.torch = torch
        self.op = op
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        assert self.device.type == "cuda" and torch.cuda.get_device_capability(self.device) == (8, 0)
        assert not torch.cuda.is_current_stream_capturing()
        assert not op._ON_GFX942 and not op._ON_GFX950 and not op.current_platform.is_fp8_fnuz()
        for mod, pin in ((op, base.OP_SHA), (fp8, base.FP8_SHA)):
            assert hashlib.sha256(Path(inspect.getfile(mod)).read_bytes()).hexdigest() == pin, "retained source pin"
        r, s = self.rows, profile.splits
        self.m = torch.empty((r, s, 16), device=self.device, dtype=torch.float32)
        self.l = torch.empty_like(self.m)
        self.acc = torch.empty((r, s, 16, 512), device=self.device, dtype=torch.float32)
        self.temp = torch.empty((r, 16, 512), device=self.device, dtype=torch.bfloat16)
        self.dummy_sink = torch.empty(1, device=self.device, dtype=torch.float32)
        self.lut = op.get_e4m3fn_bf16_lut(self.device)
        self.prepared = False
        self.capturing = False
        self.layers = []
        self.rejections = []
        self.stream = None

    def _launch(self, k):
        op = self.op
        q = k["q"]
        main = k["swa_k_cache"]
        extra = k["kv_cache"]
        m, l, a = self.m, self.l, self.acc
        profile = self.profile
        split = profile.splits
        r = self.rows
        op._sparse_attn_decode_partial_kernel[(r, split, 1)](
            q, main, k["swa_ragged_indices"], k["swa_ragged_indptr"], extra, k["topk_ragged_indices"],
            k["topk_ragged_indptr"], m, l, a, self.lut,
            q.stride(0), q.stride(1), main.stride(0), extra.stride(0), m.stride(0), m.stride(1),
            a.stride(0), a.stride(1), a.stride(2),
            main.shape[0] * main.shape[1], extra.shape[0] * extra.shape[1], main.shape[1], extra.shape[1],
            k["scale"], 16,
            HAS_EXTRA=True, NOPE_DIM=448, NOPE_BLOCK=512, ROPE_DIM=64, IS_FNUZ_MAIN=False, IS_FNUZ_EXTRA=False,
            BLOCK_H=16, BLOCK_K=32, NUM_SPLITS=split, NUM_STAGES=1, num_warps=profile.warps)
        sink = k["attn_sink"]
        dest = k["output"] if profile.direct else self.temp
        op._sparse_attn_decode_reduce_kernel[(r, 16)](
            m, l, a, sink if sink is not None else self.dummy_sink, dest,
            dest.stride(0), dest.stride(1), m.stride(0), m.stride(1), a.stride(0), a.stride(1), a.stride(2), 16,
            HAS_ATTN_SINK=sink is not None, ADAPTIVE_SPLITS=False, COMB_DIM=512, BLOCK_H=1,
            NUM_SPLITS=split, SPLITS_PAD=split, num_warps=4)
        if not profile.direct:
            k["output"].copy_(self.temp)

    def compile_probe(self):
        """Empty segments for every row in both cache geometries must produce zeros."""
        t = self.torch
        assert not t.cuda.is_current_stream_capturing() and not self.capturing
        r = self.rows
        q = t.zeros((r, 16, 512), device=self.device, dtype=t.bfloat16)
        out = t.empty_like(q)
        raw = t.zeros(2 * base.PHYSICAL_STRIDE, device=self.device, dtype=t.uint8)
        main = raw.as_strided((1, 32, 584), (base.PHYSICAL_STRIDE, 584, 1), 0)
        ids = t.full((1,), -1, device=self.device, dtype=t.int32)
        ptr = t.zeros(r + 1, device=self.device, dtype=t.int32)
        sink = t.zeros(16, device=self.device, dtype=t.float32)
        for block in (64, 128):
            extra = raw.as_strided((1, block, 584), (base.PHYSICAL_STRIDE, 584, 1), base.PHYSICAL_STRIDE)
            for active_sink in (None, sink):
                k = dict(q=q, output=out, swa_k_cache=main, kv_cache=extra, swa_ragged_indices=ids,
                         swa_ragged_indptr=ptr, topk_ragged_indices=ids, topk_ragged_indptr=ptr,
                         attn_sink=active_sink, scale=512 ** -.5)
                self._launch(k)
                t.cuda.synchronize(self.device)
                assert bool((out == 0).all()), "empty split probe"
        self.prepared = True
        return dict(version=self.profile.version, profile=self.profile.record(), compiled=True,
                    probe="generated empty segments; not engine qualification",
                    workspace_bytes=self.profile.workspace_bytes)

    def finish_capture(self):
        """Non-raising: a rejected layer ran the incumbent kernel inside the same graph, so
        the graph stays valid and only the attention component is withheld for this B."""
        assert self.capturing and not self.torch.cuda.is_current_stream_capturing()
        self.capturing = False
        complete = not self.rejections and tuple(self.layers) == EXPECTED_LAYERS
        return dict(version=self.profile.version, profile=self.profile.record(), layers=self.layers.copy(),
                    rejections=list(self.rejections), ok=complete, scope=f"compressed target T6/B{self.rows // 6}")

    def run(self, layer, k):
        if not self.prepared or not self.capturing:
            raise ValueError("attention state not in prepared capture")
        if (not self.torch.cuda.is_current_stream_capturing()
                or self.torch.cuda.current_stream(self.device).cuda_stream != self.stream):
            raise ValueError("shared workspace requires the fixed private capture stream")
        reason = eligibility(layer, k, self.device, self.rows)
        if reason is not None:
            self.rejections.append((getattr(layer, "layer_id", None), reason))
            return False
        self._launch(k)
        self.layers.append(layer.layer_id)
        return True
