# Row dispatch for the tensor-core EXL3 MoE, corrected after Gate 0.
# Coop keeps 1-42 rows. v2 is only 43-64. Prefill TC is 65 through the row cap.
# K2 is not a served shape on this checkpoint, so it is not a gate.

import logging
import os

import torch

logger = logging.getLogger("vllm.exl3.tc_moe")

_TC_V2 = os.environ.get("VLLM_EXL3_TC_MOE_V2", "0") == "1"
_TC_PREFILL = os.environ.get("VLLM_EXL3_TC_MOE_PREFILL", "0") == "1"
_TC_PREFILL_ROWS = int(os.environ.get("VLLM_EXL3_TC_PREFILL_ROWS", "2048"))
_TC_SO = os.environ.get(
    "VLLM_EXL3_TC_MOE_SO", "/opt/dsv41-tc-moe/_dsv41_tc_moe_C.abi3.so"
)
_OPS = None
_DTYPE_LOGGED = False
_SHARED_WORKSPACE = {}
_HIDDEN = 5120
_TOPK = 6
_EXPERTS = 384


def _ops():
    global _OPS
    if _OPS is None:
        torch.ops.load_library(_TC_SO)
        _OPS = torch.ops._dsv41_tc_moe_C
    return _OPS


def _reject(layer, inners) -> str | None:
    if not (_TC_V2 or _TC_PREFILL):
        return "flags off"
    if len(inners) != _EXPERTS:
        return f"local experts {len(inners)}"
    if getattr(layer, "_exl3_mixed_k", False) or getattr(layer, "_exl3_mixedk_unified", False):
        return "mixed-K"
    hidden = int(getattr(layer, "_exl3_hidden_size", -1))
    intermediate = int(getattr(layer, "_exl3_intermediate_local", -1))
    if hidden != _HIDDEN:
        return f"hidden {hidden}"
    if intermediate not in (640, 512):
        return f"intermediate {intermediate}"
    flags = tuple(getattr(layer, "_exl3_codebook_flags", ()))
    if flags != (False, True) * 3:
        return f"codebook {flags}"
    device = inners[0]["gate"].trellis.device
    if torch.cuda.get_device_capability(device) != (8, 0):
        return f"capability {torch.cuda.get_device_capability(device)}"
    bits = None
    for pack in inners:
        for name, kind in (("gate", "col"), ("up", "col"), ("down", "row")):
            linear = pack[name]
            trellis, suh, svh = linear.trellis, linear.suh, linear.svh
            if trellis.dtype != torch.int16 or not trellis.is_contiguous():
                return f"{name} trellis dtype/layout"
            if suh.dtype != torch.float16 or svh.dtype != torch.float16:
                return f"{name} suh/svh dtype"
            if not suh.is_contiguous() or not svh.is_contiguous():
                return f"{name} suh/svh layout"
            if trellis.data_ptr() % 16 or suh.data_ptr() % 16 or svh.data_ptr() % 16:
                return f"{name} alignment"
            bx2 = int(trellis.shape[-1] // 8)
            if bx2 not in (4, 6):
                return f"{name} bits_x2 {bx2}"
            if bits is None:
                bits = bx2
            elif bits != bx2:
                return f"mixed bits_x2 {bits} and {bx2}"
            if kind == "col" and tuple(trellis.shape[:2]) != (hidden // 16, intermediate // 16):
                return f"{name} trellis {tuple(trellis.shape)}"
            if kind == "row" and tuple(trellis.shape[:2]) != (intermediate // 16, hidden // 16):
                return f"{name} trellis {tuple(trellis.shape)}"
    return None


def _warmup(ops, which, tc, rows: int) -> None:
    device = tc["workspace"].device
    x = torch.zeros(rows, _HIDDEN, dtype=torch.bfloat16, device=device)
    ids = torch.zeros(rows, _TOPK, dtype=torch.long, device=device)
    weights = torch.full((rows, _TOPK), 1.0 / _TOPK, dtype=torch.float32, device=device)
    out = torch.empty_like(x)
    fn = ops.moe_out_v2 if which == "v2" else ops.moe_out_prefill
    fn(
        x, ids, weights, tc["ptrs"], tc["meta"], tc["emap"],
        tc["intermediate"], tc["bits_x2"], 10.0, out, tc["workspace"],
    )


def try_build(layer, inners) -> None:
    reason = _reject(layer, inners)
    if reason == "flags off":
        return
    layer_id = getattr(layer, "layer_idx", getattr(layer, "_exl3_layer_idx", "?"))
    if reason:
        logger.info("EXL3 TC MoE layer=%s skipped: %s", layer_id, reason)
        return
    try:
        ops = _ops()
        device = inners[0]["gate"].trellis.device
        intermediate = int(layer._exl3_intermediate_local)
        bits = int(inners[0]["gate"].trellis.shape[-1] // 8)
        ptrs = torch.empty((_EXPERTS, 3, 3), dtype=torch.int64, device=device)
        meta = torch.empty((_EXPERTS, 3, 4), dtype=torch.int64, device=device)
        for e, pack in enumerate(inners):
            for p, name in enumerate(("gate", "up", "down")):
                linear = pack[name]
                ptrs[e, p, 0] = linear.trellis.data_ptr()
                ptrs[e, p, 1] = linear.suh.data_ptr()
                ptrs[e, p, 2] = linear.svh.data_ptr()
                if p < 2:
                    meta[e, p, 0] = _HIDDEN
                    meta[e, p, 1] = intermediate
                else:
                    meta[e, p, 0] = intermediate
                    meta[e, p, 1] = _HIDDEN
                meta[e, p, 2] = bits
                meta[e, p, 3] = 2
        emap = getattr(layer, "expert_map", None)
        if emap is None:
            emap = torch.arange(_EXPERTS, dtype=torch.int32, device=device)
        elif emap.device.type != "cuda":
            emap = emap.to(device=device, dtype=torch.int32)
        # One buffer per device. All MoE layers run in order on one stream.
        # Size it for the larger intermediate so an I=512 layer can share it.
        dev_key = device.index if device.index is not None else 0
        workspace = _SHARED_WORKSPACE.get(dev_key)
        if workspace is None:
            n_v2 = int(ops.moe_workspace_size_v2(64, _HIDDEN, 640, _TOPK, _EXPERTS))
            n_pre = int(ops.moe_workspace_size_prefill(
                _TC_PREFILL_ROWS, _HIDDEN, 640, _TOPK, _EXPERTS
            ))
            workspace = torch.empty(max(n_v2, n_pre), dtype=torch.float32, device=device)
            _SHARED_WORKSPACE[dev_key] = workspace
        tc = {
            "ops": ops,
            "ptrs": ptrs,
            "meta": meta,
            "emap": emap,
            "workspace": workspace,
            "intermediate": intermediate,
            "bits_x2": bits,
            "v2": _TC_V2,
            "prefill": _TC_PREFILL,
            "prefill_rows": _TC_PREFILL_ROWS,
        }
        if _TC_V2:
            _warmup(ops, "v2", tc, 48)
        if _TC_PREFILL:
            _warmup(ops, "prefill", tc, 65)
        layer._exl3_tc = tc
        logger.info(
            "EXL3 TC MoE layer=%s I=%s bits_x2=%s v2=%s prefill=%s ws=%.1fMiB",
            layer_id, intermediate, bits,
            "on" if _TC_V2 else "off",
            "on" if _TC_PREFILL else "off",
            workspace.numel() * 4 / (1024 * 1024),
        )
    except Exception as exc:
        layer._exl3_tc = None
        logger.warning("EXL3 TC MoE layer=%s build failed: %s", layer_id, exc)


def try_apply(x2d, ids, weights, layer, expert_map, limit):
    """Return a BF16 tensor, or None to keep today's coop/stock path."""
    global _DTYPE_LOGGED
    tc = getattr(layer, "_exl3_tc", None)
    if tc is None or x2d.dim() != 2:
        return None
    if x2d.dtype != torch.bfloat16:
        if not _DTYPE_LOGGED:
            logger.warning("EXL3 TC MoE left off: activation dtype %s", x2d.dtype)
            _DTYPE_LOGGED = True
        return None
    rows = int(x2d.shape[0])
    if int(ids.shape[-1]) != _TOPK:
        return None
    if tc["v2"] and 43 <= rows <= 64:
        fn = tc["ops"].moe_out_v2
    elif tc["prefill"] and 65 <= rows <= tc["prefill_rows"]:
        fn = tc["ops"].moe_out_prefill
    else:
        return None
    emap = tc["emap"] if expert_map is None else expert_map
    if emap.device != x2d.device or not x2d.is_cuda:
        return None
    out = torch.empty_like(x2d)
    w = weights if weights.dtype == torch.float32 else weights.float()
    fn(
        x2d.contiguous(),
        ids,
        w.contiguous(),
        tc["ptrs"],
        tc["meta"],
        emap,
        tc["intermediate"],
        tc["bits_x2"],
        float(limit or 0.0),
        out,
        tc["workspace"],
    )
    return out
