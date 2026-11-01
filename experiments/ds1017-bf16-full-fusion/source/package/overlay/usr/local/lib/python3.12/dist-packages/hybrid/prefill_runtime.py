"""Prepared eager MoE replacement. KV and all decode entry points stay unchanged."""
from collections import Counter
from .prefill_contract import MOE_TRY, FEATURES, RAGGED_PROFILES, Selection, eligible_geometry

BANK = None

def dense_candidate(layer, x, bias):
    return None if BANK is None else BANK.dense_apply(layer, x, bias)

def ragged_candidate(*args, num_rows):
    return None if BANK is None else BANK.ragged_apply(*args, num_rows=num_rows)

def combine_candidate(*args):
    return None if BANK is None else BANK.combine_apply(*args)


def binding(tc):
    return tuple((name, tc[name].data_ptr(), tc[name].numel()) for name in ("ptrs", "meta", "emap", "workspace")) + (
        ("geometry", tc["intermediate"], tc["bits_x2"], tc["prefill_rows"]),)


class NativeAdapter:
    """The candidate's out/workspace/limit ABI differs from the incumbent hook."""
    def __init__(self, torch, ops):
        self.torch, self.ops = torch, ops

    def apply(self, x, ids, weights, tc, expert_map, limit):
        t = self.torch
        out = t.empty_like(x)
        route_weights = weights if weights.dtype == t.float32 else weights.float()
        self.ops.moe_out_prefill_try(
            x.contiguous(), ids, route_weights.contiguous(), tc["ptrs"], tc["meta"],
            tc["emap"] if expert_map is None else expert_map,
            tc["intermediate"], tc["bits_x2"], out, tc["workspace"], float(limit or 0.0))
        return out


class Bank:
    def __init__(self, worker, original, torch, adapter, bindings, stream, mode, identity, receipt,
                 available=None, combiner=None, dense=None, ragged=None):
        self.worker, self.original, self.torch, self.adapter = worker, original, torch, adapter
        self.bindings, self.stream, self.mode = bindings, stream, mode
        self.selection = Selection(identity, True, available=available)
        self.combiner, self.dense, self.ragged = combiner, dense, ragged
        self.receipt, self.hits, self.in_call = receipt, Counter(), False

    def abc_session(self):
        abc = getattr(self.worker, "_abc_bank", None)
        return None if abc is None or abc.selection is None else abc.selection.session

    def status(self):
        return dict(rank=int(self.worker.rank), ok=True, ready=True,
                    **self.selection.record(self.abc_session()), hits=dict(self.hits),
                    receipt=self.receipt, model_quality_verified=False)

    def apply(self, x, ids, weights, layer, expert_map, limit):
        if not FEATURES[self.selection.effective(self.abc_session())][0]:
            return self.original(x, ids, weights, layer, expert_map, limit)
        t, tc = self.torch, getattr(layer, "_exl3_tc", None)
        reason = None
        if t.cuda.is_current_stream_capturing() or self.mode() != "NONE":
            reason = "capture_or_graph_mode"
        elif tc is None or id(layer) not in self.bindings:
            reason = "unprepared_layer"
        elif not (x.dim() == 2 and x.is_cuda and x.dtype == t.bfloat16 and ids.dim() == 2):
            reason = "format"
        elif not eligible_geometry(int(x.shape[0]), int(x.shape[1]), int(ids.shape[1]),
                                   tc["intermediate"], tc["bits_x2"], int(tc["ptrs"].shape[0])):
            reason = "geometry"
        elif not tc["prefill"] or tc["prefill_rows"] < int(x.shape[0]):
            reason = "incumbent_prefill_disabled"
        elif not ids.is_contiguous() or (expert_map is not None and not expert_map.is_contiguous()):
            reason = "noncontiguous_routes"
        elif (tc["emap"] if expert_map is None else expert_map).device != x.device:
            reason = "expert_map_device"
        elif int(t.cuda.current_stream(x.device).cuda_stream) != self.stream:
            reason = "different_stream"
        if reason is not None:
            self.hits["fallback:" + reason] += 1
            return self.original(x, ids, weights, layer, expert_map, limit)
        # A prepared pointer/workspace mutation is a broken invariant, not a silent fallback.
        if binding(tc) != self.bindings[id(layer)]:
            raise RuntimeError("Prepared prefill tensors/workspace moved")
        if self.in_call:
            raise RuntimeError("Reentrant use of the shared prefill workspace")
        self.in_call = True
        try:
            out = self.adapter.apply(x, ids, weights, tc, expert_map, limit)
            self.hits[f"moe_try:I{tc['intermediate']}:K{tc['bits_x2']}:M{int(x.shape[0])}"] += 1
            return out
        finally:
            self.in_call = False

    def combine_apply(self, *args):
        if not FEATURES[self.selection.effective(self.abc_session())][1] or self.combiner is None:
            return None
        if self.torch.cuda.is_current_stream_capturing() or self.mode() != 'NONE':
            self.hits['combine_fallback:capture_or_graph_mode'] += 1
            return None
        if not self.combiner.supported(*args):
            self.hits['combine_fallback:unsupported'] += 1
            return None
        # These are the exact model dtype/layout/constexpr signatures warmed by
        # all-rank qualification. Other signatures keep the retained function.
        top,qsl,seq,gather,window,ratio,topk,_,_=args
        if (any(v.dtype!=self.torch.int32 for v in (top,qsl,seq,gather))
                or top.ndim!=2 or top.shape[1]!=512 or not top.is_contiguous()
                or window!=128 or topk!=512):
            self.hits['combine_fallback:unprepared_signature'] += 1
            return None
        if int(self.torch.cuda.current_stream(args[0].device).cuda_stream) != self.stream:
            self.hits['combine_fallback:different_stream'] += 1
            return None
        out = self.combiner.combine_topk_swa_indices(*args)
        self.hits[f'combine:M{args[0].shape[0]}:R{args[2].numel()}:ratio{args[5]}'] += 1
        return out

    def dense_apply(self, layer, x, bias):
        policy = FEATURES[self.selection.effective(self.abc_session())][2]
        if policy is None or self.dense is None:
            return None
        if self.torch.cuda.is_current_stream_capturing() or self.mode() != 'NONE':
            self.hits['dense_fallback:capture_or_graph_mode'] += 1
            return None
        if not self.dense.eligible(layer, x, bias):
            self.hits['dense_fallback:unsupported'] += 1
            return None
        if int(self.torch.cuda.current_stream(x.device).cuda_stream) != self.stream:
            self.hits['dense_fallback:different_stream'] += 1
            return None
        if self.in_call:
            raise RuntimeError('Reentrant dense use of shared prefill workspace')
        self.in_call = True
        try:
            out = self.dense.apply(layer, x, policy=policy)
            rows = x.numel() // x.shape[-1]
            if out is None:
                self.hits[f'dense_tuned_marlin:M{rows}'] += 1
            else:
                self.hits[f'dense_cublas:N{layer.output_size_per_partition}:K{layer.input_size_per_partition}:M{rows}'] += 1
            return out
        finally:
            self.in_call = False

    def ragged_apply(self, *args, num_rows):
        if self.selection.effective(self.abc_session()) not in RAGGED_PROFILES or self.ragged is None:
            return None
        t=self.torch
        if t.cuda.is_current_stream_capturing() or self.mode()!='NONE':
            self.hits['ragged_fallback:capture_or_graph_mode']+=1;return None
        if not self.ragged.supported(*args,num_rows=num_rows):
            self.hits['ragged_fallback:unsupported']+=1;return None
        top,qsl,seq,gather,window,ratio,topk,_,_=args
        if (any(v.dtype!=t.int32 for v in (top,qsl,seq,gather))
                or top.ndim!=2 or top.shape[1]!=512 or not top.is_contiguous()
                or window!=128 or not ((ratio==0 and topk==0) or (ratio in (1,2) and topk==512))):
            self.hits['ragged_fallback:unprepared_signature']+=1;return None
        if int(t.cuda.current_stream(top.device).cuda_stream)!=self.stream:
            self.hits['ragged_fallback:different_stream']+=1;return None
        if self.in_call:raise RuntimeError('Reentrant prefill integer preparation')
        self.in_call=True
        try:
            output=self.ragged.combine_ragged(*args,num_rows=num_rows)
            self.hits[f'ragged:M{top.shape[0]}:R{seq.numel()}:ratio{ratio}']+=1
            return output
        finally:self.in_call=False
