"""R2 per-descriptor private FULL graphs for B2..B8 and their serving dispatch.

Imported only when DSV41_R2=1 (r2_startup). Every hook elsewhere looks this module up
in sys.modules, so with R2 off nothing here runs and every R1 path is unchanged.

One R2 graph per served descriptor (6B, B, True, False, 0), captured at startup in one
R2-private pool (largest first; graphs replay serially on the serving stream), each with
its own attention workspace, the planned routed-expert kernel and the planned all-reduce
geometry. Components inside a graph cannot count themselves at replay (no Python runs),
so each graph carries a capture receipt per component and every replay credits the
receipt's components to the round's real B in the ledger.
"""
import contextlib
import sys

from .r2_contract import PAIR, TARGET_COMPONENTS, Ledger, ar_bytes
from .w26_dual_graph import Runtime, descriptor

BANK = None            # published R2Bank (serving)
CAPTURING = None       # R2Runtime during startup capture/qualification
ACTIVE = None          # CaptureContext while a private R2 capture is recording
DUMMY_ROWS = ()        # row counts the runner's target-bank dummy guard accepts (startup only)
TRACE = None           # r2_contract.TraceBudget when DSV41_R2_TRACE_BUDGET and a trace mode are set


class CaptureContext:
    """Per-capture receipts for the collective and expert components."""

    def __init__(self, plan):
        self.plan = plan
        self.experts = {}
        self.collective = {"calls": 0, "planned": 0, "unavailable": 0, "other_sizes": {}}

    def _planned_size(self, inp):
        return (str(inp.dtype) == "torch.bfloat16" and inp.numel() * inp.element_size() == ar_bytes(self.plan.served_b)
                and "collective" in self.plan.components)

    def collective_select(self, inp, blocks, threads, algo):
        self.collective["calls"] += 1
        if self._planned_size(inp):
            if self.plan.collective == PAIR:
                self.collective["unavailable"] += 1
                return blocks, threads, algo
            self.collective["planned"] += 1
            return self.plan.collective
        key = str(inp.numel() * inp.element_size())
        self.collective["other_sizes"][key] = self.collective["other_sizes"].get(key, 0) + 1
        return blocks, threads, algo

    def pair_reduce(self, owner, inp):
        """W19 pair kernel for the planned 48-row size, launched on the current (capturing)
        stream exactly as PairState.reduce does, without its PROFILE route check: the R2
        graph owns its selection. None (host path continues) when not planned or not ready."""
        if self.plan.collective != PAIR or not self._planned_size(inp):
            return None
        from w9_exl_ab import pair_transport as PT
        from w9_exl_ab.pair_arena import BLOCKS, NBYTES, THREADS
        from w9_exl_ab.pair_policy import shape_allowed
        import torch
        s = PT.state(owner)
        if not (s is not None and s.ready and inp.is_cuda and inp.device == owner.device
                and shape_allowed(tuple(inp.shape), inp.numel()) and inp.numel() * 2 == NBYTES):
            return None
        self.collective["calls"] += 1
        self.collective["planned"] += 1
        s.check()
        x = owner._aligned(inp.contiguous())
        out = torch.empty_like(x)
        a = s.arena
        s.driver.launch(s.small_fn, x.data_ptr(), out.data_ptr(), a.dev_base, a.seq.data_ptr(), a.rank,
                        NBYTES, a.slot_bytes, a.spin_cycles, torch.cuda.current_stream(x.device).cuda_stream,
                        BLOCKS, THREADS)
        return out

    def collective_receipt(self):
        c = self.collective
        profile = self.plan.collective if self.plan.collective == PAIR else list(self.plan.collective)
        return dict(c, profile=profile, bytes=ar_bytes(self.plan.served_b),
                    ok=c["planned"] > 0 and c["unavailable"] == 0)


def pair_ready(owner, rows):
    """Local readiness of the W19 pair route for a rows x 5120 bf16 all-reduce (pair_reduce's
    own condition minus the tensor). r2_startup votes it on every rank before any pair use."""
    from w9_exl_ab import pair_transport as PT
    from w9_exl_ab.pair_arena import NBYTES
    from w9_exl_ab.pair_policy import shape_allowed
    s = PT.state(owner)
    n = rows * 5120
    return bool(s is not None and s.ready and shape_allowed((rows, 5120), n) and n * 2 == NBYTES)


def collective_path(receipt):
    """Rank-comparable part of a collective capture receipt (which protocol each call took)."""
    return {k: receipt.get(k) for k in ("calls", "planned", "unavailable", "other_sizes", "profile", "bytes")}


def collective_hook(inp, blocks, threads, algo):
    """w9_exl_ab.collective_profiles.selected() entry; identity outside R2 captures."""
    ctx = ACTIVE
    if ctx is None:
        return None
    return ctx.collective_select(inp, blocks, threads, algo)


def pair_hook(owner, inp):
    """host_shm_all_reduce.host_all_reduce entry; None outside R2 captures."""
    ctx = ACTIVE
    if ctx is None:
        return None
    return ctx.pair_reduce(owner, inp)


class R2Runtime(Runtime):
    """w26 Runtime for one served descriptor; private pool shared by all R2 graphs."""

    def __init__(self, runner, native_contract, before_free, pool, plan, attention_state):
        super().__init__(runner, None, native_contract, before_free)
        self.pool = pool
        self.plan = plan
        self.rows = plan.served_by[0]
        self.capture_desc = plan.served_by
        self.attention_state = attention_state
        self.context = None

    def guard(self, reserve=0):
        # Incumbent-entry identity as in Runtime.guard; the memory bound for all R2
        # graphs together is r2_startup's (DSV41_R2_CAP_MIB), not the B1 256 MiB cap.
        self.torch.cuda.synchronize()
        free = self.torch.cuda.mem_get_info(self.runner.device)[0]
        assert self.baseline_entries == {k: id(v) for k, v in self.wrapper.entries.items()}, "Incumbent entries changed"
        return dict(free=free, physical_growth=max(0, self.before_free - free))

    def capture_at_call(self, args, kwargs):
        global ACTIVE
        from . import r2_experts
        self.context = CaptureContext(self.plan)
        ACTIVE = self.context
        r2_experts.ACTIVE = self.context
        try:
            return super().capture_at_call(args, kwargs)
        finally:
            ACTIVE = None
            r2_experts.ACTIVE = None

    def dummy(self, mode, *, target_only=False):
        from vllm.config import CUDAGraphMode
        from .initial_decode_contract import initial_decode
        kwargs = dict(w26_target_bank_only=True) if target_only else {}
        context = (initial_decode(self.runner, self.initial_consensus, self.initial_metadata_records, b=self.plan.served_b)
                   if target_only else contextlib.nullcontext())
        with context:
            return self.runner._dummy_run(
                self.rows, cudagraph_runtime_mode=getattr(CUDAGraphMode, mode), force_attention=True,
                uniform_decode=True, uniform_query_len=6, allow_microbatching=False, skip_eplb=True,
                remove_lora=False, num_active_loras=0, is_graph_capturing=mode == "FULL", **kwargs)

    def receipts(self):
        from . import r2_experts
        out = {"collective": self.context.collective_receipt(),
               "experts": r2_experts.receipt(self.context.experts, self.plan)}
        if self.attention_state is not None:
            st = self.attention_state
            out["attention"] = dict(calls=self.attention_calls, layers=list(st.layers), rejections=list(st.rejections),
                                    profile=st.profile.record(),
                                    ok=self.attention_calls == 38 and not st.rejections and len(st.layers) == 38)
        return out


class R2Bank:
    """Published per-worker R2 state: graphs, samplers, receipts and the hit ledger."""

    def __init__(self, contract, adapter, wrapper, runner, vocab):
        self.contract = contract
        self.adapter = adapter
        self.wrapper = wrapper
        self.runner = runner
        self.vocab = vocab
        self.entries = {}          # served descriptor -> entry (graph, output, addresses, components)
        self.excluded = {}         # served descriptor -> {component: reason}
        self.greedy = {}
        self.sampled = {}
        self.receipts = {}
        self.ledger = Ledger()

    # -- startup
    def publish(self, desc, entry, components, excluded):
        entry = dict(entry, components=tuple(components))
        self.entries[tuple(desc)] = entry
        self.excluded[tuple(desc)] = dict(excluded)

    def sampler_ready(self, b, sampled):
        table = self.sampled if sampled else self.greedy
        state = table.get(b)
        return state is not None and state.ready

    def record(self):
        return dict(contract=self.contract.digest(),
                    graphs={str(list(d)): dict(components=list(e["components"]), excluded=self.excluded[d])
                            for d, e in sorted(self.entries.items())},
                    greedy=sorted(self.greedy), sampled=sorted(self.sampled))

    # -- serving
    def _miss(self, b, reason):
        self.ledger.fallback(b, "graph", reason)
        for comp in TARGET_COMPONENTS:
            self.ledger.fallback(b, comp, reason)

    def intercept(self, wrapper, ctx, args, kwargs):
        """(handled, output) for R2 shapes, else None so the caller continues unchanged."""
        rt = CAPTURING
        if ctx.cudagraph_runtime_mode.name != "FULL":
            return None
        desc = descriptor(ctx.batch_descriptor)
        if rt is not None and wrapper is rt.wrapper and desc == rt.capture_desc:
            if rt.mode == "capture":
                return True, rt.capture_at_call(args, kwargs)
            if rt.mode == "qualify_replay":
                return True, rt.replay(rt.entry, args, kwargs)
            return None
        if wrapper is not self.wrapper or not (desc[2] is True and desc[3] is False and desc[4] == 0
                                               and desc[1] >= 2 and desc[0] == 6 * desc[1]):
            return None
        real_b = int(self.runner.input_batch.num_reqs)
        plan = self.contract.serves(desc, real_b)
        if plan is None:
            if 2 <= real_b <= 8:
                self._miss(real_b, "not_planned")
            return None
        entry = self.entries.get(plan.served_by)
        if entry is None:
            self._miss(real_b, "not_captured")
            return None
        if not self.adapter.r2_allowed("FULL"):
            self._miss(real_b, "arm_not_selected")
            return None
        out = self.wrapper_replay(entry, args, kwargs)
        self.ledger.hit(real_b, "graph")
        for comp in TARGET_COMPONENTS:
            if comp in entry["components"]:
                self.ledger.hit(real_b, comp)
            else:
                self.ledger.fallback(real_b, comp, self.excluded[plan.served_by].get(comp, "excluded"))
        return True, out

    def wrapper_replay(self, entry, args, kwargs):
        from vllm.model_executor.offloader.base import get_offloader
        assert entry["addresses"] == self.wrapper._collect_tensor_addresses(args, kwargs)
        get_offloader().sync_prev_onload()
        entry["graph"].replay()
        return entry["output"]


def intercept(wrapper, ctx, args, kwargs):
    bank = BANK
    if bank is None:
        rt = CAPTURING
        if rt is None:
            return None
        return _startup_intercept(rt, wrapper, ctx, args, kwargs)
    return bank.intercept(wrapper, ctx, args, kwargs)


def _startup_intercept(rt, wrapper, ctx, args, kwargs):
    if ctx.cudagraph_runtime_mode.name != "FULL" or wrapper is not rt.wrapper:
        return None
    if descriptor(ctx.batch_descriptor) != rt.capture_desc:
        return None
    if rt.mode == "capture":
        return True, rt.capture_at_call(args, kwargs)
    if rt.mode == "qualify_replay":
        return True, rt.replay(rt.entry, args, kwargs)
    return None


def module():
    """The loaded module when R2 is active, else None (no import side effects)."""
    return sys.modules.get(__name__)
