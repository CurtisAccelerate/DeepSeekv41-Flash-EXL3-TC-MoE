"""Reload R1: request-gated sampled DSpark drafting, g logger and step trace. Default OFF.

The runner imports this module only when the boot env ``DSV41_R1_FEATURES=1`` is set (the same
environment reaches every worker). Without it nothing here is imported and every path is the
incumbent one. Even when armed, the drafter follows the serving split. One unseeded request
with temperature above zero drafts at T_q=1. A batch of two or more stays
greedy, and so does temperature zero, a seed, or structured output. The
request flag does not override that split. A step that still needs nothing
builds no plan, and the drafter, bank, rejection sampler and graphs run
exactly as before.

Every runtime switch rides on the request: ``SamplingParams.extra_args`` (OpenAI body field
``vllm_xargs``). The scheduler output that carries the sampling params is broadcast as one
message to every worker, so all TP ranks build the same plan from the same values. Never read a
per-rank file or a per-rank env var to decide drafting: a rank that drafts differently corrupts
the verify batch.

Request flags (``extra_args``):
  dsv41_draft_tq    float, 0 < T_q <= 100   sampled drafting at draft temperature T_q
  dsv41_glog        1                       proposal-overlap logger (gate G3)
  dsv41_glog_every  int >= 1 (default 8)    positions >= 1 logged every Nth verify step
  dsv41_trace       1 or 2                  per-step GPU timing (gate G2); 2 also dumps kernels
  dsv41_xcheck      1                       debug: all-gather a hash of drafts and q over TP

Boot env (identical on all ranks): DSV41_R1_FEATURES=1 arms the code; DSV41_R1_OUT=<dir>
enables the JSONL outputs (unset: glog and trace are disabled and their flags are ignored).
DSV41_R1_ROUNDLOG=1 (with DSV41_R1_OUT) adds the all-traffic round log and q-path attestation
(dsv41_r1_round). Trace-only boot env: DSV41_R1_TRACE_MODE, DSV41_R1_CUPTI_LIB (see
dsv41_r1_trace).
"""
from __future__ import annotations

import collections
import logging
import math
import os
import sys
from dataclasses import dataclass

try:  # the pure-Python parts (flags, plans) are usable without torch
    import torch
except ImportError:  # pragma: no cover - CPU test hosts without torch
    torch = None

logger = logging.getLogger("vllm.dsv41.r1")

VERSION = "dsv41-r1-v1"
ARMED = os.environ.get("DSV41_R1_FEATURES") == "1"
OUT_DIR = os.environ.get("DSV41_R1_OUT") or None
ROUNDLOG = os.environ.get("DSV41_R1_ROUNDLOG") == "1"

KEY_TQ = "dsv41_draft_tq"
KEY_GLOG = "dsv41_glog"
KEY_GLOG_EVERY = "dsv41_glog_every"
KEY_TRACE = "dsv41_trace"
KEY_XCHECK = "dsv41_xcheck"

TQ_MAX = 100.0
GLOG_EVERY_DEFAULT = 8
GLOG_EVERY_MAX = 1 << 20
SAMPLING_EPS = 1e-5  # vllm.sampling_params._SAMPLING_EPS: below it a request is greedy
EXPECTED_VOCAB = 129280
NOISE_SEED_BASE = 0x5D41D5A1
GLOG_GRID = (0.6, 0.8, 1.0, 1.2)


# ---------------------------------------------------------------- request flags
def _number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        x = float(value)
    elif isinstance(value, str):
        try:
            x = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return x if math.isfinite(x) else None


def parse_tq(extra) -> float | None:
    """Draft temperature, or None (greedy drafting) when absent or invalid."""
    if not isinstance(extra, dict):
        return None
    x = _number(extra.get(KEY_TQ))
    if x is None or x <= 0.0 or x > TQ_MAX:
        return None
    return x


def parse_level(extra, key: str, max_level: int = 1) -> int:
    """0 when absent or invalid, else an int level in [1, max_level]."""
    if not isinstance(extra, dict):
        return 0
    value = extra.get(key)
    if value is True:
        return 1
    if isinstance(value, str) and value.strip().lower() in ("true", "yes", "on"):
        return 1
    x = _number(value)
    if x is None or x < 1.0:
        return 0
    return int(min(math.floor(x), max_level))


def parse_int(extra, key: str, default: int, lo: int, hi: int) -> int:
    if not isinstance(extra, dict):
        return default
    x = _number(extra.get(key))
    if x is None:
        return default
    return int(min(max(math.floor(x), lo), hi))


@dataclass(frozen=True)
class RequestFlags:
    tq: float | None = None
    glog: bool = False
    glog_every: int = GLOG_EVERY_DEFAULT
    trace: int = 0
    xcheck: bool = False

    @property
    def any(self) -> bool:
        return self.tq is not None or self.glog or self.trace > 0 or self.xcheck


NO_FLAGS = RequestFlags()


def flags_from_extra(extra) -> RequestFlags:
    if not isinstance(extra, dict) or not extra:
        return NO_FLAGS
    flags = RequestFlags(
        tq=parse_tq(extra),
        glog=parse_level(extra, KEY_GLOG) > 0,
        glog_every=parse_int(extra, KEY_GLOG_EVERY, GLOG_EVERY_DEFAULT, 1, GLOG_EVERY_MAX),
        trace=parse_level(extra, KEY_TRACE, 2),
        xcheck=parse_level(extra, KEY_XCHECK) > 0,
    )
    return flags if flags.any else NO_FLAGS


class FlagCache:
    """Parse each request's extra_args once (keyed by request id and params identity)."""

    def __init__(self, max_entries: int = 4096):
        self._cache: dict[str, tuple[object, RequestFlags]] = {}
        self._max = max_entries

    def get(self, req_id, sampling_params) -> RequestFlags:
        if sampling_params is None:
            return NO_FLAGS
        hit = self._cache.get(req_id)
        if hit is not None and hit[0] is sampling_params:
            return hit[1]
        flags = flags_from_extra(getattr(sampling_params, "extra_args", None))
        if len(self._cache) >= self._max:
            self._cache.clear()
        self._cache[req_id] = (sampling_params, flags)
        return flags


def row_kind(sampling_params, has_generator: bool) -> str:
    """greedy | seeded | structured | random (only 'random' rows may sample drafts)."""
    sp = sampling_params
    if sp is None:
        return "greedy"
    t = _number(getattr(sp, "temperature", 0.0))
    if t is None or t < SAMPLING_EPS:
        return "greedy"
    if has_generator or getattr(sp, "seed", None) is not None:
        return "seeded"
    if (
        getattr(sp, "structured_outputs", None) is not None
        or getattr(sp, "guided_decoding", None) is not None
    ):
        return "structured"
    return "random"


def p_is_approx(sp) -> bool:
    """True when the target distribution has processors that the g logger does not replay."""
    if sp is None:
        return True
    top_k = getattr(sp, "top_k", 0) or 0
    return bool(
        getattr(sp, "top_p", 1.0) < 1.0
        or top_k > 0
        or getattr(sp, "min_p", 0.0) > 0.0
        or getattr(sp, "presence_penalty", 0.0) != 0.0
        or getattr(sp, "frequency_penalty", 0.0) != 0.0
        or getattr(sp, "repetition_penalty", 1.0) != 1.0
        or getattr(sp, "bad_words", None)
        or getattr(sp, "allowed_token_ids", None)
        or getattr(sp, "logit_bias", None)
        or (getattr(sp, "min_tokens", 0) or 0) > 0
        or getattr(sp, "logits_processors", None)
        or getattr(sp, "thinking_token_budget", None) is not None
    )


# ---------------------------------------------------------------- draft plan
@dataclass
class DraftPlan:
    """Per-propose plan, identical on every rank (built from broadcast request state)."""

    req_ids: list
    tq: list  # per row: T_q > 0 samples, 0.0 drafts greedy
    kinds: list
    glog_rows: list
    any_sample: bool
    all_sample: bool
    any_glog: bool
    xcheck: bool
    mark: object = None  # trace marker callable or None
    vocab_expected: int | None = None

    @property
    def num_rows(self) -> int:
        return len(self.req_ids)

    @property
    def needs_r1_path(self) -> bool:
        return self.any_sample or self.any_glog


def build_draft_plan(
    req_ids,
    sampling_params,
    generator_rows,
    flag_cache: FlagCache,
    *,
    glog_enabled: bool,
    mark=None,
    vocab_expected=None,
):
    """None when no row needs anything (the drafter then runs its unmodified path)."""
    gen_rows = set(generator_rows)
    tq, kinds, glog_rows = [], [], []
    xcheck = False
    for row, (req_id, sp) in enumerate(zip(req_ids, sampling_params)):
        flags = flag_cache.get(req_id, sp)
        kind = row_kind(sp, row in gen_rows)
        kinds.append(kind)
        # One unseeded temperature>0 request samples at T_q=1. Two or more
        # requests stay greedy even if the client sent dsv41_draft_tq.
        # Temperature 0, seeded, and structured rows stay greedy.
        tq.append(1.0 if (kind == "random" and len(req_ids) == 1) else 0.0)
        glog_rows.append(bool(glog_enabled and flags.glog and kind != "greedy"))
        xcheck = xcheck or flags.xcheck
    any_sample = any(t > 0.0 for t in tq)
    any_glog = any(glog_rows)
    if not (any_sample or any_glog or xcheck or mark is not None):
        return None
    return DraftPlan(
        req_ids=list(req_ids),
        tq=tq,
        kinds=kinds,
        glog_rows=glog_rows,
        any_sample=any_sample,
        all_sample=any_sample and all(t > 0.0 for t in tq),
        any_glog=any_glog,
        xcheck=xcheck,
        mark=mark,
        vocab_expected=vocab_expected,
    )


_warned: set = set()


def warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg, *args)


def sampling_supported(proposer, vocab: int, plan: DraftPlan) -> bool:
    """Shape and mapping checks; identical on every rank (they only read config/shapes)."""
    if proposer._draft_topk is not None:
        warn_once("topk", "R1: draft top-k path active; sampled drafting disabled")
        return False
    if getattr(proposer.model, "draft_id_to_target_id", None) is not None:
        warn_once("idmap", "R1: draft->target id map present; sampled drafting disabled")
        return False
    if plan.vocab_expected is None or vocab != plan.vocab_expected:
        warn_once(
            "vocab",
            "R1: draft vocab %s != target logits width %s; sampled drafting disabled",
            vocab, plan.vocab_expected,
        )
        return False
    if vocab != EXPECTED_VOCAB:
        warn_once("vocab129280", "R1: vocab %d (expected %d); q width still matches target",
                  vocab, EXPECTED_VOCAB)
    return True


def noise_seed(vllm_config) -> int:
    seed = getattr(getattr(vllm_config, "model_config", None), "seed", None) or 0
    return (NOISE_SEED_BASE ^ int(seed)) & 0x7FFFFFFFFFFFFFFF


def tq_tensor(plan: DraftPlan, device):
    t = torch.tensor(plan.tq, dtype=torch.float32)
    if torch.device(device).type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def draft_chain(
    model,
    base_logits,
    prev,
    tq=None,
    generator=None,
    *,
    all_sample: bool = False,
    keep_logits: bool = False,
    keep_embeds: bool = False,
):
    """Sequential Markov drafting over [B, n, V] base logits (the incumbent loop, extended).

    Greedy rows (tq == 0, or tq None) take exactly the incumbent expressions on the same tensors.
    Rows with tq > 0 draw tok = argmax(q / Exp(1)) with q = softmax(logits_i.float() / tq), the
    exported q row, and feed the sampled token to the next position's Markov embedding.
    Returns (drafts int64 [B, n], q fp32 [B, n, V] or None, logits [B, n, V] or None, embeds).
    """
    num_reqs, n, vocab = base_logits.shape
    device = base_logits.device
    drafts = torch.empty(num_reqs, n, dtype=torch.int64, device=device)
    q_out = logits_out = None
    embeds = []
    if tq is not None:
        assert tq.shape == (num_reqs,), (tuple(tq.shape), num_reqs)
        mask = tq > 0
        denom = torch.where(mask, tq, torch.ones_like(tq)).unsqueeze(1)
        mask_col = mask.unsqueeze(1)
        q_out = torch.empty(num_reqs, n, vocab, dtype=torch.float32, device=device)
    if keep_logits:
        logits_out = torch.empty(num_reqs, n, vocab, dtype=base_logits.dtype, device=device)
    for i in range(n):
        markov_embed = model.markov_embed(prev)
        if keep_embeds:
            embeds.append(markov_embed)
        logits_i = base_logits[:, i] + model.markov_bias(markov_embed)
        tok = model.map_draft_to_target(logits_i.argmax(dim=-1))
        if tq is not None:
            q = torch.softmax(logits_i.float() / denom, dim=-1)
            noise = torch.empty_like(q).exponential_(generator=generator)
            sampled = (q / noise).argmax(dim=-1)
            if all_sample:
                tok = sampled
                q_out[:, i] = q
            else:
                tok = torch.where(mask, sampled, tok)
                one_hot = torch.zeros_like(q).scatter_(1, tok.unsqueeze(1), 1.0)
                q_out[:, i] = torch.where(mask_col, q, one_hot)
        if logits_out is not None:
            logits_out[:, i] = logits_i
        drafts[:, i] = tok
        prev = tok
    return drafts, q_out, logits_out, embeds


# ---------------------------------------------------------------- cross-rank debug check
_XCHECK = {"n": 0, "bad": 0}


def draft_hash(drafts, q, device):
    """[1, 4] int64: weighted sums of draft ids and of q's fp32 bit patterns (wrapping)."""
    if isinstance(drafts, list):
        lens = [len(r) for r in drafts]
        flat = [t for r in drafts for t in r] + lens
        d = torch.tensor(flat or [0], dtype=torch.int64, device=device)
    else:
        d = drafts.reshape(-1).to(torch.int64)
    h = torch.zeros(1, 4, dtype=torch.int64, device=device)
    w = torch.arange(1, d.numel() + 1, dtype=torch.int64, device=device) * 1000003
    h[0, 0] = (d * w).sum()
    h[0, 1] = d.numel()
    if q is not None:
        rows = q.reshape(-1, q.shape[-1])
        bits = rows.contiguous().view(torch.int32).to(torch.int64)
        wv = (torch.arange(rows.shape[-1], dtype=torch.int64, device=device) * 2654435761) % 2147483647 + 1
        wr = torch.arange(1, rows.shape[0] + 1, dtype=torch.int64, device=device)
        h[0, 2] = (bits * wv).sum()
        h[0, 3] = (bits.sum(dim=-1) * wr).sum()
    return h


def xcheck(proposer, drafts, q, group=None) -> bool:
    """Collective: every TP rank must call this at the same point (plan.xcheck is identical)."""
    if group is None:
        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group()
    h = draft_hash(drafts, q, proposer.device)
    rows = group.all_gather(h, dim=0).cpu().tolist()
    ok = all(r == rows[0] for r in rows)
    _XCHECK["n"] += 1
    if not ok:
        _XCHECK["bad"] += 1
        logger.error("R1 xcheck: draft/q hash mismatch across TP ranks: %s", rows)
    elif _XCHECK["n"] % 256 == 1:
        logger.info("R1 xcheck: %d checks, %d mismatches", _XCHECK["n"], _XCHECK["bad"])
    return ok


# ---------------------------------------------------------------- g logger (gate G3)
def glog_columns(grid=GLOG_GRID) -> list:
    cols = []
    for t in grid:
        cols += [f"smin_{t}", f"spq_{t}", f"maxq_{t}", f"hq_{t}", f"q_d_{t}"]
    cols += ["p_d", "p_argmax_q", "max_p", "h_p", "d_is_argmax", "smin_act", "q_d_act"]
    return cols


def glog_stats(p_logits, temperature, q_logits, d, tq_act, grid=GLOG_GRID):
    """[M, F] fp32 in glog_columns() order.

    p_logits [M, V] fp32 target logits (grammar mask already applied), temperature [M] > 0,
    q_logits [M, V] fp32 drafter logits at that position, d [M] int64 drafted id (-1 allowed),
    tq_act [M] draft temperature actually used (0 = greedy drafting).
    """
    logp = torch.log_softmax(p_logits / temperature.unsqueeze(1), dim=-1)
    p = logp.exp()
    zero = torch.zeros_like(p)
    h_p = -torch.where(p > 0, p * logp, zero).sum(-1)
    max_p = p.max(dim=-1).values
    am = q_logits.argmax(dim=-1)
    p_am = p.gather(1, am.unsqueeze(1)).squeeze(1)
    valid = d >= 0
    dd = d.clamp(min=0).unsqueeze(1)
    nan = torch.full_like(max_p, float("nan"))
    p_d = torch.where(valid, p.gather(1, dd).squeeze(1), nan)
    cols = []
    for t in grid:
        logq = torch.log_softmax(q_logits / t, dim=-1)
        q = logq.exp()
        cols += [
            torch.minimum(p, q).sum(-1),
            (p * q).sum(-1),
            q.max(dim=-1).values,
            -torch.where(q > 0, q * logq, zero).sum(-1),
            torch.where(valid, q.gather(1, dd).squeeze(1), nan),
        ]
    act = tq_act > 0
    ta = torch.where(act, tq_act, torch.ones_like(tq_act))
    qa = torch.softmax(q_logits / ta.unsqueeze(1), dim=-1)
    smin_act = torch.where(act, torch.minimum(p, qa).sum(-1), nan)
    q_d_act = torch.where(act & valid, qa.gather(1, dd).squeeze(1), nan)
    cols += [p_d, p_am, max_p, h_p, (d == am).float(), smin_act, q_d_act]
    return torch.stack(cols, dim=1).float()


def glog_records(values, meta, cols) -> list:
    out = []
    for vals, m in zip(values, meta):
        rec = dict(m)
        for c, v in zip(cols, vals):
            if c == "d":
                rec[c] = int(v)
            else:
                rec[c] = None if v != v else round(float(v), 7)
        if rec.get("mode") == "greedy" and rec.get("d", -1) >= 0 and rec.get("d_is_argmax") == 0.0:
            rec["argmax_violation"] = True
        out.append(rec)
    return out


def _dist_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:  # pragma: no cover
        pass
    return 0


# ---------------------------------------------------------------- runner glue
class RunnerHooks:
    """Owned by GPUModelRunner when armed. All entry points are cheap no-ops without flags."""

    def __init__(self, runner, out_dir=None, rank=None, roundlog=None):
        self.runner = runner
        self.flags = FlagCache()
        self.out_dir = out_dir if out_dir is not None else OUT_DIR
        self.glog_enabled = self.out_dir is not None
        self.logits_width: int | None = None
        self._rank = rank
        self._glog_pending = None
        self._glog_steps: dict = {}
        self._glog_inflight = collections.deque()
        self._glog_sink = None
        self._glog_cols = glog_columns() + ["d"]
        self._tracer = None
        self._tracer_failed = False
        self._trace_on = False
        self._step_no = 0
        self._req_sink = None
        self._req_done: set = set()
        self._round = None
        roundlog = ROUNDLOG if roundlog is None else roundlog
        if roundlog and self.out_dir is None:
            logger.warning("R1: DSV41_R1_ROUNDLOG=1 needs DSV41_R1_OUT; round log disabled")
        elif roundlog:
            from hybrid.dsv41_r1_round import RoundLog

            self._round = RoundLog(runner, self.out_dir, lambda: self.rank)
        logger.info("DSV41 R1 armed (%s): out=%s roundlog=%s", VERSION, self.out_dir,
                    self._round is not None)

    @property
    def rank(self) -> int:
        if self._rank is None:
            self._rank = _dist_rank()
        return self._rank

    def _flags_for(self, req_id):
        st = self.runner.requests.get(req_id)
        return self.flags.get(req_id, st.sampling_params if st is not None else None)

    # ---- trace (gate G2)
    def _get_tracer(self):
        if self._tracer is None and not self._tracer_failed:
            try:
                from hybrid import dsv41_r1_trace

                self._tracer = dsv41_r1_trace.Tracer(
                    rank=self.rank, out_dir=self.out_dir, device=self.runner.device
                )
            except Exception:
                self._tracer_failed = True
                logger.exception("R1 trace unavailable; dsv41_trace flags ignored")
        return self._tracer

    def step_begin(self, scheduler_output) -> None:
        """execute_model entry, before any launch of the step."""
        self._step_no += 1
        if self._round is not None:
            self._round.begin(self._step_no, scheduler_output)
        self._pump_glog()
        if self._tracer is not None:
            self._tracer.housekeep()
        if self.out_dir is None:
            return
        level, traced = 0, []
        new_ids = set()
        for new in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            new_ids.add(new.req_id)
            f = self.flags.get(new.req_id, new.sampling_params)
            if f.trace:
                level = max(level, f.trace)
                traced.append(new.req_id)
        for req_id in getattr(scheduler_output, "num_scheduled_tokens", {}) or {}:
            if req_id in new_ids:
                continue
            f = self._flags_for(req_id)
            if f.trace:
                level = max(level, f.trace)
                traced.append(req_id)
        if not level:
            budget = getattr(sys.modules.get("hybrid.r2_bank"), "TRACE", None)
            nst = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
            if (budget is not None and nst and not new_ids and all(n == 6 for n in nst.values())
                    and budget.take(len(nst))):
                level, traced = 2, list(nst)
        if level:
            tracer = self._get_tracer()
            if tracer is not None:
                self._trace_on = tracer.step_begin(
                    level=level,
                    step=self._step_no,
                    req_ids=traced,
                    num_reqs=len(scheduler_output.num_scheduled_tokens),
                    num_tokens=int(getattr(scheduler_output, "total_num_scheduled_tokens", 0)),
                )

    def mark(self, phase: str) -> None:
        if self._trace_on:
            self._tracer.mark(phase)

    def step_end(self) -> None:
        if self._trace_on:
            self._trace_on = False
            self._tracer.step_end()
        if self._round is not None:
            self._round.end()

    # ---- target forward: graph decision (round log) and request-to-verifier records
    def before_target_forward(self, cudagraph_mode, batch_desc) -> None:
        """Directly before _model_forward, with the runner's final dispatch decision."""
        self.mark("target_forward")
        if self._round is not None:
            self._round.before_forward(cudagraph_mode, batch_desc)
        if self.out_dir is not None and self.rank == 0:
            self._request_records()

    def after_target_forward(self) -> None:
        if self._round is not None:
            self._round.after_forward()

    def after_corrections(self, had_deferred: bool) -> None:
        """execute_model end, after the deferred async-scheduling corrections ran."""
        if self._round is not None:
            self._round.after_corrections(had_deferred)

    def _request_records(self) -> None:
        runner = self.runner
        ib = runner.input_batch
        done = self._req_done
        for row, req_id in enumerate(ib.req_ids[: ib.num_reqs]):
            if req_id in done:
                continue
            f = self._flags_for(req_id)
            if not (f.trace or f.glog):
                continue
            done.add(req_id)
            from hybrid.dsv41_r1_round import request_record

            st = runner.requests.get(req_id)
            kind = row_kind(st.sampling_params if st is not None else None, row in ib.generators)
            if self._req_sink is None:
                from hybrid.dsv41_r1_trace import JsonlSink

                self._req_sink = JsonlSink(
                    os.path.join(self.out_dir, f"requests-rank0-pid{os.getpid()}.jsonl"))
            rec = request_record(runner, row, req_id, self._step_no, kind)
            rec["flags"] = {"tq": f.tq, "glog": f.glog, "trace": f.trace}
            self._req_sink.put(rec)
        if len(done) > 8192:
            live = set(runner.requests)
            self._req_done = {r for r in done if r in live}

    # ---- sampled drafting (part A)
    def note_logits(self, logits) -> None:
        if logits is not None:
            self.logits_width = int(logits.shape[-1])

    def check_draft_probs(self, draft_probs, logits, spec_decode_metadata):
        """Drop (never fix up) a q that does not match the target logits: no-q stays lossless."""
        if draft_probs is None:
            if self._round is not None:
                self._round.rejection_input(None, None)
            return None
        num_tokens = spec_decode_metadata.draft_token_ids.shape[0]
        ok = (
            logits is not None
            and draft_probs.dim() == 2
            and draft_probs.dtype == torch.float32
            and draft_probs.shape[-1] == logits.shape[-1]
            and draft_probs.shape[0] == num_tokens
            and draft_probs.is_contiguous()
        )
        if not ok:
            warn_once(
                "qshape",
                "R1: draft_probs %s/%s do not match target logits %s and %d draft tokens; "
                "verifying without q",
                tuple(draft_probs.shape), draft_probs.dtype,
                None if logits is None else tuple(logits.shape), num_tokens,
            )
            passed = None
        else:
            passed = draft_probs
        if self._round is not None:
            self._round.rejection_input(draft_probs, passed)
        return passed

    def before_propose(self, drafter) -> None:
        runner = self.runner
        ib = runner.input_batch
        n = ib.num_reqs
        req_ids = list(ib.req_ids[:n])
        sps = []
        for r in req_ids:
            st = runner.requests.get(r)
            sps.append(st.sampling_params if st is not None else None)
        drafter._dsv41_r1_plan = build_draft_plan(
            req_ids,
            sps,
            ib.generators.keys(),
            self.flags,
            glog_enabled=self.glog_enabled,
            mark=self.mark if self._trace_on else None,
            vocab_expected=self.logits_width,
        )
        if self._round is not None:
            self._round.before_propose(drafter, drafter._dsv41_r1_plan)

    def after_propose(self, drafter) -> None:
        if self._round is not None:
            self._round.after_propose(drafter)
        drafter._dsv41_r1_plan = None
        stash = getattr(drafter, "_dsv41_r1_glog", None)
        drafter._dsv41_r1_glog = None
        self._glog_pending = stash

    # ---- g logger (part B)
    def drop_glog_pending(self) -> None:
        self._glog_pending = None

    def _sink(self):
        if self._glog_sink is None:
            from hybrid.dsv41_r1_trace import JsonlSink

            path = os.path.join(self.out_dir, f"glog-rank{self.rank}-pid{os.getpid()}.jsonl")
            self._glog_sink = JsonlSink(path)
            self._glog_sink.put({"type": "header", "version": VERSION, "rank": self.rank,
                                 "columns": self._glog_cols, "grid": list(GLOG_GRID)})
        return self._glog_sink

    def _pump_glog(self) -> None:
        q = self._glog_inflight
        while q and (q[0][0] is None or q[0][0].query()):
            _, host, meta = q.popleft()
            cols = self._glog_cols
            self._sink().put(lambda host=host, meta=meta: glog_records(host.tolist(), meta, cols))

    def after_rejection(self, spec_decode_metadata, logits, sampling_metadata) -> None:
        pending, self._glog_pending = self._glog_pending, None
        if pending is None or logits is None or not self.glog_enabled or self.rank != 0:
            return
        stash, stash_ids, stash_tq, stash_sampled = pending
        row_of = {rid: i for i, rid in enumerate(stash_ids)}
        runner = self.runner
        ib = runner.input_batch
        gen_rows = ib.generators
        sel, meta = [], []
        cu = 0
        for row, (req_id, nd) in enumerate(
            zip(ib.req_ids, spec_decode_metadata.num_draft_tokens)
        ):
            if nd > 0:
                st = runner.requests.get(req_id)
                sp = st.sampling_params if st is not None else None
                f = self.flags.get(req_id, sp)
                srow = row_of.get(req_id)
                kind = row_kind(sp, row in gen_rows)
                if f.glog and srow is not None and kind != "greedy":
                    count = self._glog_steps.get(req_id, 0)
                    self._glog_steps[req_id] = count + 1
                    ks = range(nd) if count % f.glog_every == 0 else (0,)
                    tq = float(stash_tq[srow]) if stash_sampled else 0.0
                    for k in ks:
                        sel.append((cu + k, srow, k, float(sp.temperature), tq))
                        meta.append({
                            "type": "glog", "req": req_id, "vstep": count, "k": k,
                            "T": float(sp.temperature), "tq": tq or None,
                            "mode": "sampled" if tq > 0 else "greedy", "kind": kind,
                            "approx": p_is_approx(sp), "nd": nd, "step": self._step_no,
                        })
            cu += nd
        if len(self._glog_steps) > 8192:
            live = set(runner.requests)
            self._glog_steps = {k: v for k, v in self._glog_steps.items() if k in live}
        if not sel:
            return
        device = logits.device
        cuda = device.type == "cuda"

        def col(i, dtype):
            t = torch.tensor([s[i] for s in sel], dtype=dtype)
            return t.pin_memory().to(device, non_blocking=True) if cuda else t.to(device)

        flat, srow_t, k_t = col(0, torch.int64), col(1, torch.int64), col(2, torch.int64)
        temp_t, tq_t = col(3, torch.float32), col(4, torch.float32)
        rows = spec_decode_metadata.target_logits_indices.index_select(0, flat)
        p_logits = logits.index_select(0, rows).float()
        q_logits = stash[srow_t, k_t].float()
        d = spec_decode_metadata.draft_token_ids.index_select(0, flat).to(torch.int64)
        stats = glog_stats(p_logits, temp_t, q_logits, d, tq_t)
        stats = torch.cat([stats, d.float().unsqueeze(1)], dim=1)
        if cuda:
            host = torch.empty(stats.shape, dtype=stats.dtype, pin_memory=True)
            host.copy_(stats, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
        else:
            host, ev = stats.cpu(), None
        self._glog_inflight.append((ev, host, meta))
        if len(self._glog_inflight) > 256:
            self._glog_inflight.popleft()
        if not cuda:
            self._pump_glog()
