"""Reload R1 round logger and q-path attestation. Default OFF; an observer only.

Enabled by boot env ``DSV41_R1_ROUNDLOG=1`` together with ``DSV41_R1_OUT=<dir>`` on a runner
armed with ``DSV41_R1_FEATURES=1`` (the hooks object only exists when armed). It is not a
request flag: every iteration of all traffic is logged. Nothing here changes a plan, a tensor,
a graph decision or a sampler path. It never synchronizes the device: every value is a host
value the runner already holds, and the one device-produced value (the committed counts of the
previous verify) is read only after ``Event.query()`` reports its copy complete.

Files (one JSONL per process, written by the background sink of dsv41_r1_trace):
  round-rank0-pid<P>.jsonl          rank 0: header, then one full record per iteration
  round-attest-rank<R>-pid<P>.jsonl other ranks: header, then the compact attestation fields
  requests-rank0-pid<P>.jsonl       rank 0: one record per request flagged dsv41_trace or
                                    dsv41_glog (written whenever DSV41_R1_OUT is set; see
                                    request_record)

Iteration ``it`` counts execute_model calls; it is identical on all ranks. One iteration holds
the verify of the drafts proposed in iteration it-1 (``rs``) and the propose for it+1
(``propose``/``q``).
"""
from __future__ import annotations

import os
import sys
import time

VERSION = "dsv41-r1-round-v1"
ROUNDLOG = os.environ.get("DSV41_R1_ROUNDLOG") == "1"
SAMPLING_EPS = 1e-5
LIST_MAX_B = 8  # per-request maps (temperatures, corrections) only for small batches


def _shape(t):
    return None if t is None else [int(x) for x in t.shape]


def _num(v):
    """JSON-safe scalar: Python scalars as they are, numpy/torch scalars via item()."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    item = getattr(v, "item", None)
    if item is not None:
        try:
            return item()
        except (TypeError, ValueError, RuntimeError):
            pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def _prompt_len(new_req):
    ids = getattr(new_req, "prompt_token_ids", None)
    if ids is not None:
        return len(ids)
    emb = getattr(new_req, "prompt_embeds", None)
    return None if emb is None else int(emb.shape[0])


# ---------------------------------------------------------------- per-iteration fields
def batch_shape(scheduler_output, requests) -> dict:
    """B, scheduled tokens, prefill (chunked included) vs decode, per-request draft counts.

    A request is in prefill while its computed tokens are below its prompt length;
    ``prefill_partial`` counts the prefills whose chunk still ends inside the prompt.
    """
    nst = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
    spec = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
    computed, prompt = {}, {}
    for new in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
        computed[new.req_id] = new.num_computed_tokens
        prompt[new.req_id] = _prompt_len(new)
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    for i, rid in enumerate(getattr(cached, "req_ids", ()) or ()):
        computed[rid] = cached.num_computed_tokens[i]
        st = requests.get(rid)
        prompt[rid] = getattr(st, "num_prompt_tokens", None) if st is not None else None
    prefill = partial = decode = unknown = 0
    for rid, n in nst.items():
        c, pl = computed.get(rid), prompt.get(rid)
        if c is None or pl is None:
            unknown += 1
        elif c < pl:
            prefill += 1
            partial += int(c + n < pl)
        else:
            decode += 1
    return {
        "B": len(nst),
        "tokens": int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0),
        "prefill": prefill,
        "prefill_partial": partial,
        "decode": decode,
        "unknown": unknown,
        "nd": {rid: len(spec.get(rid, ())) for rid in nst},
    }


def scheduled_params(scheduler_output, requests) -> dict:
    """req_id -> SamplingParams for the scheduled requests (new ones from the scheduler output)."""
    out = {}
    new_sp = {n.req_id: n.sampling_params
              for n in getattr(scheduler_output, "scheduled_new_reqs", ()) or ()}
    for rid in getattr(scheduler_output, "num_scheduled_tokens", None) or {}:
        if rid in new_sp:
            out[rid] = new_sp[rid]
        else:
            st = requests.get(rid)
            out[rid] = st.sampling_params if st is not None else None
    return out


def temperature_descriptor(params: dict) -> dict:
    """greedy / random / T == 1 exactly counts; the per-request list only when B <= 8."""
    greedy = random = t1 = seeded = structured = 0
    temps = {}
    for rid, sp in params.items():
        t = 0.0 if sp is None else float(getattr(sp, "temperature", 0.0) or 0.0)
        temps[rid] = t
        if t < SAMPLING_EPS:
            greedy += 1
        else:
            random += 1
            seeded += int(getattr(sp, "seed", None) is not None)
            structured += int(getattr(sp, "structured_outputs", None) is not None)
        t1 += int(t == 1.0)
    out = {"greedy": greedy, "random": random, "t_eq_1": t1, "seeded": seeded,
           "structured": structured}
    if len(params) <= LIST_MAX_B:
        out["T"] = temps
    return out


def membership(scheduler_output) -> dict:
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    out = {
        "joined": [n.req_id for n in getattr(scheduler_output, "scheduled_new_reqs", ()) or ()],
        "resumed": sorted(getattr(cached, "resumed_req_ids", ()) or ()),
        "finished": sorted(getattr(scheduler_output, "finished_req_ids", ()) or ()),
    }
    pre = getattr(scheduler_output, "preempted_req_ids", None)
    if pre:
        out["preempted"] = sorted(pre)
    return out


def optimistic_drafts(scheduler_output, runner) -> dict:
    """req_id -> placeholders _update_states is about to append (its prev_num_draft_len read)."""
    if not getattr(runner, "use_async_scheduling", False):
        return {}
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    in_batch = runner.input_batch.req_id_to_index
    out = {}
    for rid in getattr(cached, "req_ids", ()) or ():
        st = runner.requests.get(rid)
        n = getattr(st, "prev_num_draft_len", 0) if st is not None else 0
        if n and rid in in_batch:
            out[rid] = int(n)
    return out


# ---------------------------------------------------------------- bank / graph probes
def bank_selector(drafter):
    """(selector, controller) published by the cumulative bank, or (None, None)."""
    sel = getattr(drafter, "_w26_sampler_selector", None)
    return sel, getattr(sel, "controller", None)


def r2_ledger():
    """The R2 hit ledger when R2 is published on this worker, else None."""
    bank = getattr(sys.modules.get("hybrid.r2_bank"), "BANK", None)
    return None if bank is None else bank.ledger


def coverage_snapshot(controller):
    """(attention_hits, sampler_hits, sampler_fallbacks copy) from the bank's host counters."""
    cov = getattr(controller, "coverage", None)
    if not isinstance(cov, dict):
        return None
    return (int(cov.get("attention_hits", 0)), int(cov.get("sampler_hits", 0)),
            dict(cov.get("sampler_fallbacks") or {}))


def desc_list(desc):
    if desc is None:
        return None
    return [_num(getattr(desc, k, None))
            for k in ("num_tokens", "num_reqs", "uniform", "has_lora", "num_active_loras")]


def dual_graph_runtime(wrapper):
    """State of a hybrid.w26_dual_graph runtime registered on the target wrapper (no import)."""
    mod = sys.modules.get("hybrid.w26_dual_graph")
    rt = getattr(mod, "RUNTIMES", {}).get(id(wrapper)) if mod is not None else None
    if rt is None:
        return None
    return {"mode": getattr(rt, "mode", None),
            "attention_adapter": getattr(rt, "attention_adapter", None) is not None,
            "kernels": getattr(rt, "kernels", None) is not None}


def graph_path(mode_name, wrapper_type, attention_hits, entries_before, entries_after) -> str:
    """The path BreakableCUDAGraphWrapper.__call__ took for the target forward.

    NONE runs the model eagerly; otherwise w26_dual_graph.intercept replays a private bank
    entry when the bank's lookup counted an attention hit; otherwise the wrapper replays its
    own breakable graph for the descriptor (one artifact for FULL and PIECEWISE descriptors),
    capturing it first when no entry existed.
    """
    if mode_name == "NONE":
        return "eager"
    if attention_hits:
        return "private_bank"
    if wrapper_type != "BreakableCUDAGraphWrapper":
        return "wrapper:" + str(wrapper_type)
    if entries_before is None or entries_after is None:
        return "breakable_graph"
    return "breakable_graph_capture" if entries_after > entries_before else "breakable_graph_replay"


def proposal_path(before, after, planned_path, arm, has_selector) -> dict:
    """Which sampler produced the drafts: the bank (with its arm) or a fallback and its reason."""
    hits, fallbacks = 0, {}
    if before is not None and after is not None:
        hits = after[1] - before[1]
        fallbacks = {k: v - before[2].get(k, 0) for k, v in after[2].items()
                     if v != before[2].get(k, 0)}
    if hits > 0:
        return {"path": "bank", "arm": arm}
    if fallbacks:
        reason = ",".join(sorted(fallbacks))
    elif not has_selector:
        reason = "no_selector"
    else:
        reason = "arm_without_sampler"
    return {"path": planned_path or "incumbent", "fallback": reason, "arm": arm}


def committed_counts(runner, source):
    """{req_id: accepted + bonus} of the previous verify, from the runner's pinned host copy.

    ``source`` is the prev_sampled_token_ids tensor seen at the previous iteration's propose
    (DSpark copies the counts just before proposing). Returns None when that copy is not the
    current one, "pending" while its copy event is still running (query only, never a wait).
    """
    ib = runner.input_batch
    ev = getattr(runner, "valid_sampled_token_count_event", None)
    cpu = getattr(runner, "valid_sampled_token_count_cpu", None)
    pst = getattr(ib, "prev_sampled_token_ids", None)
    prev = getattr(ib, "prev_req_id_to_index", None)
    if ev is None or cpu is None or pst is None or source is None or pst is not source or not prev:
        return None
    if not ev.query():
        return "pending"
    n = int(pst.shape[0])
    vals = cpu[:n].tolist()
    return {rid: int(vals[i]) for rid, i in prev.items() if 0 <= i < n}


def placeholder_report(optimistic: dict, committed, had_deferred: bool, batch: int) -> dict:
    out = {"deferred": bool(had_deferred), "optimistic": len(optimistic)}
    if isinstance(committed, dict) and optimistic:
        corr = {rid: n - (committed[rid] - 1) for rid, n in optimistic.items() if rid in committed}
        out["applied"] = bool(had_deferred)
        out["corrected"] = sum(1 for c in corr.values() if c)
        if batch <= LIST_MAX_B:
            out["c"] = corr
    else:
        out["applied"] = False
    return out


# ---------------------------------------------------------------- request-to-verifier record
SP_FIELDS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
             "repetition_penalty", "seed", "max_tokens", "min_tokens", "ignore_eos",
             "thinking_token_budget")
BATCH_ARRAYS = {
    "temperature": "temperature_cpu",
    "top_p": "top_p_cpu",
    "top_k": "top_k_cpu",
    "presence_penalty": "presence_penalties_cpu",
    "frequency_penalty": "frequency_penalties_cpu",
    "repetition_penalty": "repetition_penalties_cpu",
}


def request_record(runner, row: int, req_id: str, it: int, kind: str) -> dict:
    """Effective sampling parameters as the runner holds them: the request's cached
    SamplingParams and the input-batch row the sampler and the rejection sampler read."""
    st = runner.requests.get(req_id)
    sp = st.sampling_params if st is not None else None
    ib = runner.input_batch
    sp_rec = {k: _num(getattr(sp, k, None)) for k in SP_FIELDS}
    bias = getattr(sp, "logit_bias", None)
    sp_rec["logit_bias_size"] = len(bias) if bias else 0
    sp_rec["structured_outputs"] = getattr(sp, "structured_outputs", None) is not None
    sp_rec["extra_args"] = sorted((getattr(sp, "extra_args", None) or {}).keys())
    batch = {}
    for name, attr in BATCH_ARRAYS.items():
        arr = getattr(ib, attr, None)
        batch[name] = None if arr is None else _num(arr[row])
    gens = getattr(ib, "generators", {}) or {}
    batch["generator"] = row in gens
    for name in ("greedy_reqs", "random_reqs", "top_p_reqs", "top_k_reqs"):
        s = getattr(ib, name, None)
        if s is not None:
            batch[name[:-5]] = req_id in s
    bad = getattr(ib, "bad_words_token_ids", None)
    batch["bad_words"] = bool(bad and row in bad)
    return {"type": "request", "req": req_id, "it": it, "row": row, "kind": kind,
            "num_prompt_tokens": _num(getattr(st, "num_prompt_tokens", None)),
            "sp": sp_rec, "batch": batch}


# ---------------------------------------------------------------- the logger
def boot_attestation(runner) -> dict:
    """Static facts the q path depends on (identity vocab map, no draft top-k)."""
    out = {}
    drafter = getattr(runner, "drafter", None)
    model = getattr(drafter, "model", None)
    try:
        out["drafter"] = type(drafter).__name__
        out["draft_id_to_target_id_is_none"] = getattr(model, "draft_id_to_target_id", None) is None
        out["draft_topk"] = _num(getattr(drafter, "_draft_topk", None))
        out["num_speculative_tokens"] = _num(getattr(drafter, "num_speculative_tokens", None))
        out["target_wrapper"] = type(getattr(runner, "model", None)).__name__
        out["async_scheduling"] = bool(getattr(runner, "use_async_scheduling", False))
        cc = getattr(runner, "compilation_config", None)
        out["cudagraph_mode"] = str(getattr(cc, "cudagraph_mode", None))
        mc = getattr(runner, "model_config", None)
        out["target_vocab"] = _num(mc.get_vocab_size()) if mc is not None else None
        sel, ctl = bank_selector(drafter)
        out["bank_selector"] = sel is not None
        out["bank_arm"] = getattr(ctl, "selected", None)
    except Exception as exc:  # partial header rather than no log
        out["error"] = repr(exc)
    return out


class RoundLog:
    def __init__(self, runner, out_dir: str, rank_fn):
        self.runner = runner
        self.out_dir = out_dir
        self._rank_fn = rank_fn
        self._sink = None
        self._rec = None
        self._opt = {}
        self._cov_fwd = None
        self._entries = None
        self._cov_prop = None
        self._plan = None
        self._src = None  # (iteration, prev_sampled_token_ids) captured at propose
        self._r2 = None
        self.written = 0

    @property
    def rank(self) -> int:
        return self._rank_fn()

    def sink(self):
        if self._sink is None:
            from hybrid.dsv41_r1_trace import JsonlSink

            r = self.rank
            name = (f"round-rank0-pid{os.getpid()}.jsonl" if r == 0
                    else f"round-attest-rank{r}-pid{os.getpid()}.jsonl")
            self._sink = JsonlSink(os.path.join(self.out_dir, name))
            self._sink.put({"type": "header", "version": VERSION, "rank": r, "pid": os.getpid(),
                            "boot": boot_attestation(self.runner)})
        return self._sink

    def begin(self, it: int, scheduler_output) -> None:
        if self._rec is not None:
            self._emit(complete=False)
        runner = self.runner
        rec = {"type": "round", "it": it, "t_mono_ns": time.monotonic_ns(),
               "t_wall_ns": time.time_ns(), "graph": None, "rs": None, "propose": None,
               "q": None}
        if self.rank == 0:
            rec.update(batch_shape(scheduler_output, runner.requests))
            rec["T"] = temperature_descriptor(scheduled_params(scheduler_output, runner.requests))
            rec.update(membership(scheduler_output))
        self._opt = optimistic_drafts(scheduler_output, runner)
        ledger = r2_ledger()
        self._r2 = None if ledger is None else ledger.snapshot()
        if ledger is not None and self.rank == 0:
            rec["uniform6"] = rec["decode"] == rec["B"] and rec["tokens"] == 6 * rec["B"]
        from .abc_runtime import BANK as abc_bank  # STAR_ABC integration
        self._abc_before=None if abc_bank is None else dict(abc_bank.hits)
        if abc_bank is not None and abc_bank.selection is not None:
            rec["abc"] = dict(profile=abc_bank.selection.selected,identity=abc_bank.selection.identity,epoch=abc_bank.selection.epoch,session=abc_bank.selection.session)
        self._rec = rec

    def before_forward(self, mode, desc) -> None:
        rec = self._rec
        if rec is None:
            return
        runner = self.runner
        _, ctl = bank_selector(getattr(runner, "drafter", None))
        self._cov_fwd = coverage_snapshot(ctl)
        entries = getattr(getattr(runner, "model", None), "entries", None)
        self._entries = len(entries) if isinstance(entries, dict) else None
        rec["graph"] = {"dispatch": getattr(mode, "name", str(mode)), "desc": desc_list(desc)}

    def after_forward(self) -> None:
        rec = self._rec
        if rec is None or rec["graph"] is None:
            return
        runner = self.runner
        _, ctl = bank_selector(getattr(runner, "drafter", None))
        cov = coverage_snapshot(ctl)
        hits = cov[0] - self._cov_fwd[0] if cov is not None and self._cov_fwd is not None else 0
        model = getattr(runner, "model", None)
        entries = getattr(model, "entries", None)
        after = len(entries) if isinstance(entries, dict) else None
        g = rec["graph"]
        g["path"] = graph_path(g["dispatch"], type(model).__name__, hits, self._entries, after)
        abc_bank=getattr(sys.modules.get('hybrid.abc_runtime'),'BANK',None)
        if abc_bank is not None and self._abc_before is not None:
            abc_hits={k:v-self._abc_before.get(k,0) for k,v in abc_bank.hits.items() if k.startswith('target:') and v>self._abc_before.get(k,0)}
            if abc_hits:g['path']='abc_private_bank';g['abc_target_hits']=abc_hits
        if hits:
            g["arm"] = getattr(ctl, "selected", None)
        rt = dual_graph_runtime(model)
        if rt is not None:
            g["runtime"] = rt

    def after_corrections(self, had_deferred: bool) -> None:
        rec = self._rec
        if rec is None:
            return
        src, self._src = self._src, None
        if self.rank == 0:
            fresh = src is not None and src[0] == rec["it"] - 1
            committed = committed_counts(self.runner, src[1] if fresh else None)
            rec["committed_prev"] = committed
            rec["placeholders"] = placeholder_report(self._opt, committed, had_deferred,
                                                     rec.get("B", 0))

    def rejection_input(self, mapped, passed) -> None:
        if self._rec is not None:
            self._rec["rs"] = {"mapped": _shape(mapped), "passed": _shape(passed)}

    def before_propose(self, drafter, plan) -> None:
        if self._rec is None:
            return
        drafter._dsv41_r1_path = None
        self._plan = plan
        _, ctl = bank_selector(drafter)
        self._cov_prop = coverage_snapshot(ctl)
        pst = getattr(self.runner.input_batch, "prev_sampled_token_ids", None)
        self._src = (self._rec["it"], pst) if pst is not None else None

    def after_propose(self, drafter) -> None:
        rec = self._rec
        if rec is None:
            return
        sel, ctl = bank_selector(drafter)
        prop = proposal_path(self._cov_prop, coverage_snapshot(ctl),
                             getattr(drafter, "_dsv41_r1_path", None),
                             getattr(ctl, "selected", None), sel is not None)
        plan = self._plan
        prop["plan"] = plan is not None
        if plan is not None and self.rank == 0:
            prop["tq"] = list(plan.tq)
        rec["propose"] = prop
        runner = self.runner
        rec["q"] = _shape(getattr(runner, "_draft_probs", None))
        ids = getattr(runner, "_draft_prob_req_ids", None)
        if rec["q"] is not None and ids is not None:
            rec["q_rows"] = len(ids)
        self._plan = None
        self._cov_prop = None

    def end(self) -> None:
        if self._rec is not None:
            self._emit(complete=True)

    def _emit(self, complete: bool) -> None:
        rec, self._rec = self._rec, None
        rec["complete"] = complete
        abc_bank=getattr(sys.modules.get('hybrid.abc_runtime'),'BANK',None)
        if abc_bank is not None and self._abc_before is not None and 'abc' in rec:
            rec['abc']['hits']={k:v-self._abc_before.get(k,0) for k,v in abc_bank.hits.items() if v!=self._abc_before.get(k,0)}
            captured=[k for k in rec['abc']['hits'] if k.startswith(('sampler:','sampled:'))]
            if captured and rec.get('propose') is not None:
                rec['propose']['path']='abc_captured_sampler';rec['propose']['abc_paths']=captured
        ledger = r2_ledger()
        if ledger is not None and self._r2 is not None:
            rec["r2"] = ledger.delta(self._r2, ledger.snapshot())
        self._r2 = None
        if self.rank != 0:
            prop = rec["propose"] or {}
            rec = {"type": "attest", "it": rec["it"], "complete": complete,
                   "prop": prop.get("path"), "fb": prop.get("fallback"), "arm": prop.get("arm"),
                   "plan": prop.get("plan"), "q": rec["q"],
                   "rs": rec["rs"], "g": (rec["graph"] or {}).get("path"),
                   **({"r2": rec["r2"]} if "r2" in rec else {}),**({'abc':rec['abc']} if 'abc' in rec else {})}
        self.sink().put(rec)
        self.written += 1
