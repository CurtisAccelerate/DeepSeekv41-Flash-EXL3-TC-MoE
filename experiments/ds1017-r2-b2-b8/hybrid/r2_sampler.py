"""R2 per-B draft samplers for B2..B8 (default OFF; built only by r2_startup).

GreedyRows: the B1 captured local-shard Markov chain generalised to B rows. Each
step takes the local (value, index) max per row and all-gathers 4 pairs per row;
the winner is the first maximal rank, i.e. the lowest global index, so the
result is bitwise the incumbent full-vocab argmax (qualified at boot per B).

SampledRows: the R1 draft_chain (mixed-mask form) captured per B with the
proposer's own R1 generator registered on the graph. The mixed-mask form equals
the all_sample form bitwise when every row samples, so one graph per B serves
every T_q pattern. Qualified per B against eager draft_chain at the same
generator state; the generator state is restored after qualification.

Unsupported rows (glog, top-k, confidence, margin observer) fall back losslessly
to the R1/incumbent path with a counted reason.
"""
import os
import threading

import torch
import torch.nn.functional as F

N = 5
SHARD = 32320
WORLD = 4


def gather_pairs(nccl, x):
    """PyNccl all_gather with the B1 layout: (B, 2) per rank -> (B, 4, 2), rank-major in dim 1."""
    y = torch.empty((x.shape[0] * WORLD,) + x.shape[1:], device=x.device, dtype=x.dtype)
    nccl.all_gather(y, x.contiguous())
    return y.reshape((WORLD,) + x.shape).movedim(0, 1)


def pair_select(pairs):
    """(B, 4, 2) [value, global index] -> int64 (B,): first maximal rank's index."""
    winner = pairs[:, :, 0].argmax(-1, keepdim=True)
    return pairs[:, :, 1].to(torch.int64).gather(-1, winner).squeeze(-1)


def local_pair(logits, start):
    value, index = logits.max(-1)
    return torch.stack((value.float(), (index + start).float()), -1)


def mixed_chain(model, base_logits, prev, tq, generator):
    """dsv41_r1.draft_chain(all_sample=False, keep_logits=False, keep_embeds=False), graph-safe."""
    num_reqs, n, vocab = base_logits.shape
    drafts = torch.empty(num_reqs, n, dtype=torch.int64, device=base_logits.device)
    mask = tq > 0
    denom = torch.where(mask, tq, torch.ones_like(tq)).unsqueeze(1)
    mask_col = mask.unsqueeze(1)
    q_out = torch.empty(num_reqs, n, vocab, dtype=torch.float32, device=base_logits.device)
    for i in range(n):
        logits_i = base_logits[:, i] + model.markov_bias(model.markov_embed(prev))
        tok = model.map_draft_to_target(logits_i.argmax(dim=-1))
        q = torch.softmax(logits_i.float() / denom, dim=-1)
        noise = torch.empty_like(q).exponential_(generator=generator)
        sampled = (q / noise).argmax(dim=-1)
        tok = torch.where(mask, sampled, tok)
        one_hot = torch.zeros_like(q).scatter_(1, tok.unsqueeze(1), 1.0)
        q_out[:, i] = torch.where(mask_col, q, one_hot)
        drafts[:, i] = tok
        prev = tok
    return drafts, q_out


class _Captured:
    def __init__(self, p, b):
        self.p = p
        self.b = b
        self.graph = None
        self.ready = False
        self.lock = threading.Lock()
        self.serving_stream = torch.cuda.current_stream().cuda_stream
        self.previous = torch.cuda.Event()
        self.previous.record(torch.cuda.current_stream())
        m = p.model
        self.weights = tuple((v, v.data_ptr()) for v in (
            m.model.markov_head.markov_w2.weight, m.model.markov_head.markov_w1.weight,
            m.lm_head.weight, m.model.norm.weight))

    def assert_intact(self):
        m = self.p.model
        current = (m.model.markov_head.markov_w2.weight, m.model.markov_head.markov_w1.weight,
                   m.lm_head.weight, m.model.norm.weight)
        if not self.ready or any(now is not old or now.data_ptr() != ptr
                                 for now, (old, ptr) in zip(current, self.weights)):
            raise RuntimeError(f"R2 sampler B{self.b}: readiness or weight binding changed")

    def _run(self, fill, take):
        self.assert_intact()
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Concurrent host sampler invocation unsupported")
        try:
            current = torch.cuda.current_stream()
            if current.cuda_stream != self.serving_stream:
                raise RuntimeError("Unqualified cross-stream R2 sampling")
            current.wait_event(self.previous)
            fill()
            self.graph.replay()
            owned = take()
            self.previous.record(current)
            return owned
        finally:
            self.lock.release()


class GreedyRows(_Captured):
    def __init__(self, p, nccl, start, end, b):
        super().__init__(p, b)
        self.nccl = nccl
        self.start, self.end = start, end
        self.weight = p.model.model.markov_head.markov_w2.weight
        self.local_weight = self.weight[start:end]
        dev = self.weight.device
        self.base = torch.zeros((b, N, SHARD), dtype=torch.bfloat16, device=dev)
        self.anchor = torch.zeros(b, dtype=torch.int64, device=dev)
        self.output = None

    def chain(self):
        prev = self.anchor
        result = []
        for i in range(N):
            embed = self.p.model.markov_embed(prev)
            logits = self.base[:, i] + F.linear(embed, self.local_weight)
            prev = pair_select(gather_pairs(self.nccl, local_pair(logits, self.start)))
            result.append(prev)
        return torch.stack(result, -1)

    def sample(self, hidden):
        p = self.p
        base = p.model.logits_processor(p.model.lm_head, p.model.model.norm(hidden), skip_gather=True)
        if base.dtype != torch.bfloat16 or tuple(base.shape) != (N * self.b, SHARD):
            raise RuntimeError("Unqualified head shape/dtype")

        def fill():
            self.base.copy_(base.view(self.b, N, SHARD))
            self.anchor.copy_(p.input_ids[p._anchor_idx[:self.b]])
        return self._run(fill, lambda: self.output.clone())


class SampledRows(_Captured):
    def __init__(self, p, b, vocab, generator):
        super().__init__(p, b)
        dev = p.model.lm_head.weight.device
        self.vocab = vocab
        self.generator = generator
        self.base = torch.zeros((b, N, vocab), dtype=torch.bfloat16, device=dev)
        self.anchor = torch.zeros(b, dtype=torch.int64, device=dev)
        self.tq = torch.zeros(b, dtype=torch.float32, device=dev)
        self.drafts = None
        self.q = None

    def chain(self):
        return mixed_chain(self.p.model, self.base, self.anchor, self.tq, self.generator)

    def sample(self, hidden, tq):
        p = self.p
        base = p.model.compute_draft_logits(hidden)
        if tuple(base.shape) != (N * self.b, self.vocab):
            raise RuntimeError("Unqualified draft logits shape")

        def fill():
            self.base.copy_(base.view(self.b, N, self.vocab))
            self.anchor.copy_(p.input_ids[p._anchor_idx[:self.b]])
            self.tq.copy_(tq, non_blocking=True)
        return self._run(fill, lambda: (self.drafts.clone(), self.q.clone()))


# ---------------------------------------------------------------- serving entry
def fallback_reason(p, num_reqs, hidden, plan, bank):
    """None when an R2 sampler may run this call; else the lossless fallback reason."""
    if bank is None or not bank.sampler_ready(num_reqs, plan is not None and plan.any_sample):
        return "not_prepared"
    if plan is not None and plan.any_glog:
        return "glog_rows"
    if p.num_speculative_tokens != N:
        return "depth"
    if p._draft_topk is not None or p._conf_threshold() != 0:
        return "topk_or_confidence"
    if os.environ.get("W26_OBSERVE_MARGINS", "0") == "1":
        return "margin_observer"
    if tuple(hidden.shape) != (N * num_reqs, p.model.config.hidden_size) or hidden.dtype != torch.bfloat16:
        return "hidden_shape_dtype"
    if torch.cuda.is_current_stream_capturing():
        return "outer_capture"
    return None


def try_rows(p, num_reqs, hidden, plan):
    """R2 drafts for B2..B8 or None (caller keeps its existing path). Counts every outcome."""
    selector = getattr(p, "_w26_sampler_selector", None)
    bank = getattr(selector, "r2", None)
    if bank is None or bank.contract.plan(num_reqs) is None:
        return None
    sampled = plan is not None and plan.any_sample
    comp = "sampled" if sampled else "sampler"
    if comp not in bank.contract.plan(num_reqs).components:
        bank.ledger.fallback(num_reqs, comp, "component_disabled")
        return None
    # Arm/session/production authorisation is the existing selector's; no sampler arm, no R2.
    if selector.sampler() is None:
        bank.ledger.fallback(num_reqs, comp, "arm_without_sampler")
        return None
    reason = fallback_reason(p, num_reqs, hidden, plan, bank)
    if reason is None and sampled:
        from hybrid import dsv41_r1 as r1
        if not r1.sampling_supported(p, bank.vocab, plan):
            reason = "sampling_unsupported"
    if reason is not None:
        bank.ledger.fallback(num_reqs, comp, reason)
        return None
    if sampled:
        from hybrid import dsv41_r1 as r1
        drafts, q = bank.sampled[num_reqs].sample(hidden, r1.tq_tensor(plan, p.device))
        p._last_draft_probs = q
        p._dsv41_r1_path = "r2_sampled"
    else:
        drafts = bank.greedy[num_reqs].sample(hidden)
        if plan is not None:
            p._dsv41_r1_path = "r2_greedy"
    bank.ledger.hit(num_reqs, comp)
    return drafts


# ---------------------------------------------------------------- boot preparation
@torch.inference_mode()
def prepare_rows(p, nccl, contract, consensus, agree, local, collective, start, end, vocab):
    """Build and qualify per-B greedy and sampled samplers. Caller owns all-rank ordering.

    Returns (greedy{B: GreedyRows}, sampled{B: SampledRows}, receipts{B: dict}).
    A completed bitwise mismatch disables that B's component on every rank only.
    """
    from hybrid import dsv41_r1 as r1
    greedy, sampled, receipts = {}, {}, {}
    hidden_size = p.model.config.hidden_size
    anchor_index_all = p._anchor_idx[:max(pl.b for pl in contract.plans)].clone()
    anchor_saved = p.input_ids[anchor_index_all].clone()
    gen = p._dsv41_r1_generator()
    for plan in contract.plans:
        b = plan.b
        rec = receipts[b] = {}
        hidden = local(f"R2 B{b} hidden fixture",
                       lambda: torch.empty((N * b, hidden_size), device=p.device, dtype=torch.bfloat16))
        idx = anchor_index_all[:b]

        def fill(seed):
            g = torch.Generator(device=hidden.device).manual_seed(seed)
            hidden.copy_(torch.randn(hidden.shape, device=hidden.device, dtype=hidden.dtype, generator=g))
            p.input_ids.index_copy_(0, idx, (torch.arange(b, device=idx.device, dtype=p.input_ids.dtype) * 7919 + seed) % 129280)

        if "sampler" in plan.components:
            state = local(f"R2 B{b} greedy allocation", lambda: GreedyRows(p, nccl, start, end, b))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())

            def warm():
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        state.chain()
                stream.synchronize()
            collective(f"R2 B{b} greedy warmup", warm)
            state.graph = torch.cuda.CUDAGraph(keep_graph=True)

            def capture():
                with torch.cuda.graph(state.graph, stream=stream):
                    state.output = state.chain()
            collective(f"R2 B{b} greedy capture", capture)
            agree(True, f"R2 B{b} greedy capture ended on all ranks")
            collective(f"R2 B{b} greedy instantiate", state.graph.instantiate)
            state.ready = True
            exact, first = True, None
            for seed in (17, 23, 17):
                local(f"R2 B{b} greedy inputs {seed}", lambda: fill(seed))
                ref = collective("incumbent", lambda: p._sample_sequential_incumbent(b, hidden))
                got = collective("candidate", lambda: state.sample(hidden))
                torch.cuda.synchronize()
                exact = exact and torch.equal(ref, got) and (seed != 17 or first is None or torch.equal(got, first))
                first = got.clone() if first is None else first
            ok = all(consensus(bool(exact)))
            rec["sampler"] = dict(ready=ok, whole_call_bitwise=ok, seeds=[17, 23, 17])
            if ok:
                greedy[b] = state
            else:
                state.ready = False
        if "sampled" in plan.components:
            state = local(f"R2 B{b} sampled allocation", lambda: SampledRows(p, b, vocab, gen))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            saved = gen.get_state()

            def warm_s():
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        state.chain()
                stream.synchronize()
            collective(f"R2 B{b} sampled warmup", warm_s)
            state.graph = torch.cuda.CUDAGraph(keep_graph=True)
            state.graph.register_generator_state(gen)

            def capture_s():
                with torch.cuda.graph(state.graph, stream=stream):
                    state.drafts, state.q = state.chain()
            collective(f"R2 B{b} sampled capture", capture_s)
            agree(True, f"R2 B{b} sampled capture ended on all ranks")
            collective(f"R2 B{b} sampled instantiate", state.graph.instantiate)
            state.ready = True
            exact, restore_error = True, None
            patterns = ([1.0] * b, [1.0 if i % 2 == 0 else 0.0 for i in range(b)], [0.7] * b)
            for seed, pattern in zip((17, 23, 17), patterns):
                local(f"R2 B{b} sampled inputs {seed}", lambda: fill(seed))
                tq = torch.tensor(pattern, dtype=torch.float32, device=p.device)
                before = gen.get_state()
                base = p.model.compute_draft_logits(hidden).view(b, N, vocab)
                prev = p.input_ids[p._anchor_idx[:b]]
                ref_d, ref_q, _, _ = r1.draft_chain(p.model, base, prev, tq, gen,
                                                    all_sample=all(t > 0 for t in pattern))
                try:
                    gen.set_state(before)
                except RuntimeError as exc:
                    restore_error = repr(exc)
                    exact = False
                    break
                got_d, got_q = state.sample(hidden, tq)
                torch.cuda.synchronize()
                exact = exact and torch.equal(ref_d, got_d) and torch.equal(ref_q, got_q)
            try:
                gen.set_state(saved)
            except RuntimeError as exc:
                restore_error = restore_error or repr(exc)
                exact = False
            ok = all(consensus(bool(exact)))
            rec["sampled"] = dict(ready=ok, chain_bitwise_vs_eager=ok, patterns=["all", "alternate", "all_tq0.7"],
                                  generator_restored=restore_error is None, restore_error=restore_error)
            if ok:
                sampled[b] = state
            else:
                state.ready = False
    p.input_ids.index_copy_(0, anchor_index_all, anchor_saved)
    torch.cuda.synchronize()
    agree(torch.equal(p.input_ids[anchor_index_all], anchor_saved), "R2 sampler anchor restoration")
    return greedy, sampled, receipts
