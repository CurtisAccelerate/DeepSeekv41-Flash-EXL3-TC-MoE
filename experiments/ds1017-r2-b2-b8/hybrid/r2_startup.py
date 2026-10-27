"""R2 startup: per-B private FULL captures and per-B samplers, initial RPC only.

Called from cumulative_startup.extend_worker after the B1 sampler preparation and before
the family identity is built, only when DSV41_R2=1. Same discipline as the B1 banks:
all-rank votes before and after every collective phase, journal restore around every
dummy, completed rejections disable only the affected B/component, uncertain CUDA or
communication state propagates as unsafe. Nothing here runs after serving starts.
"""
import hashlib
import json
import os

from .r2_contract import PAIR, Contract, ar_bytes, rows as rows_of


class R2Rejected(ValueError):
    pass


# Per request, the same byte budget the B1 journal gets for its one request.
JOURNAL_BUDGET_PER_REQUEST = 16 * 1024 ** 2


def journal_rows(contract):
    """(rows, requests) the R2 state journal covers: those of the largest captured R2 graph."""
    top = max(p.served_by[0] for p in contract.captured())
    return top, top // 6


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def prepare(worker, p, old, consensus, start, end, vocab):
    """Returns (bank or None, receipt). bank is attached to the adapter by the caller."""
    import torch
    from vllm.distributed import graph_capture
    from w9_exl_ab import runtime as original
    from w9_exl_ab import collective_profiles as CP
    from . import r2_bank, r2_experts, r2_sampler
    from .attention_qualification import QualificationRejected, qualify_replay
    from .attention_startup import UnsafePreparation
    from .initial_decode_contract import CompletedMetadataRejection
    from .initial_state_plan import InitialJournal, collect
    from .r2_attention import AttentionStateRows, RowsProfile

    contract = Contract.from_env(os.environ)
    receipt = dict(enabled=contract is not None, ready=False)
    if contract is None:
        return None, receipt
    runner = worker.model_runner
    roots = getattr(worker, "_w26_cumulative_roots", [])
    stage = [0]

    def agree(value, reason):
        stage[0] += 1
        tag = (stage[0], reason)
        rows = consensus(dict(phase=tag, ok=bool(value)))
        if not all(x.get("phase") == tag and x.get("ok") is True for x in rows):
            raise R2Rejected(reason)

    def local(name, fn):
        value, error = None, None
        try:
            value = fn()
            roots.append(value)
        except Exception as exc:
            torch.cuda.synchronize()
            error = repr(exc)
        if error is not None:
            receipt.setdefault("local_errors", []).append(dict(phase=name, error=error))
        agree(error is None, name)
        return value

    def collective(name, fn):
        try:
            return fn()
        except BaseException as exc:
            raise UnsafePreparation("R2 collective/capture phase uncertain: " + name) from exc

    if old is None or getattr(old, "runtime", None) is None:
        receipt["reason"] = "R2 requires the prepared B1 attention runtime"
        return None, receipt
    rt1 = old.runtime
    native = rt1.native
    journal = next((r for r in getattr(worker, "_w26_attention_roots", []) if type(r).__name__ == "InitialJournal"), None)
    try:
        agree(journal is not None, "R2 needs the initial state journal")
        agree(len(set(consensus(contract.digest()))) == 1, "R2 contract identical on all ranks")
        local("R2 capture ladder", lambda: contract.check_capture_sizes(runner.compilation_config.cudagraph_capture_sizes))
        cap = int(os.environ.get("DSV41_R2_CAP_MIB", "768")) * 1024 ** 2
        torch.cuda.synchronize()
        before_free = torch.cuda.mem_get_info(runner.device)[0]
        agree(before_free >= cap + 1024 ** 3, "R2 reserve above the 1 GiB floor")
        if any(pl.experts == "native" for pl in contract.captured()):
            local("R2 native cubin", lambda: r2_experts.load(runner.device))
        pool = local("R2 private pool", torch.cuda.graph_pool_handle)
        agree(pool != rt1.wrapper.graph_pool and pool != rt1.pool, "R2 pool distinct from incumbent and B1 pools")

        journals = [journal]

        def restore():
            for j in journals:
                j.restore()
            rows = consensus(dict(restore=all(j.verify() for j in journals)))
            if not all(x.get("restore") is True for x in rows):
                raise UnsafePreparation("R2 state restoration failed")

        # The B1 journal covers rows [:6] and request row 0 only. R2 dummies write rows up to
        # 6B and request rows up to B-1, so a second journal over the largest R2 graph is taken
        # from the restored B1 state and restored and verified around every R2 dummy.
        restore()
        jrows, jreqs = journal_rows(contract)
        r2_journal = local("R2 state journal", lambda: InitialJournal(
            collect(runner, phase="initial_before_first_request", rows=jrows, reqs=jreqs),
            budget_bytes=JOURNAL_BUDGET_PER_REQUEST * jreqs))
        journals.insert(0, r2_journal)
        receipt["journal"] = dict(rows=jrows, reqs=jreqs, bytes=r2_journal.gpu.bytes)

        bank = r2_bank.R2Bank(contract, None, rt1.wrapper, runner, vocab)
        graphs = {}
        for plan in contract.captured():
            b = plan.b
            excluded = {c: "disabled" for c in ("attention", "experts", "collective") if c not in plan.components}
            rec = graphs[b] = dict(plan=plan.record())
            owner = CP.REG[0]["owner"] if CP.REG else None
            if "collective" in plan.components and plan.collective == PAIR:
                # A rank without the pair route would take the host protocol while its peers
                # spin in the pair kernel; decide on every rank before any pair launch.
                try:
                    ready = owner is not None and r2_bank.pair_ready(owner, rows_of(b))
                except Exception:
                    ready = False
                if not all(consensus(bool(ready))):
                    excluded["collective"] = "pair_unavailable"
            # Collective parity: planned geometry vs the incumbent selection, eager, same inputs.
            if "collective" in plan.components and "collective" not in excluded:
                def parity():
                    if owner is None:
                        raise ValueError("host all-reduce registry missing")
                    g = torch.Generator(device=runner.device).manual_seed(1017 + int(worker.rank))
                    shape = (rows_of(b), 5120)
                    base = torch.randn(shape, device=runner.device, generator=g)
                    sign = -1.0 if int(worker.rank) % 2 else 1.0
                    inputs = (base, base * 1024.0 * sign + torch.randn(shape, device=runner.device, generator=g),
                              base * torch.logspace(-6, 6, 5120, device=runner.device))
                    ctx = r2_bank.CaptureContext(plan)
                    equal = True
                    for x in inputs:
                        x = x.to(torch.bfloat16)
                        ref = owner.host_all_reduce(x)
                        r2_bank.ACTIVE = ctx
                        try:
                            got = owner.host_all_reduce(x)
                        finally:
                            r2_bank.ACTIVE = None
                        equal = equal and bool(torch.equal(ref, got))
                    torch.cuda.synchronize()
                    return equal and ctx.collective["planned"] == len(inputs) and ctx.collective["unavailable"] == 0
                ok = collective(f"R2 B{b} collective parity", parity)
                ok = all(consensus(bool(ok)))
                rec["collective_parity"] = dict(bytes=ar_bytes(b), profile=plan.record()["collective"], bitwise=ok)
                if not ok:
                    excluded["collective"] = "parity_rejected"
            state = None
            if "attention" in plan.components:
                prof = RowsProfile(plan.served_by[0], *plan.split)
                try:
                    state = local(f"R2 B{b} attention workspace", lambda: AttentionStateRows(runner.device, prof))
                    rec["attention_probe"] = local(f"R2 B{b} attention probe", state.compile_probe)
                except R2Rejected:
                    state = None
                    excluded["attention"] = "probe_rejected"
            if all(c in excluded for c in ("attention", "experts", "collective")):
                rec["skipped"] = "no target component left"
                continue
            plan_used = plan
            if "collective" in excluded and "collective" in plan.components:
                from dataclasses import replace
                plan_used = replace(plan, components=tuple(c for c in plan.components if c != "collective"))
            rt2 = r2_bank.R2Runtime(runner, native, before_free, pool, plan_used, state)
            rt2.initial_consensus = consensus
            rt2.initial_metadata_records = []
            roots.append(rt2)
            r2_bank.CAPTURING = rt2
            r2_bank.DUMMY_ROWS = (rt2.rows,)
            try:
                with graph_capture(runner.device, original.S.startup_capture_context):
                    restore()
                    rt2.mode = "capture"
                    try:
                        rt2.dummy("FULL", target_only=True)
                    except CompletedMetadataRejection:
                        raise
                    except BaseException as exc:
                        raise UnsafePreparation(f"R2 B{b} private FULL capture failed") from exc
                    agree(True, f"R2 B{b} capture completed on every rank")
                    rt2.mode = "baseline"
                    restore()
                    receipts = rt2.receipts()
                    paths = consensus(_sha(r2_bank.collective_path(receipts["collective"])))
                    if len(set(paths)) != 1:
                        raise R2Rejected(f"R2 B{b} all-reduce protocol differs across ranks; graph not replayed")
                    votes = consensus({k: bool(v.get("ok")) for k, v in receipts.items()})
                    for comp, row in receipts.items():
                        if not all(v.get(comp) for v in votes):
                            excluded.setdefault(comp, "receipt_rejected")
                    rec["receipts"] = receipts
                    rec["replay"] = qualify_replay(rt2, r2_journal, native, consensus)
                restore()
                comps = [c for c in ("attention", "experts", "collective") if c not in excluded]
                bank.publish(plan.served_by, rt2.entry, comps, excluded)
                rec.update(published=True, components=comps, excluded=excluded,
                           state_sha256=_sha(dict(receipts=rec["receipts"], replay=rec["replay"])))
            except (R2Rejected, QualificationRejected, CompletedMetadataRejection) as exc:
                rt2.mode = "baseline"
                restore()
                rec.update(published=False, reason=str(exc))
            finally:
                r2_bank.CAPTURING = None
                r2_bank.DUMMY_ROWS = ()
        receipt["graphs"] = graphs
        greedy, sampled, srec = r2_sampler.prepare_rows(p, _nccl(), contract, consensus, agree, local, collective,
                                                        start, end, vocab)
        bank.greedy, bank.sampled = greedy, sampled
        receipt["samplers"] = srec
        torch.cuda.synchronize()
        free = torch.cuda.mem_get_info(runner.device)[0]
        growth = max(0, before_free - free)
        agree(growth <= cap and free >= 1024 ** 3, "R2 retained memory within DSV41_R2_CAP_MIB and floor")
        r2_bank.TRACE = _trace(contract)
        receipt.update(ready=True, memory=dict(growth=growth, cap=cap, free=free), bank=bank.record(),
                       contract=contract.record(), contract_sha256=contract.digest())
        return bank, receipt
    except R2Rejected as exc:
        receipt.update(ready=False, reason=str(exc), baseline_retained=True)
        return None, receipt


def _nccl():
    from vllm.distributed import get_tp_group
    return getattr(get_tp_group().device_communicator, "pynccl_comm", None)


def _trace(contract):
    from .r2_contract import TraceBudget
    budgets = contract.trace_budgets()
    if not budgets or os.environ.get("DSV41_R1_TRACE_MODE", "off") == "off":
        return None
    return TraceBudget(budgets, contract.trace_stride)


def family_receipt(receipt):
    """Rank-common part for FamilyIdentity.attention_banks['r2'] plus the rank state hash."""
    graphs = receipt.get("graphs", {})
    common = dict(version="dsv41-r2-multiuser-v1", contract_sha256=receipt["contract_sha256"],
                  graphs={str(b): dict(published=g.get("published", False), components=g.get("components", []))
                          for b, g in sorted(graphs.items())},
                  samplers={str(b): {k: v.get("ready", False) for k, v in s.items()}
                            for b, s in sorted(receipt.get("samplers", {}).items())})
    return dict(common, state_receipt_sha256=_sha(receipt))
