"""R2 multi-user decode contract for B2..B8 (descriptors (6B, B, True, False, 0)).

Pure Python: boot flags, per-B component plans, the per-component per-B hit ledger,
the boot-time coop/stock slot cutoff and the qualification gate. No torch, no CUDA,
no environment writes. Every flag is parsed once at boot; malformed values raise, so
a bad launch plan fails before serving instead of silently degrading.

DSV41_R2 unset or "0" means R2 is off: no plan, no ledger, no hook changes behaviour.
"""
import hashlib
import json
from dataclasses import asdict, dataclass

VERSION = "dsv41-r2-multiuser-v1"
ROWS_PER_REQ = 6                 # DSpark N=5: five drafts plus the anchor row per request
TOPK = 6
HIDDEN = 5120
AR_BYTES_PER_ROW = HIDDEN * 2    # bf16 residual all-reduce
BATCHES = (2, 3, 4, 5, 6, 7, 8)
TARGET_COMPONENTS = ("attention", "experts", "collective")
SAMPLER_COMPONENTS = ("sampler", "sampled")
COMPONENTS = ("graph",) + TARGET_COMPONENTS + SAMPLER_COMPONENTS
EXPERT_KINDS = ("coop", "stock", "native", "tc")
COOP_SLOT_LIMIT = 256            # image exllamav3_ext launcher check (exl3_moe_coop.cu:219)
NATIVE_SLOT_LIMIT = 512          # R2 native cubin: MAX_SLOTS 512, 9-bit slot key
TC_V2_ROWS = (43, 64)            # /opt/dsv41-tc-moe/exl3_tc_hook.py: moe_out_v2 serves 43..64 rows when
                                 # VLLM_EXL3_TC_MOE_V2=1, ahead of coop/stock (R1's 48-row MoE)
SPLITS = (1, 2, 4, 8, 16)
WARPS = (4, 8)
AR_MAX_BLOCKS = 32
PAD_B = 7                        # B7 (42 rows) is served by the 48-row graph when pad7 is on
PAD_TO = 8
DROPPED_CAPTURE_SIZE = 42


def descriptor(b):
    return (ROWS_PER_REQ * b, b, True, False, 0)


def rows(b):
    return ROWS_PER_REQ * b


def slots(b):
    return rows(b) * TOPK


def ar_bytes(b):
    return rows(b) * AR_BYTES_PER_ROW


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------- parsing helpers
def _batches(text):
    out = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if lo > hi:
                raise ValueError(f"R2: bad batch range {part!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def _per_b(text, parse_value, name):
    """'2:a,3-5:b' -> {2: a, 3: b, 4: b, 5: b}; later entries override earlier ones."""
    table = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"R2: {name} entry {part!r} needs B:value")
        keys, value = part.split(":", 1)
        parsed = parse_value(value.strip())
        for b in _batches(keys):
            table[b] = parsed
    return table


def _split(value):
    """'8x4' or '8x4d' -> (splits, warps, direct)."""
    direct = value.endswith("d")
    body = value[:-1] if direct else value
    s, w = (int(x) for x in body.split("x", 1))
    if s not in SPLITS or w not in WARPS:
        raise ValueError(f"R2: attention split {value!r} outside splits {SPLITS} x warps {WARPS}")
    return (s, w, direct)


def _expert(value):
    if value not in EXPERT_KINDS:
        raise ValueError(f"R2: expert kind {value!r} not in {EXPERT_KINDS}")
    return value


def _collective(value):
    """'1x512a1' -> (blocks, threads, algo) for the host all-reduce; 'pair' -> the W19
    fixed-48-row P2P pair kernel (48-row graph only)."""
    if value == PAIR:
        return PAIR
    body, algo = value.split("a", 1)
    blocks, threads = (int(x) for x in body.split("x", 1))
    algo = int(algo)
    if not 1 <= blocks <= AR_MAX_BLOCKS:
        raise ValueError(f"R2: all-reduce blocks {blocks} outside 1..{AR_MAX_BLOCKS}")
    if not (32 <= threads <= 1024 and threads % 32 == 0):
        raise ValueError(f"R2: all-reduce threads {threads} must be a multiple of 32 in 32..1024")
    if algo not in (0, 1, 2):
        raise ValueError(f"R2: all-reduce algo {algo} not in 0/1/2")
    if algo == 2 and blocks != 1:
        raise ValueError("R2: DMA publish (algo 2) needs a single block")
    return (blocks, threads, algo)


def default_split(b):
    """Pre-sweep default: partial-kernel CTAs (rows x splits) nearest the qualified B1 96."""
    target = 96.0 / rows(b)
    best = min(SPLITS, key=lambda s: (abs(_log2(s) - _log2(target)), s))
    return (best, 8, False)


def _log2(x):
    import math
    return math.log2(x)


def tc_rows(b):
    return TC_V2_ROWS[0] <= rows(b) <= TC_V2_ROWS[1]


def default_expert(b, pad7, tc_v2=False):
    """The kind the R1 path would run for the served row count (ties go to the fallback's kernel)."""
    served = PAD_TO if (pad7 and b == PAD_B) else b
    if tc_v2 and tc_rows(served):
        return "tc"
    return "coop" if slots(served) <= COOP_SLOT_LIMIT else "stock"


PAIR = "pair"
PAIR_ROWS = 48                     # pair_arena.NBYTES = 491520 = 48 rows x 5120 x bf16
HOST_THREADS = 1024                # host_shm_all_reduce DEFAULT_THREADS (block cap 1)
HOST_DMA_MIN_BYTES = 256 * 1024    # host_shm_all_reduce DMA_MIN_BYTES


def default_collective(b):
    """Pre-sweep default: the W20 pair kernel at 48 rows; elsewhere exactly the geometry the
    stock size rule picks (one block, 1024 threads, DMA publish from 256 KiB, else two-shot)."""
    if rows(b) == PAIR_ROWS:
        return PAIR
    return (1, HOST_THREADS, 2 if ar_bytes(b) >= HOST_DMA_MIN_BYTES else 1)


# ---------------------------------------------------------------- boot-time coop cutoff
@dataclass(frozen=True)
class CoopCutoff:
    """exl3.py:2773 decision for graphs outside R2 private captures.

    Default (256 slots, no token list) is byte-for-byte the R1 condition
    tokens * topk <= 256.
    """
    max_slots: int = COOP_SLOT_LIMIT
    tokens: tuple = None

    def allows(self, tokens, topk):
        if tokens * topk > min(self.max_slots, COOP_SLOT_LIMIT):
            return False
        return self.tokens is None or tokens in self.tokens

    @classmethod
    def from_env(cls, env):
        raw = env.get("VLLM_EXL3_COOP_MAX_SLOTS", "")
        max_slots = COOP_SLOT_LIMIT if raw in ("",) else int(raw)
        if not 0 <= max_slots <= COOP_SLOT_LIMIT:
            raise ValueError(f"VLLM_EXL3_COOP_MAX_SLOTS={max_slots} outside 0..{COOP_SLOT_LIMIT}")
        raw_tokens = env.get("VLLM_EXL3_COOP_TOKENS", "").strip()
        tokens = None
        if raw_tokens:
            tokens = tuple(sorted(_batches(raw_tokens)))
            if any(t < 1 for t in tokens):
                raise ValueError("VLLM_EXL3_COOP_TOKENS must list positive token counts")
        return cls(max_slots, tokens)

    def record(self):
        return dict(max_slots=self.max_slots, tokens=None if self.tokens is None else list(self.tokens))


# ---------------------------------------------------------------- per-B plan
@dataclass(frozen=True)
class BPlan:
    b: int
    descriptor: tuple
    served_by: tuple           # descriptor whose R2 graph runs this B
    components: tuple          # enabled components (subset of COMPONENTS minus graph)
    split: tuple               # (splits, warps, direct) for the served row count
    experts: str
    collective: object         # (blocks, threads, algo) for ar_bytes(served B), or PAIR
    trace_budget: int

    @property
    def served_b(self):
        return self.served_by[1]

    def record(self):
        d = asdict(self)
        d["descriptor"] = list(self.descriptor)
        d["served_by"] = list(self.served_by)
        d["components"] = list(self.components)
        d["split"] = list(self.split)
        d["collective"] = self.collective if self.collective == PAIR else list(self.collective)
        return d


@dataclass(frozen=True)
class Contract:
    plans: tuple               # BPlan per enabled B, ascending
    pad7: bool
    cutoff: CoopCutoff
    trace_b1: int
    trace_stride: int = 1

    # -- construction
    @classmethod
    def from_env(cls, env):
        """None when R2 is off. Raises ValueError on any malformed or inconsistent flag."""
        cutoff = CoopCutoff.from_env(env)
        if env.get("DSV41_R2", "0") in ("", "0"):
            return None
        if env.get("DSV41_R2") != "1":
            raise ValueError("DSV41_R2 must be 0 or 1")
        batches = _batches(env.get("DSV41_R2_BATCHES", "2-8"))
        if not batches or not batches <= set(BATCHES):
            raise ValueError(f"DSV41_R2_BATCHES must be a non-empty subset of {BATCHES}")
        comps = tuple(c.strip() for c in env.get(
            "DSV41_R2_COMPONENTS", ",".join(COMPONENTS[1:])).split(",") if c.strip())
        if not comps or any(c not in COMPONENTS[1:] for c in comps) or len(set(comps)) != len(comps):
            raise ValueError(f"DSV41_R2_COMPONENTS must be distinct names from {COMPONENTS[1:]}")
        pad7 = env.get("DSV41_R2_PAD7", "1") == "1"
        if env.get("DSV41_R2_PAD7", "1") not in ("0", "1"):
            raise ValueError("DSV41_R2_PAD7 must be 0 or 1")
        if pad7 and PAD_B in batches and PAD_TO not in batches:
            raise ValueError("DSV41_R2_PAD7 serves B7 with the B8 graph; B8 must be enabled")
        splits = _per_b(env.get("DSV41_R2_SPLITS", ""), _split, "DSV41_R2_SPLITS")
        experts = _per_b(env.get("DSV41_R2_EXPERTS", ""), _expert, "DSV41_R2_EXPERTS")
        colls = _per_b(env.get("DSV41_R2_COLLECTIVE", ""), _collective, "DSV41_R2_COLLECTIVE")
        budgets = _per_b(env.get("DSV41_R2_TRACE_BUDGET", ""), int, "DSV41_R2_TRACE_BUDGET")
        if any(v < 0 for v in budgets.values()) or not set(budgets) <= set(BATCHES) | {1}:
            raise ValueError("DSV41_R2_TRACE_BUDGET: B in 1..8, non-negative round counts")
        stride = int(env.get("DSV41_R2_TRACE_STRIDE", "1") or "1")
        if stride < 1:
            raise ValueError("DSV41_R2_TRACE_STRIDE must be >= 1")
        tc_v2 = env.get("VLLM_EXL3_TC_MOE_V2", "0") == "1"
        plans = []
        for b in sorted(batches):
            served_b = PAD_TO if (pad7 and b == PAD_B) else b
            if pad7 and b == PAD_B:
                # The 48-row graph's settings are the B8 settings; a differing B7 value would be ignored.
                for name, table, default in (("SPLITS", splits, default_split(PAD_TO)),
                                             ("EXPERTS", experts, default_expert(PAD_TO, False, tc_v2)),
                                             ("COLLECTIVE", colls, default_collective(PAD_TO))):
                    if PAD_B in table and table[PAD_B] != table.get(PAD_TO, default):
                        raise ValueError(f"DSV41_R2_{name}: B7 is served by the B8 graph; set B8 instead")
            split = splits.get(served_b, default_split(served_b))
            kind = experts.get(served_b, default_expert(served_b, False, tc_v2))
            if kind == "tc" and not (tc_v2 and tc_rows(served_b)):
                raise ValueError(f"R2 B{b}: tc needs VLLM_EXL3_TC_MOE_V2=1 and a served graph of "
                                 f"{TC_V2_ROWS[0]}..{TC_V2_ROWS[1]} rows, not {rows(served_b)}")
            if kind == "coop" and slots(served_b) > COOP_SLOT_LIMIT:
                raise ValueError(f"R2 B{b}: coop needs <= {COOP_SLOT_LIMIT} slots, served graph has "
                                 f"{slots(served_b)}; use native or stock")
            if kind == "native" and slots(served_b) > NATIVE_SLOT_LIMIT:
                raise ValueError(f"R2 B{b}: native cubin holds <= {NATIVE_SLOT_LIMIT} slots")
            coll = colls.get(served_b, default_collective(served_b))
            if coll == PAIR and rows(served_b) != PAIR_ROWS:
                raise ValueError(f"R2 B{b}: the pair all-reduce is qualified for {PAIR_ROWS} rows only")
            plans.append(BPlan(b, descriptor(b), descriptor(served_b), comps, split, kind,
                               coll, budgets.get(b, 0)))
        return cls(tuple(plans), pad7, cutoff, budgets.get(1, 0), stride)

    # -- queries
    def plan(self, b):
        for p in self.plans:
            if p.b == b:
                return p
        return None

    def captured(self):
        """Distinct served plans, largest first (capture order for one shared private pool)."""
        seen = {}
        for p in self.plans:
            if p.served_by == p.descriptor:
                seen[p.served_by] = p
        return [seen[d] for d in sorted(seen, key=lambda d: -d[0])]

    def serves(self, desc, real_b):
        """Plan for a FULL replay of descriptor desc carrying real_b requests, else None."""
        p = self.plan(real_b)
        if p is None or tuple(desc) != p.served_by:
            return None
        return p

    def collective_for(self, nbytes):
        for p in self.captured():
            if ar_bytes(p.b) == nbytes:
                return p.collective
        return None

    def trace_budgets(self):
        out = {p.b: p.trace_budget for p in self.plans if p.trace_budget}
        if self.trace_b1:
            out[1] = self.trace_b1
        return out

    def check_capture_sizes(self, sizes):
        """Boot check of compilation_config.cudagraph_capture_sizes against the plan."""
        sizes = list(sizes)
        missing = [p.served_by[0] for p in self.captured() if p.served_by[0] not in sizes]
        if missing:
            raise ValueError(f"R2: capture sizes lack served row counts {missing}")
        if self.pad7 and self.plan(PAD_B) is not None and DROPPED_CAPTURE_SIZE in sizes:
            raise ValueError("R2 pad7: 42 must be dropped from cudagraph_capture_sizes")
        return True

    def record(self):
        return dict(version=VERSION, pad7=self.pad7, cutoff=self.cutoff.record(), trace_b1=self.trace_b1,
                    trace_stride=self.trace_stride, plans=[p.record() for p in self.plans])

    def digest(self):
        return digest(self.record())


def capture_sizes(base, pad7):
    """Launch-plan capture ladder for R2: the R1 ladder minus 42 when pad7 is on."""
    out = [s for s in base if not (pad7 and s == DROPPED_CAPTURE_SIZE)]
    if out != sorted(set(out)):
        raise ValueError("capture sizes must be strictly increasing")
    return out


def attention_workspace_bytes(b, split):
    r, s = rows(b), split[0]
    return (r * s * 16 * 2 + r * s * 16 * 512) * 4 + r * 16 * 512 * 2 + 4


# ---------------------------------------------------------------- ledger
class Ledger:
    """Per-component per-B counters. Host-side O(1) dict updates on the step path."""

    def __init__(self):
        self.counts = {}

    def _cell(self, b, comp):
        key = (int(b), comp)
        cell = self.counts.get(key)
        if cell is None:
            cell = self.counts[key] = {"hit": 0, "fb": {}}
        return cell

    def hit(self, b, comp, n=1):
        self._cell(b, comp)["hit"] += n

    def fallback(self, b, comp, reason):
        fb = self._cell(b, comp)["fb"]
        fb[reason] = fb.get(reason, 0) + 1

    def snapshot(self):
        return {f"{b}:{c}": {"hit": v["hit"], "fb": dict(v["fb"])} for (b, c), v in self.counts.items()}

    @staticmethod
    def delta(before, after):
        out = {}
        for key, v in after.items():
            old = before.get(key, {"hit": 0, "fb": {}})
            hit = v["hit"] - old["hit"]
            fb = {r: n - old["fb"].get(r, 0) for r, n in v["fb"].items() if n != old["fb"].get(r, 0)}
            if hit or fb:
                out[key] = {"hit": hit, "fb": fb}
        return out


class TraceBudget:
    """Bounded boot-time auto trace: every stride-th pure-decode round per B, N rounds per B.

    A stride > 1 spreads the traced rounds over many batches (and so over both epochs of an
    A/B harness) instead of spending the budget on the first batch at each B."""

    def __init__(self, budgets, stride=1):
        if int(stride) < 1:
            raise ValueError("trace stride must be >= 1")
        self.stride = int(stride)
        self.left = {int(b): int(n) for b, n in budgets.items() if n > 0}
        self.used = {b: 0 for b in self.left}
        self.seen = {b: 0 for b in self.left}

    def take(self, b):
        n = self.left.get(b, 0)
        if n <= 0:
            return False
        self.seen[b] += 1
        if (self.seen[b] - 1) % self.stride:
            return False
        self.left[b] = n - 1
        self.used[b] += 1
        return True

    def exhausted(self):
        return all(v == 0 for v in self.left.values())


# ---------------------------------------------------------------- gate
def round_components(rec):
    """Expected components and the observed outcome per component for one round record.

    rec keys: B (real requests), uniform6 (descriptor was (6B,B,True,False,0) before
    padding), sampled (any row drafts with T_q > 0), r2 (ledger delta for the round).
    """
    b = rec.get("B")
    r2 = rec.get("r2")
    if r2 is None:
        return None
    out = {}
    for comp in ("graph",) + TARGET_COMPONENTS:
        out[comp] = r2.get(f"{b}:{comp}")
    out["draft"] = r2.get(f"{b}:{'sampled' if rec.get('sampled') else 'sampler'}")
    return out


GATE_COMPONENTS = ("graph",) + TARGET_COMPONENTS + ("draft",)
GATE_ALIASES = {"sampler": "draft", "sampled": "draft"}


def gate_components(components):
    """Normalised gate component names; sampler/sampled mean the round's drafter counter."""
    if not components:
        return GATE_COMPONENTS
    out = []
    for c in components:
        c = GATE_ALIASES.get(c, c)
        if c not in GATE_COMPONENTS:
            raise ValueError(f"R2 gate: unknown component {c!r}; use {GATE_COMPONENTS}")
        if c not in out:
            out.append(c)
    return tuple(out)


def stray(rec):
    """Ledger cells of another B with hits, and every cell with a fallback, in the round's delta."""
    b = rec.get("B")
    out = []
    for key, cell in sorted((rec.get("r2") or {}).items()):
        if cell.get("fb"):
            out.append((key, ",".join(sorted(cell["fb"]))))
        elif not key.startswith(f"{b}:") and cell.get("hit", 0):
            out.append((key, "foreign_b"))
    return out


def gate(rounds, batches=(2, 3, 4, 5, 6, 7, 8), min_rounds=1, components=None):
    """FAIL on any uniform B2..B8 decode round where any enabled component did not hit.

    Returns dict(passed, failures, per_b, non_uniform, incomplete, missing). Non-uniform
    decode rounds (mixed query lengths, prefill chunks) are counted and reported, not
    gated: no (6B, B) descriptor exists for them. Incomplete uniform rounds are not
    required to hit, but any fallback recorded in them fails, as does a fallback under
    any ledger key in a gated round.
    """
    comps = gate_components(components)
    per_b = {b: {"rounds": 0, **{c: {"hit": 0, "fb": {}} for c in comps}} for b in batches}
    non_uniform = {b: 0 for b in batches}
    incomplete = {b: 0 for b in batches}
    failures = []
    for rec in rounds:
        b = rec.get("B")
        if b not in per_b:
            continue
        if not rec.get("uniform6"):
            non_uniform[b] += 1
            continue
        ungated = {f"{b}:{c}" for c in ("graph",) + TARGET_COMPONENTS if c not in comps}
        if "draft" not in comps:
            ungated |= {f"{b}:sampler", f"{b}:sampled"}
        if rec.get("complete", True) is False:
            incomplete[b] += 1
            for key, reason in stray(rec):
                if key not in ungated:
                    failures.append(dict(it=rec.get("it"), B=b, component=key, reason="incomplete:" + reason))
            continue
        per_b[b]["rounds"] += 1
        got = round_components(rec)
        if got is None:
            failures.append(dict(it=rec.get("it"), B=b, component="all", reason="no_r2_record"))
            continue
        own = {f"{b}:{c}" for c in ("graph",) + TARGET_COMPONENTS if c in comps}
        if "draft" in comps:
            own.add(f"{b}:{'sampled' if rec.get('sampled') else 'sampler'}")
        for key, reason in stray(rec):
            if key not in own and key not in ungated:
                failures.append(dict(it=rec.get("it"), B=b, component=key, reason="stray:" + reason))
        for comp in comps:
            cell = got.get(comp)
            if cell is None:
                failures.append(dict(it=rec.get("it"), B=b, component=comp, reason="no_counter"))
                per_b[b][comp]["fb"]["no_counter"] = per_b[b][comp]["fb"].get("no_counter", 0) + 1
                continue
            per_b[b][comp]["hit"] += cell.get("hit", 0)
            for r, n in cell.get("fb", {}).items():
                per_b[b][comp]["fb"][r] = per_b[b][comp]["fb"].get(r, 0) + n
            if cell.get("fb") or cell.get("hit", 0) != 1:
                failures.append(dict(it=rec.get("it"), B=b, component=comp,
                                     reason=",".join(sorted(cell.get("fb", {}))) or f"hits={cell.get('hit', 0)}"))
    missing = [b for b in batches if per_b[b]["rounds"] < min_rounds]
    return dict(passed=not failures and not missing, failures=failures[:200], failure_count=len(failures),
                per_b=per_b, non_uniform=non_uniform, incomplete=incomplete, missing=missing)
