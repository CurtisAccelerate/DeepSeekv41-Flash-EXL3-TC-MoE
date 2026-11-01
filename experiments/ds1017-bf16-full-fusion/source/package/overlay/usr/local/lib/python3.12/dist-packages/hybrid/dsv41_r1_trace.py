"""Reload R1 step trace (gate G2) and the background JSONL sink. Default OFF.

Only reached for steps that contain a request with ``extra_args["dsv41_trace"]`` while the
runner is armed (DSV41_R1_FEATURES=1) and DSV41_R1_OUT is set. Each rank traces itself; no
collective is involved and drafting is never affected, so ranks may differ in trace mode.

Modes (boot env DSV41_R1_TRACE_MODE, default "auto"):
  cupti   CUPTI activity API through ctypes: CONCURRENT_KERNEL (+ MEMCPY) records, including
          kernels launched by CUDA graph replays (graphNodeId != 0). Phase boundaries are
          zero-cycle ``torch.cuda._sleep(0)`` marker kernels on the serving stream, so every
          kernel is attributed to the runner phase it executed in. Activity is enabled at the
          first launch of a traced step and disabled once that step's last marker has
          completed; kernels outside traced windows are discarded.
  events  torch.cuda.Event pairs at the same phase boundaries. Loses per-kernel resolution:
          no attention / indexer / experts / dense / collective split inside the target
          forward, and no host gap (event spans include idle time).
  auto    cupti when libcupti loads and its self-test passes, else events.
  off     trace flags are ignored.
DSV41_R1_CUPTI_LIB=<path> pins the library; if that path does not load, CUPTI stays off.
"""
from __future__ import annotations

import bisect
import collections
import ctypes
import glob
import json
import logging
import os
import queue
import re
import struct
import sys
import threading
import time

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

logger = logging.getLogger("vllm.dsv41.r1")

VERSION = "dsv41-r1-trace-v1"


# ---------------------------------------------------------------- JSONL sink
class JsonlSink:
    """Bounded queue drained by one daemon thread. put() never blocks and never does I/O.

    Items are dicts, lists of dicts, or callables returning either (run on the writer thread;
    they must not call CUDA).
    """

    def __init__(self, path: str, maxsize: int = 4096):
        self.path = path
        self.dropped = 0
        self.errors = 0
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._run, name="dsv41-r1-sink", daemon=True)
        self._thread.start()

    def put(self, item) -> bool:
        try:
            self._q.put_nowait(item)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def drain(self, timeout: float = 5.0) -> None:
        """Test helper: wait until everything queued so far is written and flushed."""
        done = threading.Event()
        self.put(done)
        done.wait(timeout)

    def _run(self) -> None:
        f = None
        while True:
            item = self._q.get()
            try:
                if isinstance(item, threading.Event):
                    if f is not None:
                        f.flush()
                    item.set()
                    continue
                rows = item() if callable(item) else item
                if rows is None:
                    continue
                if isinstance(rows, dict):
                    rows = [rows]
                if f is None:
                    os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                    f = open(self.path, "a", encoding="utf-8")
                for r in rows:
                    if self.dropped:
                        r = dict(r, sink_dropped=self.dropped)
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
                if self._q.empty():
                    f.flush()
            except Exception:
                self.errors += 1
                if self.errors == 1:
                    logger.exception("R1 sink %s: write failed", self.path)


# ---------------------------------------------------------------- classification
COMPONENTS = (
    "attention", "indexer", "routed_experts", "dense_gemm", "collectives",
    "drafter_forward", "draft_sampler", "target_sampling", "other", "marker",
)
_NAME_CLASSES = (
    ("collectives", re.compile(
        r"nccl|oneshot|twoshot|all_?reduce|allgather|all_gather|reduce_?scatter|"
        r"cross_device|custom_ar", re.I)),
    ("routed_experts", re.compile(
        r"moe|expert|grouped_gemm|mgemm|permute|topk_softmax|topkgating|router", re.I)),
    ("indexer", re.compile(r"indexer|mqa_logits|topk|top_k|radix_select", re.I)),
    ("attention", re.compile(
        r"attn|attention|flash|fmha|mla|paged|swa|softmax_lse|compress|kv_cache|rotary|rope",
        re.I)),
    ("dense_gemm", re.compile(
        r"gemm|gemv|cutlass|xmma|cublas|sm80|ampere|matmul|exl3|marlin|splitk|linear", re.I)),
)
# None: classify by kernel name (the target forward); else every kernel of the phase
PHASE_COMPONENT = {
    "prepare": "other",
    "target_forward": None,
    "target_sample": "target_sampling",
    "drafter_prep": "drafter_forward",
    "draft_inputs": "drafter_forward",
    "draft_forward": "drafter_forward",
    "draft_sampler": "draft_sampler",
    "bookkeep": "other",
    "end": "other",
}
MARKER_NAME = "spin_kernel"


def classify_kernel(name: str) -> str:
    for cls, rx in _NAME_CLASSES:
        if rx.search(name):
            return cls
    return "other"


def interval_union(intervals) -> int:
    total = 0
    cur_s = cur_e = None
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def summarize_step(kernels, markers, phases, meta, level: int = 1) -> dict:
    """Pure function: one traced step -> JSON record.

    kernels: iterable of (start_ns, end_ns, name, stream, graph_node, kind, nbytes), kind 'k'
    (kernel) or 'm' (memcpy); markers: [(start_ns, end_ns)] in launch order, one per phase;
    phases[-1] is "end". Phase i spans [marker_i.start, marker_{i+1}.start).
    """
    assert len(markers) == len(phases) and len(phases) >= 2
    t0, t1 = markers[0][0], markers[-1][1]
    starts = [m[0] for m in markers]
    comp_iv = collections.defaultdict(list)
    comp_sum = collections.Counter()
    ph_iv = collections.defaultdict(list)
    ph_cls = collections.defaultdict(collections.Counter)
    ph_n = collections.Counter()
    names = {}
    all_iv = []
    kept = []
    n_graph = n_memcpy = memcpy_bytes = n_coll = 0
    for m in markers:
        comp_iv["marker"].append(m)
        all_iv.append(m)
    for k in kernels:
        s, e, name, stream, gnode, kind, nbytes = k
        if s < t0 or s > t1 or (kind == "k" and MARKER_NAME in name):
            continue
        e_c = min(e, t1)
        pi = bisect.bisect_right(starts, s) - 1
        ph = phases[pi]
        if kind == "m":
            cls = "memcpy"
            n_memcpy += 1
            memcpy_bytes += nbytes or 0
        else:
            cls = classify_kernel(name)
        if cls == "collectives":
            n_coll += 1
        comp = PHASE_COMPONENT.get(ph, "other")
        if comp is None:
            comp = "collectives" if cls == "memcpy" else cls
        comp_iv[comp].append((s, e_c))
        comp_sum[comp] += e - s
        ph_iv[ph].append((s, e_c))
        ph_cls[ph][cls] += e - s
        ph_n[ph] += 1
        all_iv.append((s, e_c))
        if gnode:
            n_graph += 1
        agg = names.setdefault(name, [0, 0, comp])
        agg[0] += 1
        agg[1] += e - s
        kept.append(k)
    span = t1 - t0
    union = interval_union(all_iv)
    phase_rows = []
    for i, ph in enumerate(phases[:-1]):
        p_s, p_e = starts[i], starts[i + 1]
        busy = interval_union(ph_iv[ph]) if phases.count(ph) == 1 else None
        phase_rows.append({
            "phase": ph,
            "span_us": round((p_e - p_s) / 1e3, 3),
            "busy_us": None if busy is None else round(busy / 1e3, 3),
            "n": ph_n[ph],
            "classes_us": {c: round(v / 1e3, 3) for c, v in ph_cls[ph].items()},
        })
    top = sorted(names.items(), key=lambda kv: -kv[1][1])[:25]
    rec = dict(meta)
    rec.update({
        "type": "trace",
        "gpu_span_us": round(span / 1e3, 3),
        "kernel_union_us": round(union / 1e3, 3),
        "host_gap_us": round((span - union) / 1e3, 3),
        "components_us": {c: round(interval_union(comp_iv[c]) / 1e3, 3) for c in COMPONENTS},
        "components_sum_us": {c: round(comp_sum[c] / 1e3, 3) for c in COMPONENTS if comp_sum[c]},
        "phases": phase_rows,
        "n_kernels": sum(ph_n.values()),
        "n_graph_kernels": n_graph,
        "n_collective_kernels": n_coll,
        "n_memcpy": n_memcpy,
        "memcpy_bytes": memcpy_bytes,
        "top_names": [[n[:160], a[0], round(a[1] / 1e3, 3), a[2]] for n, a in top],
    })
    if level >= 2:
        table = {}
        rows = []
        for s, e, name, stream, gnode, kind, nbytes in sorted(kept):
            idx = table.setdefault(name, len(table))
            rows.append([s - t0, e - s, idx, stream, gnode, kind, nbytes])
        rec["kernel_names"] = list(table)
        rec["kernels"] = rows
    return rec


def summarize_events(phases, durations_ms, meta) -> dict:
    rec = dict(meta)
    comp = collections.Counter()
    rows = []
    for ph, ms in zip(phases[:-1], durations_ms):
        c = PHASE_COMPONENT.get(ph, "other") or "target_forward_unsplit"
        comp[c] += ms * 1e3
        rows.append({"phase": ph, "span_us": round(ms * 1e3, 3)})
    rec.update({
        "type": "trace",
        "gpu_span_us": round(sum(durations_ms) * 1e3, 3),
        "kernel_union_us": None,
        "host_gap_us": None,
        "components_us": {c: round(v, 3) for c, v in comp.items()},
        "phases": rows,
    })
    return rec


# ---------------------------------------------------------------- CUPTI (ctypes)
KIND_MEMCPY = 1
KIND_CONCURRENT_KERNEL = 10
# CUpti_ActivityKernel4..9 and CUpti_ActivityMemcpy5 field offsets (packed, 8-aligned).
# The self-test below rejects the layout (and falls back to events) if they do not hold.
KERNEL_FIELDS = {"start": 16, "end": 24, "device": 40, "stream": 48, "corr": 92,
                 "name": 104, "graph_node": 144}
MEMCPY_FIELDS = {"copy_kind": 4, "bytes": 8, "start": 16, "end": 24, "device": 32,
                 "stream": 40, "graph_node": 64}
KERNEL_READ = 160
MEMCPY_READ = 80


def parse_kernel_record(raw: bytes, name_of):
    f = KERNEL_FIELDS
    start, end = struct.unpack_from("<QQ", raw, f["start"])
    device = struct.unpack_from("<I", raw, f["device"])[0]
    stream = struct.unpack_from("<I", raw, f["stream"])[0]
    name_ptr = struct.unpack_from("<Q", raw, f["name"])[0]
    gnode = struct.unpack_from("<Q", raw, f["graph_node"])[0]
    return (start, end, name_of(name_ptr), stream, gnode, "k", 0, device)


def parse_memcpy_record(raw: bytes):
    f = MEMCPY_FIELDS
    nbytes = struct.unpack_from("<Q", raw, f["bytes"])[0]
    start, end = struct.unpack_from("<QQ", raw, f["start"])
    device = struct.unpack_from("<I", raw, f["device"])[0]
    stream = struct.unpack_from("<I", raw, f["stream"])[0]
    gnode = struct.unpack_from("<Q", raw, f["graph_node"])[0]
    copy_kind = raw[f["copy_kind"]]
    return (start, end, f"memcpy{copy_kind}", stream, gnode, "m", nbytes, device)


def cupti_candidates() -> list:
    pinned = os.environ.get("DSV41_R1_CUPTI_LIB")
    if pinned:
        return [pinned]
    out = []
    for d in ("/usr/local/cuda/extras/CUPTI/lib64", "/usr/local/cuda/lib64",
              "/usr/local/cuda/targets/x86_64-linux/lib"):
        out += [f"{d}/libcupti.so", f"{d}/libcupti.so.12", f"{d}/libcupti.so.13"]
    for base in sys.path:
        if base and os.path.isdir(os.path.join(base, "nvidia")):
            out += sorted(glob.glob(os.path.join(base, "nvidia", "*", "lib", "libcupti.so*")))
    return out


class Cupti:
    BUF = 8 << 20
    PAD = 512
    MAX_BUFFERS = 64

    def __init__(self, path: str):
        self.path = path
        L = self.lib = ctypes.CDLL(path)
        u8p = ctypes.POINTER(ctypes.c_uint8)
        self._REQ = ctypes.CFUNCTYPE(None, ctypes.POINTER(u8p), ctypes.POINTER(ctypes.c_size_t),
                                     ctypes.POINTER(ctypes.c_size_t))
        self._CMP = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, u8p,
                                     ctypes.c_size_t, ctypes.c_size_t)
        L.cuptiActivityRegisterCallbacks.argtypes = [self._REQ, self._CMP]
        L.cuptiActivityRegisterCallbacks.restype = ctypes.c_int
        for fn in (L.cuptiActivityEnable, L.cuptiActivityDisable):
            fn.argtypes = [ctypes.c_int]
            fn.restype = ctypes.c_int
        L.cuptiActivityFlushAll.argtypes = [ctypes.c_uint32]
        L.cuptiActivityFlushAll.restype = ctypes.c_int
        L.cuptiActivityGetNextRecord.argtypes = [u8p, ctypes.c_size_t,
                                                 ctypes.POINTER(ctypes.c_void_p)]
        L.cuptiActivityGetNextRecord.restype = ctypes.c_int
        L.cuptiActivityGetNumDroppedRecords.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                        ctypes.POINTER(ctypes.c_size_t)]
        L.cuptiActivityGetNumDroppedRecords.restype = ctypes.c_int
        self._lock = threading.Lock()
        self._pool: list = []
        self._live: dict = {}
        self._records: list = []
        self._names: dict = {}
        self.refused = 0
        self.kinds: tuple = ()
        self._req_cb = self._REQ(self._on_request)
        self._cmp_cb = self._CMP(self._on_complete)
        rc = L.cuptiActivityRegisterCallbacks(self._req_cb, self._cmp_cb)
        if rc != 0:
            raise RuntimeError(f"cuptiActivityRegisterCallbacks -> {rc}")

    def _on_request(self, buf_pp, size_p, maxrec_p):
        with self._lock:
            if self._pool:
                b = self._pool.pop()
            elif len(self._live) < self.MAX_BUFFERS:
                b = (ctypes.c_uint64 * ((self.BUF + self.PAD) // 8))()
            else:
                b = None
            if b is None:
                self.refused += 1
                size_p[0] = 0
                maxrec_p[0] = 0
                return
            self._live[ctypes.addressof(b)] = b
        buf_pp[0] = ctypes.cast(b, ctypes.POINTER(ctypes.c_uint8))
        size_p[0] = self.BUF
        maxrec_p[0] = 0

    def _name(self, ptr: int) -> str:
        name = self._names.get(ptr)
        if name is None:
            name = ctypes.string_at(ptr).decode("utf-8", "replace") if ptr else "?"
            self._names[ptr] = name
        return name

    def _on_complete(self, ctx, stream_id, buf, size, valid):
        out = []
        try:
            rec = ctypes.c_void_p()
            while self.lib.cuptiActivityGetNextRecord(buf, valid, ctypes.byref(rec)) == 0:
                addr = rec.value
                kind = ctypes.c_uint32.from_address(addr).value
                if kind == KIND_CONCURRENT_KERNEL:
                    out.append(parse_kernel_record(ctypes.string_at(addr, KERNEL_READ), self._name))
                elif kind == KIND_MEMCPY:
                    out.append(parse_memcpy_record(ctypes.string_at(addr, MEMCPY_READ)))
        except Exception:  # pragma: no cover - never raise into CUPTI
            logger.exception("R1 CUPTI buffer parse failed")
        finally:
            addr = ctypes.cast(buf, ctypes.c_void_p).value
            with self._lock:
                self._records.extend(out)
                b = self._live.pop(addr, None)
                if b is not None:
                    self._pool.append(b)

    def enable(self, kinds) -> None:
        for k in kinds:
            rc = self.lib.cuptiActivityEnable(k)
            if rc != 0:
                raise RuntimeError(f"cuptiActivityEnable({k}) -> {rc}")
        self.kinds = tuple(kinds)

    def disable(self) -> None:
        for k in self.kinds:
            self.lib.cuptiActivityDisable(k)
        self.kinds = ()

    def flush(self) -> None:
        self.lib.cuptiActivityFlushAll(0)

    def take(self) -> list:
        with self._lock:
            out, self._records = self._records, []
        return out

    def dropped(self) -> int:
        n = ctypes.c_size_t(0)
        self.lib.cuptiActivityGetNumDroppedRecords(None, 0, ctypes.byref(n))
        return int(n.value) + self.refused


def cupti_selftest(cu: Cupti, device) -> dict:
    """Validate the record layout on this CUPTI build with one marker, one fill, one D2H."""
    stream = torch.cuda.Stream(device=device)
    x = torch.empty(4096, dtype=torch.float32, device=device)
    host = torch.empty(2048, dtype=torch.float32, pin_memory=True)
    num_devices = torch.cuda.device_count()
    cu.enable((KIND_CONCURRENT_KERNEL, KIND_MEMCPY))
    try:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1000)
            x.fill_(1.0)
            host.copy_(x[:2048], non_blocking=True)
        stream.synchronize()
    finally:
        cu.disable()
    deadline = time.time() + 2.0
    recs = []
    while time.time() < deadline:
        cu.flush()
        recs += cu.take()
        if any(r[5] == "k" and MARKER_NAME in r[2] for r in recs) and any(r[5] == "m" for r in recs):
            break
        time.sleep(0.02)
    spin = [r for r in recs if r[5] == "k" and MARKER_NAME in r[2]]
    fill = [r for r in recs if r[5] == "k" and "FillFunctor" in r[2]]
    mcpy = [r for r in recs if r[5] == "m" and r[6] == 2048 * 4]

    def sane(r):
        return 0 < r[1] - r[0] < 10**9 and r[7] < num_devices

    # CUPTI stream/device ids are not torch's ids: require the marker, the fill and the copy
    # (one private stream) to agree with each other and to be ordered.
    kernel_ok = (
        len(spin) == 1 and len(fill) == 1 and all(sane(r) for r in spin + fill)
        and spin[0][3] == fill[0][3] and spin[0][7] == fill[0][7]
        and spin[0][1] <= fill[0][0]
    )
    memcpy_ok = (
        kernel_ok and len(mcpy) == 1 and 0 < mcpy[0][1] - mcpy[0][0] < 10**9
        and mcpy[0][3] == fill[0][3] and mcpy[0][0] >= fill[0][1]
    )
    return {"kernel_layout_ok": kernel_ok, "memcpy_layout_ok": memcpy_ok,
            "n_records": len(recs), "graph_node_field": "unverified (offset 144)"}


# ---------------------------------------------------------------- tracer
class _Step:
    __slots__ = ("id", "level", "meta", "phases", "events", "t0", "t1", "markers", "seen_end")

    def __init__(self, sid, level, meta):
        self.id = sid
        self.level = level
        self.meta = meta
        self.phases = []
        self.events = []
        self.t0 = time.perf_counter_ns()
        self.t1 = None
        self.markers = []
        self.seen_end = 0


class Tracer:
    STALE_NS = 5_000_000_000

    def __init__(self, rank: int, out_dir: str, device):
        self.rank = rank
        self.device = device
        self.sink = JsonlSink(os.path.join(out_dir, f"trace-rank{rank}-pid{os.getpid()}.jsonl"))
        self.mode = "off"
        self.cupti = None
        self.header = {"type": "header", "version": VERSION, "rank": rank}
        self._step: _Step | None = None
        self._closed = collections.deque()
        self._lock = threading.Lock()
        self._expected = collections.deque()  # (step) per marker, launch order
        self._enabled = False
        self._last_end_event = None
        self._kernels: list = []
        self._markers: list = []
        self._thread = None
        pref = (os.environ.get("DSV41_R1_TRACE_MODE") or "auto").strip().lower()
        self.header["mode_pref"] = pref
        if pref in ("auto", "cupti"):
            self._init_cupti()
        if self.mode == "off" and pref in ("auto", "events"):
            self.mode = "events"
        self.header["mode"] = self.mode
        self.sink.put(dict(self.header))
        logger.info("R1 trace rank %d: mode %s (%s)", rank, self.mode, self.header.get("cupti"))

    def _init_cupti(self) -> None:
        tried = []
        for path in cupti_candidates():
            tried.append(path)
            if not os.path.exists(path):
                continue
            try:
                cu = Cupti(path)
                result = cupti_selftest(cu, self.device)
            except Exception as exc:
                self.header["cupti"] = {"path": path, "error": repr(exc)}
                return
            self.header["cupti"] = dict(result, path=path)
            if result["kernel_layout_ok"]:
                self.cupti = cu
                self.kinds = ((KIND_CONCURRENT_KERNEL, KIND_MEMCPY) if result["memcpy_layout_ok"]
                              else (KIND_CONCURRENT_KERNEL,))
                self.mode = "cupti"
                self._thread = threading.Thread(target=self._bg, name="dsv41-r1-cupti",
                                                daemon=True)
                self._thread.start()
            return
        self.header["cupti"] = {"error": "libcupti not found", "tried": tried[:12]}

    # ---- main thread API
    def step_begin(self, level: int, step: int, req_ids, num_reqs: int, num_tokens: int) -> bool:
        if self._step is not None:
            self._abort_open()
        if self.mode == "off":
            return False
        meta = {"rank": self.rank, "step": step, "mode": self.mode, "level": level,
                "req_ids": list(req_ids)[:16], "num_reqs": num_reqs, "num_tokens": num_tokens}
        if self.mode == "cupti" and not self._enabled:
            try:
                self.cupti.enable(self.kinds)
                self._enabled = True
            except Exception as exc:
                logger.error("R1 trace: CUPTI enable failed (%r); trace off on this rank", exc)
                self.sink.put({"type": "mode_change", "rank": self.rank, "to": "off",
                               "error": repr(exc)})
                self.mode = "off"
                return False
        self._step = _Step(step, level, meta)
        self.mark("prepare")
        return True

    def mark(self, phase: str) -> None:
        s = self._step
        if s is None:
            return
        s.phases.append(phase)
        if self.mode == "cupti":
            with self._lock:
                self._expected.append(s)
            torch.cuda._sleep(0)
        else:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            s.events.append(ev)

    def step_end(self) -> None:
        s = self._step
        if s is None:
            return
        self.mark("end")
        s.t1 = time.perf_counter_ns()
        s.meta["host_wall_us"] = round((s.t1 - s.t0) / 1e3, 3)
        self._step = None
        if self.mode == "cupti":
            ev = torch.cuda.Event()
            ev.record()
            self._last_end_event = ev
        with self._lock:
            self._closed.append(s)

    def _abort_open(self) -> None:
        """A step that never reached step_end (early return): close it as truncated."""
        s = self._step
        s.meta["truncated"] = True
        self.step_end()

    def housekeep(self) -> None:
        """Called at every step start (traced or not). Never blocks."""
        if self.mode == "events":
            while self._closed and self._closed[0].events[-1].query():
                s = self._closed.popleft()
                ev = s.events
                d = [ev[i].elapsed_time(ev[i + 1]) for i in range(len(ev) - 1)]
                self.sink.put(lambda s=s, d=d: summarize_events(s.phases, d, s.meta))
        elif self.mode == "cupti" and self._enabled and self._step is None:
            ev = self._last_end_event
            if ev is not None and ev.query():
                self.cupti.disable()
                self._enabled = False

    # ---- CUPTI background thread (no CUDA runtime calls; CUPTI flush only)
    def _bg(self) -> None:
        while True:
            time.sleep(0.05)
            try:
                with self._lock:
                    busy = self._enabled or bool(self._closed) or bool(self._expected)
                if busy:
                    self.cupti.flush()
                    self._process(self.cupti.take())
            except Exception:  # pragma: no cover
                logger.exception("R1 trace background step failed")

    def _process(self, recs) -> None:
        for r in recs:
            if r[5] == "k" and MARKER_NAME in r[2]:
                self._markers.append(r)
            else:
                self._kernels.append(r)
        self._markers.sort()
        with self._lock:
            while self._markers and self._expected:
                self._expected.popleft().markers.append(self._markers.pop(0))
            ready, stale = [], []
            now = time.perf_counter_ns()
            while self._closed and now - self._closed[0].t1 > self.STALE_NS and (
                len(self._closed[0].markers) < len(self._closed[0].phases)
            ):
                s = self._closed.popleft()  # a marker record was lost: never wait on it
                self._expected = collections.deque(e for e in self._expected if e is not s)
                stale.append(s)
            while self._closed and len(self._closed[0].markers) == len(self._closed[0].phases):
                s = self._closed[0]
                s.seen_end += 1
                if s.seen_end < 2:  # one more flush so lagging side-stream records arrive
                    break
                ready.append(self._closed.popleft())
            idle = not self._enabled and not self._closed and not self._expected
        for s in stale:
            self.sink.put(dict(s.meta, type="trace_lost", markers_seen=len(s.markers),
                               markers_expected=len(s.phases), dropped_records=self.cupti.dropped()))
        for s in ready:
            lo, hi = s.markers[0][0], s.markers[-1][1]
            ks = [k[:7] for k in self._kernels if lo <= k[0] <= hi]
            meta = dict(s.meta, dropped_records=self.cupti.dropped())
            rec = summarize_step(ks, [(m[0], m[1]) for m in s.markers], s.phases, meta, s.level)
            self.sink.put(rec)
            self._kernels = [k for k in self._kernels if k[0] > hi]
        if idle:
            self._kernels.clear()
            self._markers.clear()
        elif len(self._kernels) > 2_000_000:
            del self._kernels[: len(self._kernels) // 2]
