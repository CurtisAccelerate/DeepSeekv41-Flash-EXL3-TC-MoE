"""R2 routed-expert kernel choice per B: coop (image extension), stock (exl3_moe) or native.

Only consulted inside an R2 private capture (the R2 runtime sets ACTIVE). Outside it,
exl3.py keeps its boot-time CoopCutoff (default: the R1 condition tokens*topk <= 256).

native = the W25 native-mixed kernels specialised per row count (BSZ template; MAX_SLOTS
512 and a 9-bit slot key so 48 rows x 6 = 288 slots fit). Launched directly at capture
time from a pinned R2 cubin; no node swap. The BSZ=6 kernels A/B are byte-identical to
the R1 native kernels (kernel/compile-report.json).
"""
import ctypes as C
import hashlib
import os
from pathlib import Path

ACTIVE = None          # R2 capture receipt context: object with .plan and .experts dict
CUBIN_ENV = "DSV41_R2_NATIVE_CUBIN"
CUBIN_SHA_ENV = "DSV41_R2_NATIVE_CUBIN_SHA256"
ROWS = (12, 18, 24, 30, 36, 42, 48)
ROT = "_ZN16exl3_moe_coop_ns24exl3_moe_coop_rot_kernelE13MoeCoopParams"


def symbol(stage, k, rows):
    return f"_ZN16exl3_moe_coop_ns22exl3_moe_coop_{stage}_kernelILi{k}ELi2ELb1ELb0ELi{rows}EEEv13MoeCoopParams"


def mode(tokens, topk):
    """Kernel kind the R2 plan requests for this MoE call, or None outside R2 capture."""
    ctx = ACTIVE
    if ctx is None or "experts" not in ctx.plan.components:
        return None
    if tokens != ctx.plan.served_by[0] or topk != 6:
        return "unplanned"
    return ctx.plan.experts


def record(kind):
    ctx = ACTIVE
    if ctx is not None:
        ctx.experts[kind] = ctx.experts.get(kind, 0) + 1


def receipt(counts, plan):
    """Capture receipt: every non-mixed-K MoE call used the planned kind; no misses."""
    want = plan.experts
    misses = {k: v for k, v in counts.items() if k not in (want, "mixedk")}
    return dict(planned=want, counts=dict(counts), misses=misses, ok=counts.get(want, 0) > 0 and not misses)


def eligible_rows(args, rows):
    """W9 runtime.eligible generalised to rows (CPU metadata only)."""
    if len(args) != 36:
        return False
    x, sel, rw, lo, hi, H = args[:6]
    tables = args[6:15]
    bias = args[15:18]
    Kg, Ku, Kd, mcg, mul1, act, limit, gated = args[18:26]
    hadg, hadu, gug, guu, actout, dout, ctr, out = args[26:34]
    if any(v is not None for v in (*bias, *args[34:36])):
        return False
    if not (lo == 0 and hi == 384 and H == 5120 and Kg == Ku == Kd and Kg in (2, 3)
            and not mcg and mul1 and act == 0 and limit == 10.0 and gated):
        return False
    if os.environ.get("EXL3_MOE_COOP_KSPLIT", "1") not in ("", "0", "1"):
        return False
    if os.environ.get("EXL3_MOE_COOP_WIDE", "1") not in ("", "-1", "1"):
        return False
    if os.environ.get("EXL3_MOE_COOP_DBG", "0") not in ("", "0"):
        return False
    tensors = (x, sel, rw, *tables, hadg, hadu, gug, guu, actout, dout, ctr, out)
    if not all(t.is_cuda and t.is_contiguous() and t.device == x.device for t in tensors):
        return False
    dt = lambda t, name: str(t.dtype) == "torch." + name
    if not all(dt(t, "float16") for t in (x, rw, hadg, hadu, gug, guu, actout)):
        return False
    if not all(dt(t, "int64") for t in (sel, *tables)) or not dt(ctr, "int32"):
        return False
    if not all(dt(t, "float32") for t in (dout, out)):
        return False
    if tuple(x.shape) != (rows, 5120) or x.stride(0) != 5120:
        return False
    if tuple(sel.shape) != (rows, 6) or tuple(rw.shape) != (rows, 6) or tuple(out.shape) != (rows, 5120):
        return False
    if any(tuple(t.shape) != (384,) for t in tables):
        return False
    I = int(guu.shape[-1])
    if I not in (512, 640):
        return False
    if any(t.shape[-1] != 5120 for t in (hadg, hadu, dout)) or any(t.shape[-1] != I for t in (gug, guu, actout)):
        return False
    s = rows * 6
    slots = min(t.numel() // w for t, w in ((hadg, 5120), (hadu, 5120), (gug, I), (guu, I), (actout, I), (dout, 5120)))
    return slots == s and ctr.numel() >= s * (I // 128) + rows * 40 + 2 * s + 3


def grids(rows, I):
    """(rot, a, b) grids; rows=6 reproduces the R1 W9 launch (180, 72*(I/128), 1440)."""
    s = rows * 6
    rot = (s * (5120 // 128) * 2 + 15) // 16
    return rot, s * 2 * (I // 128), s * 40


class NativeRows:
    """Pinned R2 cubin in the worker's current context; modules retained for its lifetime."""

    def __init__(self, path, pin, device):
        import torch
        from w9_exl_ab.params import Params
        self.torch = torch
        self.Params = Params
        self.device = device
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != pin:
            raise RuntimeError("R2 native cubin identity mismatch")
        self.lib = C.CDLL("libcuda.so.1")
        cur = C.c_void_p()
        self.call("cuCtxGetCurrent", C.byref(cur))
        if not cur.value:
            raise RuntimeError("R2 native requires the worker's current context")
        self.context = cur.value
        self.module = C.c_void_p()
        self.call("cuModuleLoad", C.byref(self.module), C.c_char_p(os.fsencode(str(path))))
        self.fn = {}
        names = {"rot": ROT}
        for rows in ROWS:
            for k in (2, 3):
                for stage in ("a", "b"):
                    names[(stage, k, rows)] = symbol(stage, k, rows)
        for key, sym in names.items():
            f = C.c_void_p()
            self.call("cuModuleGetFunction", C.byref(f), self.module, C.c_char_p(sym.encode()))
            self.fn[key] = f
        self.launches = {}

    def call(self, name, *args):
        f = getattr(self.lib, name)
        f.restype = C.c_int
        r = f(*args)
        if r:
            raise RuntimeError(f"{name}: CUDA {r}")

    def launch(self, args, rows):
        cur = C.c_void_p()
        self.call("cuCtxGetCurrent", C.byref(cur))
        if cur.value != self.context:
            raise RuntimeError("R2 native CUDA context changed")
        p = self.Params()
        x, sel, rw, lo, hi, H = args[:6]
        p.x = x.data_ptr(); p.sel = sel.data_ptr(); p.rw = rw.data_ptr(); p.x_stride = x.stride(0)
        p.bsz = rows; p.topk = 6; p.H = p.Hi = p.Ho = p.H_out = 5120
        p.min_expert = lo; p.max_expert = hi; p.n_local = 384
        for name, t in zip(("g_trellis", "g_suh", "g_svh", "u_trellis", "u_suh", "u_svh",
                            "d_trellis", "d_suh", "d_svh"), args[6:15]):
            setattr(p, name, t.data_ptr())
        p.act = 0; p.act_limit = 10; p.gated = True; p.a_global = True; p.gu_f32 = False
        p.ksplit_a = p.ksplit_b = 1; p.dbg = 0
        for name, t in zip(("had_g", "had_u", "gu_g", "gu_u", "act_out", "d_out"), args[26:32]):
            setattr(p, name, t.data_ptr())
        s = rows * 6
        p.I = int(args[29].shape[-1]); p.slots_max = s; p.rows_max = rows
        p.ctr_a_len = s * (p.I // 128); p.ctr_b_len = rows * 40
        p.ctr_a = args[32].data_ptr(); p.ctr_b = p.ctr_a + p.ctr_a_len * 4; p.runs = p.ctr_b + p.ctr_b_len * 4
        p.out = args[33].data_ptr(); p.out_stride = args[33].stride(-2)
        stream = C.c_void_p(self.torch.cuda.current_stream(self.device).cuda_stream)
        k = int(args[18])
        stage = 8192 if k == 3 else 0
        g_rot, g_a, g_b = grids(rows, p.I)
        params = (C.c_void_p * 1)(C.addressof(p))
        for key, grid, smem in (("rot", g_rot, 0), (("a", k, rows), g_a, 10240 + 16384 + stage),
                                (("b", k, rows), g_b, 16384 + stage + 8192)):
            self.call("cuLaunchKernel", self.fn[key], *[C.c_uint(v) for v in (grid, 1, 1, 512, 1, 1, smem)],
                      stream, params, C.c_void_p())
        self.launches[rows] = self.launches.get(rows, 0) + 1


BACKEND = None


def load(device):
    """Load once at R2 startup when any plan selects native; raises on identity mismatch."""
    global BACKEND
    if BACKEND is None:
        BACKEND = NativeRows(os.environ[CUBIN_ENV], os.environ[CUBIN_SHA_ENV], device)
    return BACKEND


def dispatch_native(args, rows):
    """exl3.py hook for kind == native. False (caller takes the stock kernel and the capture
    receipt records the miss) when the call is not native-eligible."""
    if BACKEND is not None and eligible_rows(args, rows):
        BACKEND.launch(args, rows)
        record("native")
        return True
    record("native_ineligible")
    return False
