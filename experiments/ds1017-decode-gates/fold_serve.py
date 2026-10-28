"""Launch the rebuilt coop kernels for the row counts that clear the speed bar.

One request is 6 rows. These counts are the ones measured faster by at least
about 0.05 ms per K3 layer, bit-identical to the loaded server. Every other
shape returns False and the caller keeps the existing kernel.
"""
import ctypes
import hashlib
from pathlib import Path

import torch

IMPORTANT_ROWS = (6, 8, 12, 18, 24, 30, 36, 42)
LIB_DIR = Path("/w9/integration/fold")
PINS = {
    3: ("libgu_rows.so", "bc4ee433b0204642ef48d86ed4e8dc79ce690e8497232cc5f95ec2cfd5a916b6"),
    2: ("libgu_k2.so", "d21632dbb376f93b8291ee9a98e28e4cceb29ecdb01667b25f285076fe2f39c5"),
}
_LIBS = {}
_SEEN = set()


class GuLaunch(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_void_p), ("sel", ctypes.c_void_p), ("rw", ctypes.c_void_p),
        ("g_trellis", ctypes.c_void_p), ("g_suh", ctypes.c_void_p), ("g_svh", ctypes.c_void_p),
        ("u_trellis", ctypes.c_void_p), ("u_suh", ctypes.c_void_p), ("u_svh", ctypes.c_void_p),
        ("d_trellis", ctypes.c_void_p), ("d_suh", ctypes.c_void_p), ("d_svh", ctypes.c_void_p),
        ("had_g", ctypes.c_void_p), ("had_u", ctypes.c_void_p),
        ("gu_g", ctypes.c_void_p), ("gu_u", ctypes.c_void_p),
        ("act_out", ctypes.c_void_p), ("d_out", ctypes.c_void_p),
        ("ctr", ctypes.c_void_p), ("out", ctypes.c_void_p),
        ("x_stride", ctypes.c_int64), ("out_stride", ctypes.c_int64),
        ("bsz", ctypes.c_int64), ("topk", ctypes.c_int64),
        ("H", ctypes.c_int64), ("Hi", ctypes.c_int64), ("I", ctypes.c_int64),
        ("Ho", ctypes.c_int64), ("H_out", ctypes.c_int64),
        ("min_expert", ctypes.c_int64), ("max_expert", ctypes.c_int64),
        ("act", ctypes.c_int64), ("gated", ctypes.c_int64), ("gu_f32", ctypes.c_int64),
        ("slots_max", ctypes.c_int64), ("rows_max", ctypes.c_int64),
        ("n_local", ctypes.c_int64), ("act_limit", ctypes.c_double),
    ]


def _lib(k):
    loaded = _LIBS.get(k)
    if loaded is not None:
        return loaded
    name, pin = PINS[k]
    path = LIB_DIR / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != pin:
        raise RuntimeError(f"fold library {name} hash {digest} != {pin}")
    lib = ctypes.CDLL(str(path))
    lib.gu_launch.argtypes = [ctypes.POINTER(GuLaunch), ctypes.c_void_p]
    lib.gu_launch.restype = None
    _LIBS[k] = lib
    return lib


def _important(args):
    if len(args) != 36:
        return None
    x, sel, rw, lo, hi, H = args[:6]
    tables = args[6:15]
    bias = args[15:18]
    kg, ku, kd, mcg, mul1, act, limit, gated = args[18:26]
    had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out = args[26:34]
    shared = args[34:36]
    rows = int(x.shape[0]) if getattr(x, "shape", None) else 0
    if rows not in IMPORTANT_ROWS:
        return None
    if any(v is not None for v in (*bias, *shared)):
        return None
    if not (lo == 0 and hi == 384 and H == 5120 and kg == ku == kd and kg in (2, 3)
            and not mcg and mul1 and act == 0 and float(limit) == 10.0 and gated):
        return None
    tensors = (x, sel, rw, *tables, had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out)
    if not all(t.is_cuda and t.is_contiguous() and t.device == x.device for t in tensors):
        return None
    if x.dtype != torch.float16 or sel.dtype != torch.int64 or rw.dtype != torch.float16:
        return None
    if ctr.dtype != torch.int32 or out.dtype != torch.float32 or d_out.dtype != torch.float32:
        return None
    if tuple(sel.shape) != (rows, 6) or tuple(out.shape) != (rows, 5120) or x.shape[1] != 5120:
        return None
    if any(tuple(t.shape) != (384,) for t in tables):
        return None
    width = int(gu_u.shape[-1])
    if width not in (512, 640):
        return None
    slots = rows * 6
    # The plugin sizes the gate/up scratch for split-k, four times the live slots.
    # The installed launcher uses the smaller hadamard buffer, which is `slots`.
    live_slots = min(
        had_g.numel() // 5120, had_u.numel() // 5120,
        gu_g.numel() // width, gu_u.numel() // width,
        act_out.numel() // width, d_out.numel() // 5120,
    )
    if live_slots != slots:
        return None
    if ctr.numel() < slots * (width // 128) + rows * (5120 // 128) + 2 + (slots + 1) + slots:
        return None
    return kg, rows, width, slots


def try_launch(args):
    """Launch the rebuilt kernel. False means the caller uses the loaded one."""
    found = _important(args)
    if found is None:
        return False
    k, rows, width, slots = found
    x, sel, rw, lo, hi, H = args[:6]
    tables = args[6:15]
    had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out = args[26:34]
    launch = GuLaunch()
    launch.x = x.data_ptr()
    launch.sel = sel.data_ptr()
    launch.rw = rw.data_ptr()
    launch.g_trellis, launch.g_suh, launch.g_svh = (t.data_ptr() for t in tables[0:3])
    launch.u_trellis, launch.u_suh, launch.u_svh = (t.data_ptr() for t in tables[3:6])
    launch.d_trellis, launch.d_suh, launch.d_svh = (t.data_ptr() for t in tables[6:9])
    launch.had_g = had_g.data_ptr()
    launch.had_u = had_u.data_ptr()
    launch.gu_g = gu_g.data_ptr()
    launch.gu_u = gu_u.data_ptr()
    launch.act_out = act_out.data_ptr()
    launch.d_out = d_out.data_ptr()
    launch.ctr = ctr.data_ptr()
    launch.out = out.data_ptr()
    launch.x_stride = x.stride(0)
    launch.out_stride = out.stride(-2)
    launch.bsz = rows
    launch.topk = 6
    launch.H = launch.Hi = launch.Ho = launch.H_out = 5120
    launch.I = width
    launch.min_expert = lo
    launch.max_expert = hi
    launch.act = 0
    launch.gated = 1
    launch.gu_f32 = 0
    launch.slots_max = slots
    launch.rows_max = rows
    launch.n_local = 384
    launch.act_limit = 10.0
    stream = ctypes.c_void_p(torch.cuda.current_stream(x.device).cuda_stream)
    _lib(k).gu_launch(ctypes.byref(launch), stream)
    seen = (k, rows)
    if seen not in _SEEN:
        _SEEN.add(seen)
        print(f"fold-serve launch k={k} rows={rows} width={width}", flush=True)
    return True
