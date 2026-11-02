"""Extra narrow rows only. Staged, not installed or GPU-qualified.
All old rows continue through the original fold dispatcher and original libraries.
"""
import ctypes,hashlib,os
from pathlib import Path
import torch
from w9_exl_ab.fold_serve import GuLaunch
EXTRA_ROWS=(4,16,20,28,32)
LIB_DIR=Path('/w9/integration/fold-narrow')
PINS={2:('libgu_narrow_k2.so','4ec2e535c524f10710145cdf8f95e7f2d3d2a98e8af71aa9ed80cae29c4e63d4'),3:('libgu_narrow_k3.so','956508b54ca3b4f11165c7cf218026e2cc380f583057f5a98a510562957cb720')}
_LIBS={};_SEEN=set();READY=False

def _extra_lib(k):
 if k not in _LIBS:
  if torch.cuda.is_current_stream_capturing():raise RuntimeError('Narrow library must be loaded and qualified before capture')
  name,pin=PINS[k];p=LIB_DIR/name
  if hashlib.sha256(p.read_bytes()).hexdigest()!=pin:raise RuntimeError('Extra narrow library pin changed')
  lib=ctypes.CDLL(str(p));lib.gu_launch.argtypes=[ctypes.POINTER(GuLaunch),ctypes.c_void_p];lib.gu_launch.restype=None;_LIBS[k]=lib
 return _LIBS[k]

def _extra_important(args):
    from hybrid.abc_runtime import context as abc_context  # STAR_ABC integration
    ctx=abc_context()
    if ctx is not None and ctx.plan.b==1 and ctx.plan.k==5 and ctx.plan.experts!="tc":
        return None  # preserve the exact old B1 mixed executable via private-node update
    if os.environ.get('EXL3_MOE_COOP_KSPLIT','1') not in ('','0','1') or os.environ.get('EXL3_MOE_COOP_WIDE','1') not in ('','-1','1') or os.environ.get('EXL3_MOE_COOP_DBG','0') not in ('','0'):
        return None
    if len(args) != 36:
        return None
    x, sel, rw, lo, hi, H = args[:6]
    tables = args[6:15]
    bias = args[15:18]
    kg, ku, kd, mcg, mul1, act, limit, gated = args[18:26]
    had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out = args[26:34]
    shared = args[34:36]
    rows = int(x.shape[0]) if getattr(x, "shape", None) else 0
    if rows not in EXTRA_ROWS:
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

def _launch_checked(args):
    """Launch the rebuilt kernel. False means the caller uses the loaded one."""
    found = _extra_important(args)
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
    _extra_lib(k).gu_launch(ctypes.byref(launch), stream)
    seen = (k, rows)
    if seen not in _SEEN:
        _SEEN.add(seen)
        print(f"extra-row-fold launch k={k} rows={rows} width={width}", flush=True)
    return True

def fixture_launch(args):
 if torch.cuda.is_current_stream_capturing():raise RuntimeError('Fixture launch cannot run in a serving capture')
 if _extra_important(args) is None:raise RuntimeError('Actual-pack narrow fixture metadata rejected')
 return _launch_checked(args)

def try_launch(args):
 # Returning False leaves the EXACT original dispatch call in place.
 from hybrid.abc_runtime import context
 ctx=context()
 if ctx is None or ctx.mode not in ('warm','capture') or ctx.plan.k!=3 or not args or int(args[0].shape[0]) not in EXTRA_ROWS:return False
 rows=int(args[0].shape[0])
 if rows!=4*ctx.plan.b or not READY:raise RuntimeError('Unqualified narrow fold shape requested')
 if _extra_important(args) is None:raise RuntimeError('Qualified narrow capture changed metadata')
 result=_launch_checked(args)
 if not result:raise RuntimeError('New row fell back instead of using qualified fold kernel')
 ctx.paths['experts_extra_fold']=ctx.paths.get('experts_extra_fold',0)+1
 return True
