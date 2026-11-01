# dsv41: wo_a (grouped attention output projection) as an FP8 W8A16 Triton GEMM for small token counts on sm80.
#
# The Ampere path dequantizes the MXFP8 wo_a to bf16 at load and runs einsum("tgd,grd->tgr"): a batched cuBLAS
# GEMM over 16.8 MB of bf16 per rank, written in [g, t, r] order and then copied to [t, g*r] for wo_b. Here the
# FP8 codes are recovered exactly from the bf16 weight (power-of-two 32x32 block scales: w = q * 2^e, verified per
# layer), decoded to bf16 in registers with Marlin's packed bit trick, multiplied on bf16 tensor cores with fp32
# accumulation, scaled per 32x32 block, and written straight into the contiguous [t, g*r] layout wo_b reads.
import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3_to_bf16(w_u8):
    return tl.inline_asm_elementwise(
        asm="""{
        .reg .b32 hi, lo, t, bias, nz;
        mov.b32 bias, 0x7B807B80;
        mov.b32 nz, 0x80008000;
        and.b32 t, $2, 0x80008000;
        and.b32 hi, $2, 0x7F007F00;
        shr.b32 hi, hi, 4;
        or.b32 hi, hi, t;
        shl.b32 lo, $2, 8;
        and.b32 t, lo, 0x80008000;
        and.b32 lo, lo, 0x7F007F00;
        shr.b32 lo, lo, 4;
        or.b32 lo, lo, t;
        prmt.b32 $0, lo, hi, 0x5410;
        prmt.b32 $1, lo, hi, 0x7632;
        fma.rn.bf16x2 $0, $0, bias, nz;
        fma.rn.bf16x2 $1, $1, bias, nz;
        }""",
        constraints="=r,=r,r",
        args=[w_u8],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=4,
    )


@triton.jit(do_not_specialize=["M"])
def _wo_a_kernel(x_ptr, w_ptr, s_ptr, o_ptr, p_ptr, M, G: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
                 KSPLIT_LEN: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr,
                 BK: tl.constexpr):
    # x: [M, G, D] bf16; w: [G * R, D] e4m3 bytes; s: [G, R / 32, D / 32] fp32; out: [M, G * R]
    pid_n = tl.program_id(0)          # over G * R / BN
    pid_k = tl.program_id(1)
    nb = R // BN
    g = pid_n // nb
    offs_n = (pid_n % nb) * BN + tl.arange(0, BN)      # within the group
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    mmask = offs_m < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k_lo = pid_k * KSPLIT_LEN
    nb32 = (pid_n % nb) * (BN // 32) + tl.arange(0, BN // 32)
    for k0 in range(k_lo, k_lo + KSPLIT_LEN, BK):
        k = k0 + offs_k
        xv = tl.load(x_ptr + offs_m[:, None] * (G * D) + g * D + k[None, :], mask=mmask[:, None], other=0.0)
        wu = tl.load(w_ptr + (g * R + offs_n[None, :]) * D + k[:, None])
        # power-of-two block scales: w * 2^e is exact in bf16, so scale the tile before one BK-deep dot
        kb = k0 // 32 + tl.arange(0, BK // 32)
        sc = tl.load(s_ptr + (g * (R // 32) + nb32[None, :]) * (D // 32) + kb[:, None])        # [BK/32, BN/32]
        sc = tl.reshape(tl.broadcast_to(sc[:, None, :, None], (BK // 32, 32, BN // 32, 32)), (BK, BN))
        acc = tl.dot(xv, _e4m3_to_bf16(wu) * sc.to(tl.bfloat16), acc)
    if SPLIT == 1:
        tl.store(o_ptr + offs_m[:, None] * (G * R) + g * R + offs_n[None, :], acc.to(o_ptr.dtype.element_ty),
                 mask=mmask[:, None])
    else:
        tl.store(p_ptr + pid_k * M * (G * R) + offs_m[:, None] * (G * R) + g * R + offs_n[None, :], acc,
                 mask=mmask[:, None])


@triton.jit(do_not_specialize=["MN"])
def _reduce(p_ptr, o_ptr, MN, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(p_ptr + s * MN + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, acc.to(o_ptr.dtype.element_ty), mask=mask)


MAX_M = 16
_CFG = {"BN": 64, "SPLIT": 2, "warps": 4, "stages": 3, "BK": 256}


def recover_fp8(w_bf16: torch.Tensor):
    """[G, R, D] bf16 -> (e4m3 bytes [G*R, D], fp32 scales [G, R/32, D/32]) if w is exactly q * 2^e per 32x32
    block, else None."""
    G, R, D = w_bf16.shape
    if R % 32 or D % 32:
        return None
    wb = w_bf16.float().view(G, R // 32, 32, D // 32, 32)
    amax = wb.abs().amax(dim=(2, 4)).clamp_min(2.0 ** -126)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (wb / s[:, :, None, :, None]).to(torch.float8_e4m3fn)
    back = (q.float() * s[:, :, None, :, None]).to(torch.bfloat16).view(G, R, D)
    if not torch.equal(back, w_bf16):
        return None
    return q.view(G * R, D).contiguous().view(torch.uint8), s.contiguous()


def wo_a(x: torch.Tensor, w_u8: torch.Tensor, scale: torch.Tensor, G: int, R: int, D: int) -> torch.Tensor:
    """x [M, G*D] bf16 (contiguous) -> [M, G*R] bf16."""
    M = x.shape[0]
    BN, SPLIT = _CFG["BN"], _CFG["SPLIT"]
    BK = _CFG["BK"]
    while D % (SPLIT * BK):
        SPLIT //= 2
    out = torch.empty(M, G * R, dtype=x.dtype, device=x.device)
    part = torch.empty(SPLIT, M, G * R, dtype=torch.float32, device=x.device) if SPLIT > 1 else out
    _wo_a_kernel[(G * R // BN, SPLIT)](x, w_u8, scale, out, part, M, G, R, D, D // SPLIT, BM=16, BN=BN, SPLIT=SPLIT, BK=BK,
                                       num_warps=_CFG["warps"], num_stages=_CFG["stages"])
    if SPLIT > 1:
        _reduce[(triton.cdiv(M * G * R, 1024),)](part, out, M * G * R, SPLIT=SPLIT, BLOCK=1024)
    return out


@triton.jit
def _dequant_kernel(w_ptr, s_ptr, o_ptr, R: tl.constexpr, D: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr):
    # o[g * R + r, d] = e4m3(w[g * R + r, d]) * s[g, r / 32, d / 32]; power-of-two scales, so exact in bf16
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    w = tl.load(w_ptr + rows[:, None] * D + cols[None, :])
    srow = (rows // R) * (R // 32) + (rows % R) // 32
    s = tl.load(s_ptr + srow[:, None] * (D // 32) + (cols // 32)[None, :])
    tl.store(o_ptr + rows[:, None] * D + cols[None, :], _e4m3_to_bf16(w) * s.to(tl.bfloat16))


def dequant(w_u8: torch.Tensor, scale: torch.Tensor, out: torch.Tensor, G: int, R: int, D: int) -> torch.Tensor:
    """dsv41 (ds41-patches/patch_wo_a_free.py): rebuild the exact bf16 wo_a [G, R, D] from its FP8 codes into
    `out` (>= G*R*D bf16 elements)."""
    o = out[: G * R * D].view(G * R, D)
    _dequant_kernel[(G * R // 32, D // 128)](w_u8, scale, o, R, D, BR=32, BD=128, num_warps=4)
    return o.view(G, R, D)


# dsv41 (ds41-patches/patch_wo_a_cuda.py): CUDA decode wo_a with the inverse RoPE fused into the activation load
import os as _os

from vllm.utils.torch_utils import direct_register_custom_op as _reg

_WO_A_CUDA = _os.environ.get("DSV41_WO_A_CUDA", "1") != "0"
# validation only: DSV41_WO_A_ALL=1 also runs M > 8 here, 8 rows at a time, so prompt scoring (dqeval) checks it
_WO_A_ALL = _os.environ.get("DSV41_WO_A_ALL", "0") == "1"
_WO_A_CFG = {(2, 1024, 4096): (8, 1, 3)}   # (G, R, D) -> (warps per tile, tiles per block, depth), M=6 on CMP 170HX


def cuda_small_ok(o: torch.Tensor) -> bool:
    if not _WO_A_CUDA or (o.shape[0] > 8 and not _WO_A_ALL) or o.shape[0] < 1:
        return False
    try:
        import dsv41_woa_ext  # noqa: F401
    except ImportError:
        return False
    return True


def _wo_a_small(o: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor, w8: torch.Tensor, s: torch.Tensor,
                G: int, R: int, rope_dim: int) -> torch.Tensor:
    import dsv41_woa_ext as _ext

    M, H, HEAD = o.shape
    D = H * HEAD // G
    x = o.reshape(M, -1)
    if not x.is_contiguous() or x.data_ptr() % 16:
        x = x.contiguous()
    pos = positions if positions.dtype == torch.int64 else positions.to(torch.int64)
    out = torch.empty(M, G * R, dtype=torch.bfloat16, device=o.device)
    wpt, tpb, depth = _WO_A_CFG.get((G, R, D), (8, 1, 2))
    for r in range(0, M, 8):
        _ext.wo_a_small(w8, s, x[r:r + 8], pos[r:r + 8], cos_sin, out[r:r + 8], G, R, D, HEAD, HEAD - rope_dim, wpt, tpb, depth)
    return out


def _wo_a_small_fake(o, positions, cos_sin, w8, s, G, R, rope_dim):
    return o.new_empty((o.shape[0], G * R))


_reg("dsv41_wo_a_small", _wo_a_small, fake_impl=_wo_a_small_fake)
