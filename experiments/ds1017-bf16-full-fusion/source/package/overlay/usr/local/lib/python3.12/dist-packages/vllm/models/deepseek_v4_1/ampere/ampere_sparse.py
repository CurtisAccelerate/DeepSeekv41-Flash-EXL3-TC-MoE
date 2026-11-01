# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 sparse MLA attention for SM8x (Ampere: A100 / CMP 170HX).

Same approach as the V4 Ampere path (haosdent/vllm@f8ea5bb): the ROCm attention
class is built on platform-neutral Triton kernels (ragged sparse prefill/decode,
bf16 o_proj einsum, torch compressor insert); only its aiter dispatches are
ROCm-specific and they self-disable on CUDA. fp8 conversions inside the Triton
kernels go through ``vllm.v1.attention.ops.fp8_sm80`` (LUT decode / RNE encode)
because Triton refuses fp8e4nv converts below SM89.
"""

import torch

from vllm.models.deepseek_v4_1.amd.rocm import (
    DeepseekV4ROCMAiterMLASparseBackend,
    DeepseekV41ROCMAiterMLAAttention,
    DeepseekV41ROCMAiterSparseSWABackend,
)
from vllm.platforms.interface import DeviceCapability


class DeepseekV41AmpereMLASparseBackend(DeepseekV4ROCMAiterMLASparseBackend):
    supported_kv_cache_dtypes = ["auto", "fp8_ds_mla", "fp8", "bfloat16"]
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV41"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekV41AmpereSparseSWABackend(DeepseekV41ROCMAiterSparseSWABackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_SPARSE_SWA_DSV41"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekV41AmpereMLAAttention(DeepseekV41ROCMAiterMLAAttention):
    """SM8x DeepSeek V4.1 attention: ROCm Triton path on CUDA Ampere."""

    backend_cls = DeepseekV41AmpereMLASparseBackend
    swa_backend_cls = DeepseekV41AmpereSparseSWABackend

    def _uses_fp8_ds_mla_layout(self):
        from hybrid.abc_kv_contract import requested_dtype
        return requested_dtype() != 'bfloat16'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from hybrid.abc_kv_contract import requested_dtype
        if requested_dtype() != self.kv_cache_dtype:
            raise ValueError('Staged cache dtype and actual attention dtype differ')
        # The ROCm class flips wo_a.is_bmm off so its (aiter/native) MXFP8
        # linear kernel keeps the raw block-fp8 weight for the bf16 einsum
        # (_get_cached_wo_a_bf16). On SM8x the MXFP8 kernel is Marlin, whose
        # repack would destroy that layout, so keep is_bmm on and let the
        # overlay's Marlin exemption skip the repack for bmm layers.
        self.wo_a.is_bmm = True

    def _o_proj(self, o, positions):
        # wo_a went through the MXFP8 *emulation* kernel (the only bmm-capable
        # MXFP8 kernel below SM90), which dequantizes it to bf16 at load. The
        # einsum path multiplies by weight_scale_inv when present, so drop a
        # leftover scale once the weight is already bf16.
        wo_a = self.wo_a
        if not getattr(self, "_dsv41_wo_a_checked", False):
            assert wo_a.weight.element_size() >= 2, (
                "wo_a is still 1-byte MXFP8 on SM8x; expected the emulation "
                "kernel to dequantize it at load"
            )
            # The emulation kernel keeps the (now already applied) E8M0
            # ``weight_scale`` on the layer, and _get_cached_wo_a_bf16 would
            # multiply the bf16 weight by it a second time (output ~2^-11 too
            # small, block-wise distorted). Install the bf16 weight as the
            # cached einsum operand directly so no scale is re-applied.
            wo_a._dsv4_wo_a_bf16 = (
                wo_a.weight.data.view(self.n_local_groups, self.o_lora_rank, -1)
                .to(torch.bfloat16)
                .contiguous()
            )
            self._dsv41_wo_a_checked = True
        from hybrid.abc_kernels import woa  # STAR_ABC integration
        candidate = woa(self, o, positions)
        if candidate is not None:
            return candidate
        return super()._o_proj(o, positions)

    def prepare_attn_preshuffle(self) -> None:
        # aiter-only weight preshuffle; keep the plain MXFP8/Marlin linears.
        self._wqa_wkv_scale = None
        self._wo_b_scale = None


# ---------------------------------------------------------------------------
# Stage-level dumps (DSV41_DEBUG_DUMP=<dir>, DSV41_DEBUG_DUMP_LAYERS=0,1,2):
# capture q/kv before and after the fused RoPE insert, the gathered K rows,
# the sparse-prefill inputs/outputs and the o_proj input/output for small
# real batches (1 < T <= 64) so ktests/ref_stages.py can compare each stage
# against the torch reference.
# ---------------------------------------------------------------------------
import os as _os

import torch as _torch


def _dsv41_dump_dir() -> str:
    return _os.environ.get("DSV41_DEBUG_DUMP", "")


def _dsv41_dump_layers() -> set[int]:
    s = _os.environ.get("DSV41_DEBUG_DUMP_LAYERS", "0,1,2")
    return {int(v) for v in s.split(",") if v.strip()}


if _dsv41_dump_dir():
    from vllm.distributed import get_pp_group as _get_pp_group
    from vllm.forward_context import get_forward_context as _get_fc
    from vllm.models.deepseek_v4_1.amd import rocm as _rocm_mod

    _STAGE: dict = {}

    def _stage_active(self, num_tokens: int) -> bool:
        if self.layer_id not in _dsv41_dump_layers() or not (1 < num_tokens <= 64):
            return False
        fc = _get_fc()
        if not isinstance(fc.attn_metadata, dict):
            return False
        f = _stage_file(self)
        return not _os.path.exists(f)

    def _stage_file(self) -> str:
        rank = _get_pp_group().rank_in_group
        return f"{_dsv41_dump_dir()}/L{self.layer_id:02d}_r{rank}_stages.pt"

    _orig_insert = DeepseekV41AmpereMLAAttention._fused_qnorm_rope_kv_insert

    def _insert_dump(self, q, kv, positions, attn_metadata):
        active = _stage_active(self, q.shape[0])
        if active:
            _STAGE.clear()
            _STAGE["layer_id"] = self.layer_id
            _STAGE["q_pre"] = q.detach().clone().cpu()
            _STAGE["kv_pre"] = kv.detach().clone().cpu()
            _STAGE["positions"] = positions.detach().clone().cpu()
        out = _orig_insert(self, q, kv, positions, attn_metadata)
        if active:
            _STAGE["q_rope"] = out.detach().clone().cpu()
        return out

    DeepseekV41AmpereMLAAttention._fused_qnorm_rope_kv_insert = _insert_dump

    _orig_prefill = _rocm_mod.rocm_sparse_attn_prefill

    def _prefill_dump(q, kv, indices, topk_length, scale, head_dim, nope_head_dim,
                      rope_head_dim, attn_sink, output, **kw):
        _orig_prefill(q, kv, indices, topk_length, scale, head_dim, nope_head_dim,
                      rope_head_dim, attn_sink, output, **kw)
        if _STAGE.get("q_rope") is not None and "attn_o" not in _STAGE:
            idx = indices.reshape(indices.shape[0], -1)
            _STAGE["prefill_q"] = q.detach().clone().cpu()
            _STAGE["prefill_indices"] = idx.detach().clone().cpu()
            _STAGE["prefill_lens"] = None if topk_length is None else topk_length.detach().clone().cpu()
            used = idx[idx >= 0].unique()
            kv2 = kv.reshape(-1, kv.shape[-1])
            _STAGE["prefill_kv_rows"] = used.cpu()
            _STAGE["prefill_kv"] = kv2[used].detach().clone().cpu()
            _STAGE["scale"] = float(scale)
            _STAGE["attn_sink"] = None if attn_sink is None else attn_sink.detach().clone().cpu()
            _STAGE["attn_o"] = output.detach().clone().cpu()

    _rocm_mod.rocm_sparse_attn_prefill = _prefill_dump

    _orig_o_proj = DeepseekV41AmpereMLAAttention._o_proj

    def _o_proj_dump(self, o, positions):
        res = _orig_o_proj(self, o, positions)
        if _STAGE.get("layer_id") == self.layer_id and "attn_o" in _STAGE and "o_proj_out" not in _STAGE:
            _STAGE["o_proj_in"] = o.detach().clone().cpu()
            _STAGE["o_proj_out"] = res.detach().clone().cpu()
            _STAGE["wo_a_dtype"] = str(self.wo_a.weight.dtype)
            f = _stage_file(self)
            _os.makedirs(_os.path.dirname(f), exist_ok=True)
            _torch.save(dict(_STAGE), f)
            _STAGE.clear()
        return res

    DeepseekV41AmpereMLAAttention._o_proj = _o_proj_dump
