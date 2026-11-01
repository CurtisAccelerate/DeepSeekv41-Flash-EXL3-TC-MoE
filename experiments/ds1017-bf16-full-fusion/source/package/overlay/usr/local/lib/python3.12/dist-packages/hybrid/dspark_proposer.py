"""DSpark (DeepSeek-V4.1 parallel drafter) on the V1 model runner, under pipeline parallel.

The image ships DSpark only for the V2 runner (``v1/worker/gpu/spec_decode/dspark``), while
Engram lookback needs the V1 runner. This proposer ports the V2 speculator's DSpark-specific
pieces onto the V1 ``DFlashProposer`` machinery:

* anchor-as-first-prediction layout: each request contributes N = num_speculative_tokens
  query tokens (bonus/anchor + N-1 noise tokens) and every query position is sampled
  (``sample_from_anchor``), where DFlash uses 1 + N queries and samples the N mask slots;
* sequential Markov sampling: base logits from the draft head at all N positions, then a
  left-to-right pass adding a prefix-dependent bias from the previously drafted token
  (greedy; rejection sampling keeps the target distribution exact either way);
* draft KV layers spread over several kv-cache groups: the hybrid KV manager puts each of
  the draft's three sliding-window caches in its own group here, so context/query slot
  mappings and attention metadata are built per group (V1's base assumes one group).

Under PP the draft lives on the last rank next to the lm_head and the target layers it taps
(``dspark_target_layer_ids``); the token embedding is a PPMissingLayer there, so it is read
from the checkpoint instead of aliased.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import replace as _dc_replace

import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode import llm_base_proposer as _base
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import copy_and_expand_dflash_inputs_kernel, next_power_of_2

logger = init_logger("vllm.dsv41.dspark")  # under the vllm logging tree so it prints


class DSparkProposer(DFlashProposer):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        spec = vllm_config.speculative_config
        assert spec is not None and spec.method == "dspark"
        # DFlashProposer.__init__ asserts method == "dflash"; replicate its body.
        SpecDecodeBaseProposer.__init__(
            self,
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
        self._dsv41_runner = runner
        n = self.num_speculative_tokens
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        self.num_query_per_req = n if self.sample_from_anchor else 1 + n
        self.max_query_tokens = self.max_batch_size * self.num_query_per_req
        self.max_padded_query_tokens = max(
            self.max_query_tokens,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )
        self.max_positions = self.max_num_tokens + self.max_padded_query_tokens
        z = lambda n_, dt: torch.zeros(n_, dtype=dt, device=device)  # noqa: E731
        self._context_slot_mapping_buffer = z(self.max_num_tokens, torch.int64)
        self._slot_mapping_buffer = z(self.max_padded_query_tokens, torch.int64)
        self._context_positions_buffer = z(self.max_num_tokens, torch.int64)
        self.positions = z(self.max_padded_query_tokens, torch.int64)
        self.arange = torch.arange(self.max_positions + 1, device=device, dtype=torch.int32)
        self.parallel_drafting_hidden_state_tensor = None
        from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

        self.dflash_causal = not dflash_has_any_non_causal(self.draft_model_config.hf_config)
        self._anchor_idx = (
            torch.arange(self.max_batch_size, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        self._draft_topk: int | None = getattr(
            self.draft_model_config.hf_config, "dspark_draft_topk", None
        )
        # dummy/capture runs feed the draft's context projection (which takes the
        # combined [T, hidden] states, not the base proposer's wider buffer)
        self._ctx_dummy = torch.zeros(
            self.max_num_tokens,
            self.draft_model_config.hf_config.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        self._dbg = os.environ.get("DSV41_DEBUG_DSPARK") == "1"
        self._t_acc: dict[str, float] = {}
        self._t_n = 0
        self._conf_default = float(os.environ.get("DSV41_DSPARK_CONF", "0") or 0)
        # reload R1 (hybrid.dsv41_r1): the runner sets a plan per propose only when armed
        # and some request carries an R1 flag; None keeps every path below unchanged
        self._dsv41_r1_plan = None
        self._dsv41_r1_glog = None
        self._dsv41_r1_gen: torch.Generator | None = None
        self._dsv41_r1_path: str | None = None  # planned-path sampler, for the round log
        # per kv-cache group buffers (filled in initialize_attn_backend)
        self._draft_layer_gid: dict[str, int] = {}
        self._draft_gids: list[int] = []
        self._gid_block_size: dict[int, int] = {}
        self._ctx_slot: dict[int, torch.Tensor] = {}
        self._q_slot: dict[int, torch.Tensor] = {}
        self._per_gid_cad: dict[int, CommonAttentionMetadata] = {}
        logger.info(
            "DSparkProposer (V1 runner): N=%d queries/request (sample_from_anchor=%s), "
            "causal=%s, draft_topk=%s, pp rank %d/%d",
            self.num_query_per_req, self.sample_from_anchor, self.dflash_causal,
            self._draft_topk, get_pp_group().rank_in_group, get_pp_group().world_size,
        )

    # ---------------- kv groups ----------------
    def validate_same_kv_cache_group(self, kv_cache_config) -> None:
        # several groups are fine here (handled per group); just log the layout
        groups: dict[str, int] = {}
        for gid, g in enumerate(kv_cache_config.kv_cache_groups):
            for name in g.layer_names:
                groups[name] = gid
        by_group: dict[int, list[str]] = {}
        for name in sorted(self._draft_attn_layer_names):
            by_group.setdefault(groups.get(name, -1), []).append(name)
        logger.info("DSpark: draft layers by kv group: %s", by_group)
        assert -1 not in by_group, f"draft layer without a kv group: {by_group[-1]}"

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        all_attn_layers = _base.get_layers_from_vllm_config(
            self.vllm_config, _base.AttentionLayerBase
        )
        self.validate_same_kv_cache_group(kv_cache_config)
        layer_gid: dict[str, int] = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                if name in self._draft_attn_layer_names:
                    layer_gid[name] = gid
        self._draft_layer_gid = layer_gid
        self._draft_gids = sorted(set(layer_gid.values()))
        self.kv_cache_gid = self._draft_gids[0]
        self.draft_attn_groups = []
        for gid in self._draft_gids:
            names = sorted(n for n, g in layer_gid.items() if g == gid)
            spec = kv_cache_config.kv_cache_groups[gid].kv_cache_spec
            if isinstance(spec, _base.UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[names[0]]
            kbs = (
                kernel_block_sizes[gid]
                if kernel_block_sizes is not None and gid < len(kernel_block_sizes)
                else None
            )
            group = _base.AttentionGroup(
                backend=all_attn_layers[names[0]].get_attn_backend(),
                layer_names=names,
                kv_cache_spec=spec,
                kv_cache_group_id=gid,
            )
            group.create_metadata_builders(self.vllm_config, self.device, kernel_block_size=kbs)
            self.draft_attn_groups.append(group)
            self._gid_block_size[gid] = kbs if kbs is not None else spec.block_size
            self._ctx_slot[gid] = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=self.device)
            self._q_slot[gid] = torch.zeros(
                self.max_padded_query_tokens, dtype=torch.int64, device=self.device
            )
        self.block_size = self._gid_block_size[self.kv_cache_gid]
        logger.info("DSpark: draft kv groups %s block sizes %s", self._draft_gids, self._gid_block_size)

    def _block_table_for(self, gid: int, cad: CommonAttentionMetadata) -> torch.Tensor:
        if gid == self.kv_cache_gid:
            return cad.block_table_tensor
        bt = self._dsv41_runner.input_batch.block_table[gid]
        return bt.get_device_tensor(cad.block_table_tensor.shape[0])

    def _dspark_slot_mapping(self, num_tokens: int, num_actual: int) -> dict[str, torch.Tensor]:
        out = {}
        for name, gid in self._draft_layer_gid.items():
            buf = self._q_slot[gid]
            if num_tokens > num_actual:
                buf[num_actual:num_tokens].fill_(_base.PADDING_SLOT_ID)
            out[name] = buf[:num_tokens]
        return out

    # ---------------- inputs ----------------
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        batch_size = cad.batch_size()
        num_context = target_token_ids.shape[0]
        nq = self.num_query_per_req
        num_query_total = batch_size * nq
        self._dflash_num_context = num_context
        self._dflash_hidden_states = target_hidden_states
        scratch_indices = torch.empty(
            max(1, batch_size * (nq - 1)), dtype=torch.int32, device=self.device
        )
        max_tokens_per_req = cad.max_query_len + nq
        BLOCK_SIZE = min(256, next_power_of_2(max_tokens_per_req))
        num_blocks = (max_tokens_per_req + BLOCK_SIZE - 1) // BLOCK_SIZE
        has_num_rejected = num_rejected_tokens_gpu is not None
        # one kernel launch per kv group: same ids/positions, group-specific slots
        for gid in self._draft_gids:
            bt = self._block_table_for(gid, cad)
            copy_and_expand_dflash_inputs_kernel[(batch_size, num_blocks)](
                next_token_ids_ptr=next_token_ids,
                target_positions_ptr=target_positions,
                out_input_ids_ptr=self.input_ids,
                out_context_positions_ptr=self._context_positions_buffer,
                out_query_positions_ptr=self.positions,
                out_context_slot_mapping_ptr=self._ctx_slot[gid],
                out_query_slot_mapping_ptr=self._q_slot[gid],
                out_token_indices_ptr=scratch_indices,
                block_table_ptr=bt,
                block_table_stride=bt.stride(0),
                query_start_loc_ptr=cad.query_start_loc,
                num_rejected_tokens_ptr=(num_rejected_tokens_gpu if has_num_rejected else 0),
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                block_size=self._gid_block_size[gid],
                num_query_per_req=nq,
                num_speculative_tokens=nq - 1,
                total_input_tokens=num_context,
                BLOCK_SIZE=BLOCK_SIZE,
                HAS_NUM_REJECTED=has_num_rejected,
            )
        if self.sample_from_anchor:
            token_indices_to_sample = self.arange[:num_query_total].to(torch.int64)
        else:
            token_indices_to_sample = scratch_indices.to(torch.int64)
        new_query_start_loc = self.arange[: batch_size + 1] * nq
        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu
        upper = (
            cad.seq_lens_cpu_upper_bound + nq
            if cad.seq_lens_cpu_upper_bound is not None
            else None
        )
        qsl_cpu = torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * nq
        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=effective_seq_lens + nq,
            query_start_loc_cpu=qsl_cpu,
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            seq_lens_cpu_upper_bound=upper,
            num_reqs=cad.num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=nq,
            max_seq_len=cad.max_seq_len + nq,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=self._q_slot[self.kv_cache_gid][:num_query_total],
            causal=self.dflash_causal,
        )
        self._per_gid_cad = {}
        for gid in self._draft_gids:
            self._per_gid_cad[gid] = _dc_replace(
                new_cad,
                block_table_tensor=self._block_table_for(gid, cad),
                slot_mapping=self._q_slot[gid][:num_query_total],
            )
        return num_query_total, token_indices_to_sample, new_cad

    def build_per_group_and_layer_attn_metadata(self, common_attn_metadata, draft_index=0):
        per_group, per_layer = [], {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            cad = self._per_gid_cad.get(gid, common_attn_metadata)
            md = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=cad, draft_index=draft_index
            )
            # (the sparse-SWA builder handles causal=False internally via its
            # non-causal index width; its metadata carries no `causal` field)
            per_group.append(md)
            for name in attn_group.layer_names:
                per_layer[name] = md
        return per_group, per_layer

    def build_model_inputs_first_pass(self, num_tokens, num_input_tokens, mm_embed_inputs):
        num_context = self._dflash_num_context
        ctx_slots = []
        for layer in self.model.model.layers:
            gid = self._draft_layer_gid[layer.attn.swa_cache_layer.prefix]
            ctx_slots.append(self._ctx_slot[gid][:num_context])
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states,
            self._context_positions_buffer[:num_context],
            ctx_slots,
        )
        return (
            dict(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            ),
            num_input_tokens,
        )

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings=None,
    ) -> None:
        num_query_tokens = min(num_tokens, self.max_query_tokens)
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_query_tokens, use_cudagraphs=use_cudagraphs
            )
        )
        # projection only (no slot mapping -> nothing is written to the cache)
        self.model.precompute_and_store_context_kv(
            self._ctx_dummy[:num_tokens], self._context_positions_buffer[:num_tokens]
        )
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._dspark_slot_mapping(num_input_tokens, 0) if self._q_slot else {},
        ):
            self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            )

    # ---------------- drafting ----------------
    @torch.inference_mode()
    def propose(
        self,
        num_speculative_tokens,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata,
        mm_embed_inputs=None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings=None,
    ) -> torch.Tensor:
        assert 1 <= num_speculative_tokens <= self.num_speculative_tokens, (
            num_speculative_tokens, self.num_speculative_tokens)
        self._last_draft_probs = None
        batch_size = common_attn_metadata.batch_size()
        r1_plan = self._dsv41_r1_plan
        r1_mark = r1_plan.mark if r1_plan is not None else None
        if r1_mark is not None:
            r1_mark("draft_inputs")
        import time as _time
        _t = None
        if self._dbg:
            torch.cuda.synchronize()
            _t = [_time.perf_counter()]
        def _mark(name):
            if _t is not None:
                torch.cuda.synchronize()
                now = _time.perf_counter()
                self._t_acc[name] = self._t_acc.get(name, 0.0) + (now - _t[0])
                _t[0] = now
        # [T, hidden * len(target_layer_ids)] -> [T, hidden]
        target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
        num_tokens, token_indices_to_sample, cad = self.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )
        _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(cad)
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )
        _mark("inputs+meta")
        if r1_mark is not None:
            r1_mark("draft_forward")
        model_kwargs, _ = self.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs
        )
        _mark("ctx_insert")
        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._dspark_slot_mapping(num_input_tokens, num_tokens),
        ):
            head_hidden = self.model(**model_kwargs)
        _mark("forward")
        sample_hidden = head_hidden[token_indices_to_sample]
        if r1_mark is not None:
            r1_mark("draft_sampler")
        out = self._sample_sequential(batch_size, sample_hidden)
        _mark("sample")
        if _t is not None:
            self._t_n += 1
            if self._t_n % 32 == 0:
                logger.info("DSpark draft timing (ms/call, %d calls, mode %s, tokens %d): %s",
                            self._t_n, cudagraph_runtime_mode, num_input_tokens,
                            ", ".join(f"{k} {1e3 * v / self._t_n:.1f}" for k, v in self._t_acc.items()))
        if num_speculative_tokens < self.num_speculative_tokens:
            out = out[:, :num_speculative_tokens].contiguous()
        return out

    def _sample_sequential(self, num_reqs: int, sample_hidden: torch.Tensor) -> torch.Tensor:
        from hybrid.abc_probe import drafts as abc_teacher_drafts
        teacher = abc_teacher_drafts(self,num_reqs,sample_hidden)
        if teacher is not None:
            return teacher
        from hybrid.abc_runtime import try_sample as abc_sample  # STAR_ABC integration
        candidate = abc_sample(self, num_reqs, sample_hidden)
        if candidate is not None:
            return candidate
        from hybrid.w26_sampler_bank import try_sample
        if self._dsv41_r1_plan is not None:
            return self._sample_sequential_planned(num_reqs, sample_hidden)
        r2 = sys.modules.get("hybrid.r2_sampler")
        if r2 is not None and num_reqs > 1:
            candidate = r2.try_rows(self, num_reqs, sample_hidden, None)
            if candidate is not None:
                return candidate
        candidate = try_sample(self, num_reqs, sample_hidden)
        if candidate is not None:
            return candidate
        return self._sample_sequential_incumbent(num_reqs, sample_hidden)

    def _sample_sequential_planned(self, num_reqs: int, sample_hidden: torch.Tensor):
        # R1 plan present. try_sample falls back ('random_rows' / 'glog_rows') whenever
        # the plan samples or logs, so those batches never enter the captured bank.
        from hybrid import dsv41_r1 as r1
        from hybrid.w26_sampler_bank import try_sample
        plan = self._dsv41_r1_plan
        if plan.num_rows != num_reqs:
            r1.warn_once("rows", "R1: plan rows %d != drafter batch %d; plan dropped",
                         plan.num_rows, num_reqs)
            self._dsv41_r1_plan = None
            self._dsv41_r1_path = "plan_dropped"
            candidate = try_sample(self, num_reqs, sample_hidden)
            if candidate is not None:
                return candidate
            return self._sample_sequential_incumbent(num_reqs, sample_hidden)
        r2 = sys.modules.get("hybrid.r2_sampler")
        candidate = None
        if r2 is not None and num_reqs > 1:
            candidate = r2.try_rows(self, num_reqs, sample_hidden, plan)
        if candidate is None:
            candidate = try_sample(self, num_reqs, sample_hidden)
            if candidate is not None:
                self._dsv41_r1_path = "bank"
            elif plan.needs_r1_path:
                self._dsv41_r1_path = "r1"
                candidate = self._sample_sequential_r1(num_reqs, sample_hidden, plan)
            else:
                self._dsv41_r1_path = "incumbent"
                candidate = self._sample_sequential_incumbent(num_reqs, sample_hidden)
        if plan.xcheck:
            r1.xcheck(self, candidate, self._last_draft_probs)
        return candidate

    def _sample_sequential_r1(self, num_reqs: int, sample_hidden: torch.Tensor, plan):
        # Option (a) of AUDIT-DRAFTER-T1: eager full-vocab q for rows with T_q > 0, exported
        # as fp32 [B, n, V] in input-batch row order; every other row keeps the incumbent
        # argmax on the same logits tensor and exports one-hot(tok) when q is exported.
        from hybrid import dsv41_r1 as r1
        if self._draft_topk is not None:
            r1.warn_once("topk_r1", "R1: draft top-k path; plan ignored (greedy drafts)")
            self._dsv41_r1_path = "incumbent_topk"
            return self._sample_sequential_incumbent(num_reqs, sample_hidden)
        n = self.num_speculative_tokens
        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n, vocab)
        prev = self.input_ids[self._anchor_idx[:num_reqs]]
        sample = plan.any_sample and r1.sampling_supported(self, vocab, plan)
        self._dsv41_r1_path = "r1_sampled" if sample else "r1_greedy"
        thr = self._conf_threshold()
        drafts, q, logits_all, embeds = r1.draft_chain(
            self.model,
            base_logits,
            prev,
            r1.tq_tensor(plan, self.device) if sample else None,
            self._dsv41_r1_generator() if sample else None,
            all_sample=plan.all_sample,
            keep_logits=plan.any_glog,
            keep_embeds=thr > 0,
        )
        if q is not None:
            self._last_draft_probs = q
        if logits_all is not None:
            self._dsv41_r1_glog = (logits_all, list(plan.req_ids), list(plan.tq), sample)
        if thr > 0 and self.model.model.confidence_head is not None:
            conf = self.model.compute_confidence(
                sample_hidden, torch.stack(embeds, dim=1).flatten(0, 1)
            ).view(num_reqs, n)
            keep = (conf.cumprod(dim=1) >= thr).to(torch.int32).cumprod(dim=1).sum(dim=1)
            keep_l = keep.tolist()
            rows = drafts.tolist()
            return [row[:k] for row, k in zip(rows, keep_l)]
        return drafts

    def _dsv41_r1_generator(self) -> torch.Generator:
        # same seed on every rank and advanced by identical calls (same plan everywhere)
        if self._dsv41_r1_gen is None:
            from hybrid import dsv41_r1 as r1

            gen = torch.Generator(device=self.device)
            gen.manual_seed(r1.noise_seed(self.vllm_config))
            self._dsv41_r1_gen = gen
        return self._dsv41_r1_gen

    def take_last_draft_probs(self) -> torch.Tensor | None:
        # read by the runner after every propose; only the R1 path above sets it
        return self._last_draft_probs

    def _sample_sequential_incumbent(self, num_reqs: int, sample_hidden: torch.Tensor) -> torch.Tensor:
        n = self.num_speculative_tokens
        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n, vocab)
        drafts = torch.empty(num_reqs, n, dtype=torch.int64, device=self.device)
        prev = self.input_ids[self._anchor_idx[:num_reqs]]
        if self._draft_topk is not None:
            base_values, cand = base_logits.topk(self._draft_topk, dim=-1)
        thr = self._conf_threshold()
        embeds = []
        for i in range(n):
            markov_embed = self.model.markov_embed(prev)
            if thr > 0:
                embeds.append(markov_embed)
            if self._draft_topk is not None:
                logits_i = self.model.apply_markov_bias_gathered(
                    markov_embed, base_logits[:, i], base_values[:, i], cand[:, i]
                )
            else:
                logits_i = base_logits[:, i] + self.model.markov_bias(markov_embed)
            tok = self.model.map_draft_to_target(logits_i.argmax(dim=-1))
            drafts[:, i] = tok
            prev = tok
        if thr > 0 and self.model.model.confidence_head is not None:
            # adaptive verification (synchronous PP path): keep the prefix of drafts whose
            # cumulative acceptance confidence stays above the threshold; the scheduler
            # then verifies exactly that many rows, so low-confidence steps stay cheap
            conf = self.model.compute_confidence(
                sample_hidden, torch.stack(embeds, dim=1).flatten(0, 1)
            ).view(num_reqs, n)
            keep = (conf.cumprod(dim=1) >= thr).to(torch.int32).cumprod(dim=1).sum(dim=1)
            keep_l = keep.tolist()
            rows = drafts.tolist()
            return [row[:k] for row, k in zip(rows, keep_l)]
        return drafts

    def _conf_threshold(self) -> float:
        # /dump/dspark_conf (mounted rw) overrides the boot-time default so the
        # threshold can be tuned without a restart
        try:
            with open("/dump/dspark_conf") as f:
                return float(f.read().strip() or 0)
        except Exception:
            return self._conf_default

    # ---------------- loading ----------------
    def load_model(self, target_model: torch.nn.Module) -> None:
        import vllm.model_executor.model_loader.weight_utils as wu

        # only the mtp.* draft tensors are read from the (475 GiB) target checkpoint
        wu._DSV41_ONLY_RE = re.compile(r"^mtp\.")
        # the base loader copies target_model.config.image_token_index for multimodal
        # targets; V4.1 names it image_token_id
        tcfg = getattr(target_model, "config", None)
        if tcfg is not None and not hasattr(tcfg, "image_token_index"):
            try:
                tcfg.image_token_index = getattr(tcfg, "image_token_id", None)
            except Exception:  # pragma: no cover
                pass
        try:
            super().load_model(target_model)
        finally:
            wu._DSV41_ONLY_RE = None
        if get_pp_group().world_size > 1:
            self._load_embedding_from_checkpoint()
        lm = getattr(self.model, "lm_head", None)
        tlm = getattr(target_model, "lm_head", None)
        if lm is not None and tlm is not None and lm is not tlm:
            self.model.lm_head = tlm
            logger.info("DSpark: draft lm_head aliased to the target's")
        # Only the draft's own sliding-window caches are drafter KV layers (the V2
        # speculator's get_draft_kv_cache_layer_names); anything else the draft
        # attention registers belongs to the target's groups.
        own = set(self.model.get_draft_kv_cache_layer_names())
        extra = sorted(self._draft_attn_layer_names - own)
        logger.info("DSpark: draft kv layers %s; other draft-registered attn layers: %s",
                    sorted(own), extra)
        self._draft_attn_layer_names = own

    def _load_embedding_from_checkpoint(self) -> None:
        from safetensors import safe_open

        model_dir = os.environ.get("DSV41_MODEL_DIR") or self.vllm_config.model_config.model
        index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        shard = index["weight_map"]["embed.weight"]
        with safe_open(os.path.join(model_dir, shard), "pt", device="cpu") as f:
            w = f.get_tensor("embed.weight")
        embed = self.model.model.embed_tokens
        loader = getattr(embed.weight, "weight_loader", None)
        if loader is not None:
            loader(embed.weight, w)
        else:
            embed.weight.data.copy_(w.to(embed.weight.dtype))
        logger.info("DSpark: loaded embed.weight %s from %s (PP rank %d)",
                    tuple(w.shape), shard, get_pp_group().rank_in_group)
