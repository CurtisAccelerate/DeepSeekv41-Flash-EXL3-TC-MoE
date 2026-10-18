#!/bin/bash
# DeepSeek V4.1 Flash, tensor parallel 4.
# Full CUDA graphs, host-staged all-reduce, tensor-core EXL3 MoE.
# Row dispatch: coop <= 42, moe_out_v2 for 43-64, moe_out_prefill for 65-2048.
#
# Cherry-pick the host all-reduce from
# https://github.com/Morrowmake/vllm-cmp170hx at 18305948ac
# (vllm/sm80, envs.py, and the CUDA communicator selection).
# Wire exl3_tc_hook.try_build into expert-state setup and
# exl3_tc_hook.try_apply at the start of the fused MoE call.
#
# This script does not download weights and does not build a container image.
set -eu

: "${DSV41_IMAGE:?set DSV41_IMAGE to the serving image}"
: "${DSV41_MODEL_DIR:?set DSV41_MODEL_DIR to the EXL3 checkpoint directory}"
: "${DSV41_EXL3_PY:?set DSV41_EXL3_PY to the EXL3 plugin that calls the hook}"
: "${DSV41_TC_HOOK:?set DSV41_TC_HOOK to exl3_tc_hook.py}"
: "${DSV41_TC_SO:?set DSV41_TC_SO to _dsv41_tc_moe_C.abi3.so}"
: "${DSV41_SM80_DIR:?set DSV41_SM80_DIR to the sm80 package from vllm-cmp170hx}"
: "${DSV41_ENVS_PY:?set DSV41_ENVS_PY to envs.py from vllm-cmp170hx}"
: "${DSV41_CUDA_PY:?set DSV41_CUDA_PY to platforms/cuda.py from vllm-cmp170hx}"

DOCKER=(docker)
if [ -n "${DSV41_DOCKER:-}" ]; then
  # shellcheck disable=SC2206
  DOCKER=(${DSV41_DOCKER})
fi

ARENA=()
if [ -n "${DSV41_ARENA_CACHE_DIR:-}" ]; then
  mkdir -p "${DSV41_ARENA_CACHE_DIR}"
  ARENA+=(
    -v "${DSV41_ARENA_CACHE_DIR}:${DSV41_ARENA_CACHE_DIR}"
    -e VLLM_EXL3_ARENA_CACHE=1
    -e VLLM_EXL3_ARENA_CACHE_DIR="${DSV41_ARENA_CACHE_DIR}"
  )
fi

"${DOCKER[@]}" rm -f dsv41-tp4 >/dev/null 2>&1 || true
"${DOCKER[@]}" run -d \
  --name dsv41-tp4 \
  --cap-add SYS_NICE \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${DSV41_MODEL_DIR}:/model-view:ro" \
  -v "${DSV41_EXL3_PY}:/usr/local/lib/python3.12/dist-packages/vllm_exl3/exl3.py:ro" \
  -v "${DSV41_SM80_DIR}:/usr/local/lib/python3.12/dist-packages/vllm/sm80:ro" \
  -v "${DSV41_ENVS_PY}:/usr/local/lib/python3.12/dist-packages/vllm/envs.py:ro" \
  -v "${DSV41_CUDA_PY}:/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py:ro" \
  -v "${DSV41_TC_HOOK}:/opt/dsv41-tc-moe/exl3_tc_hook.py:ro" \
  -v "${DSV41_TC_SO}:/opt/dsv41-tc-moe/_dsv41_tc_moe_C.abi3.so:ro" \
  -e VLLM_EXL3_MODEL_DIR=/model-view \
  -e VLLM_PLUGINS=vllm_exl3 \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e DSV41_DSPARK_V1=1 \
  -e VLLM_EXL3_COOP=1 \
  -e VLLM_HOST_ALLREDUCE=1 \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_IGNORE_DISABLED_P2P=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e VLLM_EXL3_TC_MOE_V2=1 \
  -e VLLM_EXL3_TC_MOE_PREFILL=1 \
  -e VLLM_EXL3_TC_PREFILL_ROWS=2048 \
  -e VLLM_EXL3_TC_MOE_SO=/opt/dsv41-tc-moe/_dsv41_tc_moe_C.abi3.so \
  -e VLLM_EXL3_MOE_TP_ALIGN=128 \
  -e VLLM_EXL3_MOE_TP_ROTATE=1 \
  -e TORCH_CUDA_ARCH_LIST=8.0 \
  "${ARENA[@]}" \
  --entrypoint vllm \
  "$DSV41_IMAGE" \
  serve /model-view \
  --quantization exl3 \
  --load-format safetensors \
  --safetensors-load-strategy lazy \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 4 \
  --served-model-name deepseek-v4.1-flash \
  --host 0.0.0.0 \
  --port 8095 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.97 \
  --kv-cache-dtype fp8_ds_mla \
  --engram-config '{"cpu_offload": true}' \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 2, 3, 4, 6, 8, 12, 18, 24, 30, 36, 42, 48]}' \
  --async-scheduling \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 5}' \
  --tokenizer-mode deepseek_v41 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v41 \
  --reasoning-parser deepseek_v41

echo "container $(${DOCKER[@]} inspect -f '{{.Id}}' dsv41-tp4)"
