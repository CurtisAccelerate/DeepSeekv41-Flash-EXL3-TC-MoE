# Uses existing pinned inference kernels; no GPU at build.
ARG BASE_IMAGE=star-release-runtime:7ecbcf725f31
FROM ${BASE_IMAGE}
ENV DS1017_LANE=speed STAR_ABC=1 STAR_ABC_SCOPE=draft-speed STAR_ABC_KV_DTYPE=bfloat16 STAR_ABC_CAPTURE_MIB=1152 STAR_ABC_OPERATOR_CHECKS=1 STAR_PREFILL=1
# Two filesystem layers avoid the retained base's overlay-depth limit.
COPY . /opt/star-release/
RUN python3 /opt/star-release/materialize_host_overlay.py && CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void python3 /opt/star-release/verify_release.py
LABEL org.opencontainers.image.title="DS1017 speed default with BF16 and quality presets" org.opencontainers.image.version="2026.11.02-speed-k5" org.opencontainers.image.source="https://github.com/CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE"
ENTRYPOINT ["python3","/opt/star-release/lane_entrypoint.py"]
CMD ["describe"]
