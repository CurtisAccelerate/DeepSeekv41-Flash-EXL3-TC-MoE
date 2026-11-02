# Versioned derivative of the exact GPU-qualified runtime; no GPU used at build.
ARG BASE_IMAGE=sha256:7ecbcf725f31d1b5bc05fc6863a16a94ef9c1c6cf7441e0a5c257fcd430e4d94
FROM ${BASE_IMAGE}
# Keep this to two filesystem layers: the qualified base is already deeply layered.
COPY . /opt/star-release/
RUN python3 /opt/star-release/materialize_host_overlay.py && CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void python3 /opt/star-release/verify_release.py
LABEL org.opencontainers.image.title="DS1017 balanced BF16 fixed-five recipe" org.opencontainers.image.version="2026.11.02" org.opencontainers.image.source="https://github.com/CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE"
ENTRYPOINT ["python3"]
