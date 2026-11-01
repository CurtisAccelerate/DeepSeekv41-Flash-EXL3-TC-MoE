# full_stage.py creates and verifies the source-specific local alias before build.
ARG BASE_IMAGE=star-prefill-base:dc87ecb3dfdca94c
FROM ${BASE_IMAGE}
COPY prefill-reference /opt/star-prefill/reference
COPY tools/prefill_verify_base.py /opt/star-prefill/tools/prefill_verify_base.py
RUN CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void python3 /opt/star-prefill/tools/prefill_verify_base.py
COPY prefill-native /opt/star-prefill/native
RUN CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 python3 /opt/star-prefill/native/build_try.py --out /opt/star-abc/lib --build-dir /opt/star-prefill/build
COPY . /opt/star-prefill/package
COPY overlay /opt/star-abc/overlay
COPY overlay /
COPY host-overlay /opt/star-abc/host-overlay
COPY tools /opt/star-abc/tools
RUN CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void python3 /opt/star-abc/tools/prefill_image_manifest.py
RUN CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void python3 /opt/star-abc/tools/prefill_cpu_check.py
WORKDIR /
ENTRYPOINT ["python3"]
