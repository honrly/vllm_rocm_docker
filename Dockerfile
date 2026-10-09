FROM rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1

ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

WORKDIR /app

COPY app /app
