ARG PYTORCH_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface

WORKDIR /workspace/project
COPY stage2_gpu/requirements-linux.lock.txt /tmp/requirements-linux.lock.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /tmp/requirements-linux.lock.txt

COPY . /workspace/project
CMD ["python", "-m", "stage2_gpu.preflight", "--mode", "gpu"]
