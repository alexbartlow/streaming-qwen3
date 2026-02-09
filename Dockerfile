# Qwen3-Coder-30B-A3B with RAM-streamed MoE inference
# NVIDIA CUDA 12.4 + Flash Attention 2

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch with CUDA 12.4
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Flash attention build deps + install (compiles from source, takes ~10 min)
RUN pip install psutil ninja packaging
RUN pip install flash-attn --no-build-isolation

# Dependencies
RUN pip install \
    transformers>=4.45.0 \
    accelerate \
    fastapi \
    uvicorn[standard] \
    pydantic>=2.0 \
    optimum-quanto

WORKDIR /app

# Copy package
COPY pyproject.toml README.md /app/
COPY src/ /app/src/

# Install package
RUN pip install -e .

ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

EXPOSE 8000

CMD ["python", "-m", "streaming_qwen.cli", "--server"]
