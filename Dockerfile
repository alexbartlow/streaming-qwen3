# Qwen3-Coder-30B-A3B with RAM-streamed MoE inference
# PyTorch NGC container with Flash Attention 2 prebuilt

FROM nvcr.io/nvidia/pytorch:24.12-py3

ENV DEBIAN_FRONTEND=noninteractive

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
