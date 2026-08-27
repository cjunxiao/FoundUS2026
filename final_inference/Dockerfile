FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime@sha256:fc47f8018254e6df30f48c48f2db1c758d44de21a8c553de1a1c451a65baa70a

LABEL org.opencontainers.image.title="FUB 2026 final single-model inference" \
      org.opencontainers.image.description="Single-model task-specific USFM LoRA and contextual heatmap correction" \
      org.opencontainers.image.version="postcompetition-v1.3.1" \
      org.opencontainers.image.licenses="Apache-2.0 AND CC-BY-NC-4.0"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    TORCH_HOME=/tmp/torch \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY predict.py model.py model_manifest.json best_model.pth THIRD_PARTY_NOTICES.md /app/

ENTRYPOINT ["python3", "/app/predict.py"]
