# ── Stage 1: Builder ────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed for faster-whisper (ffmpeg) + compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_healthcare.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements_healthcare.txt

# ── Stage 2: Runtime ────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy only what's needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Application files
COPY healthcare_main.py    .
COPY healthcare_index.html .
COPY cipla_data.json       .

# FastAPI runs on 8001 (matches healthcare_main.py uvicorn config)
EXPOSE 8001

# Download faster-whisper "base" model at build time so cold starts are fast
RUN python - <<'EOF'
from faster_whisper import WhisperModel
WhisperModel("base", device="cpu", compute_type="int8")
print("faster-whisper base model cached ✓")
EOF

ENV OLLAMA_BASE_URL=http://ollama:11434
ENV OLLAMA_MODEL=meditron:latest
ENV TZ=UTC

CMD ["uvicorn", "healthcare_main:app", "--host", "0.0.0.0", "--port", "8001"]