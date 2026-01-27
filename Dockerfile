FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        sox \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md MANIFEST.in /app/
COPY qwen_tts /app/qwen_tts

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch==2.3.1+cu121 \
        torchaudio==2.3.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["qwen-tts-api"]
