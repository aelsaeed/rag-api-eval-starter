FROM python:3.11-slim

ARG INSTALL_ML=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAG_FAKE_EMBEDDINGS=1 \
    HF_HOME=/home/rag/.cache/huggingface

RUN groupadd --gid 10001 rag \
    && useradd --uid 10001 --gid rag --create-home --home-dir /home/rag \
        --shell /usr/sbin/nologin rag

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY app /app/app
COPY eval /app/eval
COPY data /app/data

RUN python -m pip install --no-cache-dir --upgrade pip \
    && case "$INSTALL_ML" in \
        true) python -m pip install --no-cache-dir ".[ml]" ;; \
        false) python -m pip install --no-cache-dir . ;; \
        *) echo "INSTALL_ML must be 'true' or 'false'" >&2; exit 1 ;; \
    esac \
    && python -m pip check \
    && mkdir /workspace \
    && chown rag:rag /workspace

WORKDIR /workspace

USER rag:rag

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).close()"]

STOPSIGNAL SIGTERM

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
