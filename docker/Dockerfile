# syntax=docker/dockerfile:1
# Multi-stage build for the Grey-Resolve REST service (Phase 3 packaging).
#
# Backbone weights are NEVER baked into the image: the InsightFace buffalo_l
# weights are non-commercial (docs/BACKBONE_LICENSES.md) and must not be
# redistributed. Mount them at runtime; docker/serve.py checks for them and
# exits with a clear error when they are missing.

FROM python:3.12-slim AS builder
WORKDIR /build
# build-essential: insightface ships as an sdist and compiles a small extension.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY src ./src
# [index,backbone,api] are the service's runtime extras; the [qdrant] extra is
# included so GREY_RESOLVE_INDEX=qdrant works out of the box (docker-compose).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[index,backbone,api,qdrant]"

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 1000 app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    GREY_RESOLVE_STORAGE=/data \
    GREY_RESOLVE_INSIGHTFACE_ROOT=/home/app/.insightface \
    GREY_RESOLVE_CONFIG_DIR=/app/configs
WORKDIR /app
COPY configs ./configs
COPY scripts/fetch_backbone.py ./scripts/fetch_backbone.py
COPY docker/serve.py ./docker/serve.py
RUN mkdir -p /data && chown -R app:app /data /app
USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
# serve.py verifies weights before loading any model (clear error, no baked weights).
ENTRYPOINT ["python", "docker/serve.py"]
