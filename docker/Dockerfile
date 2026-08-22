# check=skip=InvalidDefaultArgInFrom
# TrainFactory production API image. The digest-pinned GPU provider image is
# supplied by the build/release caller and verified before dependencies change.
ARG API_BASE_IMAGE
FROM ${API_BASE_IMAGE} AS runtime

ARG BUILD_VERSION
ARG VCS_REF
ARG VCS_DATE
ARG SOURCE_REPOSITORY
RUN test -n "${BUILD_VERSION}" \
    && test -n "${VCS_REF}" \
    && test -n "${VCS_DATE}" \
    && test -n "${SOURCE_REPOSITORY}"

LABEL org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${VCS_DATE}" \
      org.opencontainers.image.source="${SOURCE_REPOSITORY}"

WORKDIR /app

COPY requirements/base-image-provided.txt requirements/base-image-provided.txt
COPY requirements/runtime.lock requirements/runtime.lock
COPY scripts/check_base_image_providers.py scripts/check_base_image_providers.py
RUN --mount=type=cache,target=/root/.cache/pip \
    python scripts/check_base_image_providers.py requirements/base-image-provided.txt \
    && python -m pip --isolated install \
        --index-url https://pypi.org/simple \
        --require-hashes --only-binary=:all: -r requirements/runtime.lock \
    && python scripts/check_base_image_providers.py requirements/base-image-provided.txt \
    && python -c "import torch; assert torch.__version__ == '2.6.0+cu124'; assert torch.version.cuda == '12.4'"

COPY pyproject.toml README.md ./
COPY train_factory/ train_factory/
COPY --chmod=0555 docker/entrypoint.sh /app/docker/entrypoint.sh
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip --isolated install --no-deps --no-build-isolation . \
    && rm -rf build train_factory.egg-info \
    && mkdir -p /app/data /app/models /app/output /app/cache /app/logs

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    GPU_PREFLIGHT_MODE=required \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TRAINING_CACHE=/app/cache \
    MODELS_DIR=/app/models \
    OUTPUT_DIR=/app/output \
    HF_HOME=/app/cache/hf_home \
    MODELSCOPE_CACHE=/app/cache/modelscope \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"API_PORT\", \"8000\")}/health', timeout=5)"]

EXPOSE 8000
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["python", "-m", "train_factory.api.server"]
