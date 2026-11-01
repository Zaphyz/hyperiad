# Hyperiad's server image. Build context: the repository root (image_build_context in wrangler.jsonc).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app
COPY . .
RUN uv sync --frozen --no-dev --package hyperiad \
    && useradd --create-home --uid 10001 hyperiad

ENV PATH=/app/.venv/bin:$PATH
USER hyperiad
EXPOSE 8080

# Limits come from the Worker (envVars in src/index.ts); the defaults match it.
CMD ["sh", "-c", "exec hyperiad serve --host 0.0.0.0 --port 8080 --isolate --max-jobs 1 --budget \"${HYPERIAD_BUDGET:-2e9}\" --job-timeout \"${HYPERIAD_JOB_TIMEOUT:-60}\" --job-memory \"${HYPERIAD_JOB_MEMORY:-1024}\""]
