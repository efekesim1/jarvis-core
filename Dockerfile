# jarvis-core
#
# Agent workloads run in containers, so what runs on a laptop and what runs in
# the cloud are the same thing. No secret is baked in: a provider credential
# arrives at run time through the environment, and the image is useful without
# one because the suite and the demos run against the offline provider.

FROM python:3.15-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    JARVIS_ENVIRONMENT=sandbox \
    JARVIS_DATA_DIR=/var/jarvis

WORKDIR /app

# Dependencies first, so a code change does not reinstall the world.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --quiet --upgrade pip "setuptools>=83.0.0" \
    && pip install --quiet -e ".[dev]"

COPY agents/ ./agents/
COPY config/ ./config/
COPY docs/ ./docs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY examples/ ./examples/
COPY Makefile ./

# Nothing in here runs as root.
RUN useradd --create-home --shell /bin/bash jarvis \
    && mkdir -p /var/jarvis \
    && chown -R jarvis:jarvis /app /var/jarvis
USER jarvis

# Proves the image is sound without needing a key or a network.
HEALTHCHECK --interval=60s --timeout=20s --start-period=5s --retries=2 \
    CMD python -c "import jarvis_core; from jarvis_core.agents.registry import build_registry; assert build_registry().all()" || exit 1

ENTRYPOINT []
CMD ["jarvis", "--help"]
