FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the package first so dependency layers are cached across source edits.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[demo,langfuse]"

COPY examples ./examples

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"

CMD ["mcp-orchestrator", "serve", "--config", "examples/servers.docker.toml", "--host", "0.0.0.0", "--port", "8000"]
