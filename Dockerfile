# The service: the API with the built UI. Replay mode by default (recorded runs, no API key, no
# database); live mode needs an API key and the compose PostgreSQL. The statistical guardrail
# runs a sandbox container, which this image cannot start, so it is off here: a comparative
# question is answered with SQL and a notice that says so. Run the service on the host
# (`uv run python -m src.serving --mode live`) for the full guardrail.

FROM node:20-slim@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0 AS ui
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 UV_LINK_MODE=copy
RUN pip install --no-cache-dir uv==0.11.3
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project
COPY configs/ configs/
COPY prompts/ prompts/
COPY dictionary/ dictionary/
COPY src/ src/
COPY results/demo/ results/demo/
COPY results/metrics/calibration.json results/metrics/calibration.json
# the registry: which agent configuration is the champion, and what it was evaluated as
COPY results/registry/ results/registry/
COPY --from=ui /ui/dist ui/dist
RUN useradd --system --uid 10001 analyst && mkdir -p data/serving && chown -R analyst data
USER 10001
ENV ANALYST_SERVING_MODE=replay ANALYST_GUARDRAIL=0 PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"
CMD ["python", "-m", "src.serving", "--host", "0.0.0.0", "--port", "8000"]