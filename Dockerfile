# syntax=docker/dockerfile:1
FROM python:3.11-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /bin/uv
WORKDIR /src
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv build --wheel --out-dir /dist

FROM python:3.11-slim
ARG WITH_OTEL=1
RUN useradd --uid 10001 --create-home vaultkv && mkdir -p /data && chown vaultkv /data
COPY --from=build /dist/*.whl /tmp/
RUN WHEEL=$(ls /tmp/*.whl) \
    && if [ "$WITH_OTEL" = "1" ]; then pip install --no-cache-dir "${WHEEL}[otel]"; \
       else pip install --no-cache-dir "$WHEEL"; fi \
    && rm /tmp/*.whl
# Sample corpora for smoke tests and the CI ingest job.
COPY examples /opt/vaultkv/examples
USER 10001
WORKDIR /home/vaultkv
EXPOSE 8080
ENTRYPOINT ["vaultkv"]
CMD ["api", "--host", "0.0.0.0", "--port", "8080", "--db", "/data/vaultkv.sqlite3"]
