FROM python:3.12-slim AS builder

WORKDIR /app

COPY src /app/src
COPY pyproject.toml .
COPY README.md .

RUN apt-get update && apt-get install -y gcc && pip install uv && uv build 

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/dist/*.whl /app/

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu *.whl

EXPOSE 8000

CMD ["mlflow_api"]
