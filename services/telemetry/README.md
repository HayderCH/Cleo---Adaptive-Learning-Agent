# Telemetry Service

Telemetry Service

Run the FastAPI telemetry ingestion service and smoke test locally.

Prereqs

- Python 3.11 environment with requirements installed

Run the API

- From the repo root, start the server in dev mode:
  - Use: python -m uvicorn services.telemetry.main:app --host 127.0.0.1 --port 8000 --reload

Smoke test

- In another terminal, run:
  - python services/telemetry/client_smoke_test.py
- It will call:
  - GET /healthz
  - POST /events
  - POST /ingest_sample

Output files

- data/raw/events_YYYYMMDD.jsonl
- data/processed/samples_YYYYMMDD.jsonl

Notes

- JSONL is append-only; rotate by day via the YYYYMMDD filename.
  FastAPI service to ingest raw events and full samples into JSONL files under `data/`.

## Endpoints

- `GET /healthz` — liveness check
- `POST /events` — append an event envelope to `data/raw/events_YYYYMMDD.jsonl`
- `POST /ingest_sample` — append a full schema-compliant sample to `data/processed/samples_YYYYMMDD.jsonl`

## Run locally

- Option A (module):
  - python -m uvicorn services.telemetry.main:app --reload --host 127.0.0.1 --port 8000
- Option B (script):
  - python services/telemetry/main.py

## Smoke test

- Ensure the server is running on http://127.0.0.1:8000
- Run: python services/telemetry/client_smoke_test.py

The smoke test will hit `/healthz`, then POST to `/events` and `/ingest_sample`, printing the responses and the target files.
