# Web UI Quickstart

This project now includes a web stack:

- Backend API: `FastAPI` (`src/webapi/main.py`)
- Frontend UI: `Next.js` (`web/`)

## 1) Install dependencies

Backend:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Frontend:

```bash
cd web
npm install
```

## 2) Start backend API

From repo root:

```bash
python scripts/run_api.py
```

API defaults to `http://localhost:8000`.

Optional CORS override:

```bash
set WBE_API_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

## 3) Start frontend

In `web/`:

```bash
npm run dev
```

UI defaults to `http://localhost:3000` and calls `http://localhost:8000` by default.

Set custom API URL with:

```bash
set NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

## 4) What the UI can do

- Show summary metrics from `outputs/anomaly_event_catalog.csv` and `outputs/investigation_results.csv`
- Browse events and view per-event trace details
- Trigger jobs for `prepare_data`, `pipeline`, `detection`, `agent`, and `evaluation`
- Stream job log tail and status (`queued/running/succeeded/failed`)
