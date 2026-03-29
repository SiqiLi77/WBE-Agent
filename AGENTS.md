# Repository Guidelines

## Project Structure & Module Organization
`src/` holds production code by workflow: `src/data_pipeline/` for stepwise ingestion and cleaning, `src/anomaly_detection/` for detectors and event catalogs, `src/agent/` for investigation logic, prompts, schema, and tools, `src/evaluation/` for baselines and metrics, and `src/webapi/` for API endpoints. Keep shared helpers in `src/utils/` and config access in `src/config.py`.

Use `scripts/` for runnable entrypoints, `config/settings.yaml` for thresholds and paths, `docs/` for design notes, `paper/` for manuscript assets, `data/` for raw/processed/labeled datasets, and `outputs/` for reports, traces, and experiment artifacts. `web/` contains the Next.js dashboard. Treat `tmp/` as scratch space, not production code.

## Build, Test, and Development Commands
Create an environment and install dependencies:
`python -m venv .venv`
`.\.venv\Scripts\activate`
`pip install -r requirements.txt`

Core workflows:
`python scripts/prepare_data.py --all` downloads and prepares source datasets.
`python scripts/run_pipeline.py --step all` runs pipeline steps 1-7.
`python scripts/run_detection.py --min-votes 2` builds anomaly events.
`python scripts/run_agent.py --max-events 10` runs agent investigations.
`python scripts/run_evaluation.py --baselines-only` evaluates baseline models.

If you touch the dashboard:
`cd web && npm install`
`npm run dev` starts the UI, and `npm run build` checks production build health.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, snake_case for modules/functions/variables, PascalCase for classes, and explicit type hints on public interfaces. Keep pipeline modules in `step{n}_{topic}.py` format such as `step6_merge.py`. Prefer `pathlib.Path`, centralized settings access, and `loguru` instead of `print`.

## Testing Guidelines
There is no committed `pytest` suite yet. Every change should include a script-level smoke test that exercises the affected workflow. For data pipeline changes, inspect `outputs/quality_report/*.csv` and note row-count or metric deltas. Temporary checks in `tmp/smoke_test_*.py` are useful locally but should not replace reproducible script runs.

## Commit & Pull Request Guidelines
Recent history uses short, imperative, scope-first subjects such as `agent v3: ...` or `fix agent epidemic detection: ...`. Keep commits focused and descriptive. PRs should include scope, commands run, affected paths, and output evidence such as CSV snippets, metric deltas, or dashboard screenshots. Do not commit secrets or large raw datasets.
