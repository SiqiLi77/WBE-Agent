from __future__ import annotations

import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, settings
from src.webapi.schemas import JobCreateRequest, JobRecord


MAX_LOG_TAIL_LINES = 120


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            items = list(self._jobs.values())
        return sorted(items, key=lambda x: x.created_at, reverse=True)

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def create_job(self, req: JobCreateRequest) -> JobRecord:
        command = self._build_command(req.job_type, req.args)
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        job = JobRecord(
            id=job_id,
            job_type=req.job_type,
            status="queued",
            command=command,
            args=req.args,
            created_at=datetime.utcnow(),
            log_file=str(self._log_path(job_id)),
            log_tail=[],
        )
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        thread.start()
        return job

    def _build_command(self, job_type: str, args: dict[str, Any]) -> list[str]:
        base = [sys.executable]
        if job_type == "prepare_data":
            cmd = base + [str(PROJECT_ROOT / "scripts" / "prepare_data.py")]
            if args.get("all", True):
                cmd.append("--all")
            else:
                if args.get("download"):
                    cmd.append("--download")
                if args.get("explore"):
                    cmd.append("--explore")
                if args.get("select_sites"):
                    cmd.append("--select-sites")
                if args.get("prepare_hhs"):
                    cmd.append("--prepare-hhs")
            return cmd

        if job_type == "pipeline":
            step = str(args.get("step", "all"))
            return base + [str(PROJECT_ROOT / "scripts" / "run_pipeline.py"), "--step", step]

        if job_type == "detection":
            cmd = base + [str(PROJECT_ROOT / "scripts" / "run_detection.py")]
            if "min_votes" in args and args["min_votes"] is not None:
                cmd.extend(["--min-votes", str(args["min_votes"])])
            if "output" in args and args["output"]:
                cmd.extend(["--output", str(args["output"])])
            return cmd

        if job_type == "agent":
            cmd = base + [str(PROJECT_ROOT / "scripts" / "run_agent.py")]
            if "max_events" in args and args["max_events"] is not None:
                cmd.extend(["--max-events", str(args["max_events"])])
            if "event_id" in args and args["event_id"]:
                cmd.extend(["--event-id", str(args["event_id"])])
            if "events_file" in args and args["events_file"]:
                cmd.extend(["--events-file", str(args["events_file"])])
            if "model" in args and args["model"]:
                cmd.extend(["--model", str(args["model"])])
            return cmd

        if job_type == "evaluation":
            cmd = base + [str(PROJECT_ROOT / "scripts" / "run_evaluation.py")]
            if args.get("baselines_only"):
                cmd.append("--baselines-only")
            if args.get("ablation"):
                cmd.append("--ablation")
            if "labels_file" in args and args["labels_file"]:
                cmd.extend(["--labels-file", str(args["labels_file"])])
            if "predictions_file" in args and args["predictions_file"]:
                cmd.extend(["--predictions-file", str(args["predictions_file"])])
            return cmd

        raise ValueError(f"Unsupported job_type: {job_type}")

    def _log_path(self, job_id: str) -> Path:
        log_dir = PROJECT_ROOT / settings.logging.log_dir / "web_jobs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / f"{job_id}.log"

    def _append_log_tail(self, job_id: str, line: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.log_tail.append(line.rstrip("\n"))
            if len(job.log_tail) > MAX_LOG_TAIL_LINES:
                job.log_tail = job.log_tail[-MAX_LOG_TAIL_LINES:]
            self._jobs[job_id] = job

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = datetime.utcnow()
            self._jobs[job_id] = job

        log_path = self._log_path(job_id)
        return_code: int | None = None
        error: str | None = None

        try:
            with open(log_path, "w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    self._jobs[job_id].command,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

                if process.stdout is None:
                    raise RuntimeError("Failed to capture process output.")

                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                    self._append_log_tail(job_id, line)

                return_code = process.wait()
        except Exception as exc:
            error = str(exc)

        with self._lock:
            job = self._jobs[job_id]
            job.finished_at = datetime.utcnow()
            job.return_code = return_code
            job.error = error
            job.status = "succeeded" if (error is None and return_code == 0) else "failed"
            self._jobs[job_id] = job


job_manager = JobManager()

