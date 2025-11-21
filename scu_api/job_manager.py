from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from scu_api.config import TrainingConfig
from scu_api.training_engine import TrainingEngine, TrainingMetrics


@dataclass
class JobState:
    job_id: str
    status: str
    progress: float = 0.0
    adapter_path: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_metrics: Optional[TrainingMetrics] = None
    target_steps: Optional[int] = None


class JobManager:
    """In-memory job scheduler suitable for single-node experiments."""

    def __init__(self):
        self.jobs: Dict[str, JobState] = {}
        self.logger = logging.getLogger("scu_api.job_manager")

    def submit(self, config: TrainingConfig) -> str:
        job_id = str(uuid.uuid4())[:8]
        state = JobState(job_id=job_id, status="queued", target_steps=config.steps)
        self.jobs[job_id] = state
        asyncio.create_task(self._run_job(job_id, config))
        return job_id

    def get(self, job_id: str) -> Optional[JobState]:
        return self.jobs.get(job_id)

    def list(self) -> Dict[str, JobState]:
        return self.jobs

    async def _run_job(self, job_id: str, config: TrainingConfig):
        state = self.jobs[job_id]
        state.status = "running"
        state.updated_at = datetime.utcnow()

        engine = TrainingEngine(config=config, job_id=job_id)

        def progress_cb(metrics: TrainingMetrics):
            state.last_metrics = metrics
            if state.target_steps:
                state.progress = min(100.0, (metrics.step + 1) / state.target_steps * 100.0)
            else:
                state.progress = max(state.progress, min(99.0, metrics.step))
            state.updated_at = datetime.utcnow()

        try:
            adapter_dir: Path = await asyncio.to_thread(engine.run, progress_cb)
            state.status = "succeeded"
            state.progress = 100.0
            state.adapter_path = str(adapter_dir)
            state.updated_at = datetime.utcnow()
            self.logger.info("Job %s completed -> %s", job_id, adapter_dir)
        except Exception as exc:  # pragma: no cover - surfaced via state
            state.status = "failed"
            state.error = str(exc)
            state.updated_at = datetime.utcnow()
            self.logger.exception("Job %s failed", job_id)
