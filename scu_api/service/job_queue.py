import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

import sqlite3

from scu_api.config import TrainingConfig

if TYPE_CHECKING:  # pragma: no cover
    from scu_api.training_engine import TrainingEngine, TrainingMetrics


@dataclass
class JobRecord:
    job_id: str
    status: str
    progress: float
    adapter_path: Optional[str]
    error: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


class JobQueueManager:
    """Async job queue with SQLite persistence."""

    def __init__(self, db_path: Path, max_concurrent: int = 1):
        self.db_path = Path(db_path)
        self.max_concurrent = max_concurrent
        self.queue: asyncio.Queue = asyncio.Queue()
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self._worker_started = False
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT,
                config_json TEXT,
                progress REAL DEFAULT 0.0,
                adapter_path TEXT,
                error TEXT,
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    async def ensure_worker(self):
        if self._worker_started:
            return
        self._worker_started = True
        asyncio.create_task(self._worker_loop(), name="scu_job_worker")

    async def submit(self, config: TrainingConfig) -> str:
        import uuid

        job_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO jobs(job_id, status, config_json, progress, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            (job_id, "queued", json.dumps(config.model_dump()), 0.0, now),
        )
        conn.commit()
        conn.close()

        await self.queue.put((job_id, config))
        return job_id

    async def _worker_loop(self):
        while True:
            if len(self.active_jobs) >= self.max_concurrent:
                await asyncio.sleep(0.5)
                continue

            job_id, config = await self.queue.get()
            task = asyncio.create_task(self._run_job(job_id, config), name=f"job_{job_id}")
            self.active_jobs[job_id] = task

            # Cleanup finished jobs
            to_delete = [jid for jid, t in self.active_jobs.items() if t.done()]
            for jid in to_delete:
                self.active_jobs.pop(jid, None)

    async def _run_job(self, job_id: str, config: TrainingConfig):
        self._update_status(job_id, status="running", started_at=datetime.utcnow().isoformat())

        def progress_cb(metrics):
            target_steps = config.target_steps or metrics.step + 1
            progress = min(100.0, ((metrics.step + 1) / max(target_steps, 1)) * 100.0)
            self._update_status(job_id, progress=progress)

        try:
            artifact_path = await asyncio.to_thread(self._train_job, job_id, config, progress_cb)
            self._update_status(
                job_id,
                status="succeeded",
                progress=100.0,
                adapter_path=str(artifact_path),
                completed_at=datetime.utcnow().isoformat(),
            )
        except Exception as exc:  # pragma: no cover
            self._update_status(
                job_id,
                status="failed",
                error=str(exc),
                completed_at=datetime.utcnow().isoformat(),
            )

    def _train_job(self, job_id: str, config: TrainingConfig, progress_cb):
        from scu_api.training_engine import TrainingEngine

        engine = TrainingEngine(config=config, job_id=job_id)
        return engine.run(progress_callback=progress_cb)

    def _update_status(self, job_id: str, **fields):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        assignments = []
        values = []
        for key, val in fields.items():
            assignments.append(f"{key} = ?")
            values.append(val)
        values.append(job_id)
        cur.execute(f"UPDATE jobs SET {', '.join(assignments)} WHERE job_id = ?", values)
        conn.commit()
        conn.close()

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT job_id, status, progress, adapter_path, error, created_at, started_at, completed_at FROM jobs WHERE job_id = ?", (job_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return JobRecord(
            job_id=row[0],
            status=row[1],
            progress=float(row[2] or 0.0),
            adapter_path=row[3],
            error=row[4],
            created_at=datetime.fromisoformat(row[5]) if row[5] else datetime.utcnow(),
            started_at=datetime.fromisoformat(row[6]) if row[6] else None,
            completed_at=datetime.fromisoformat(row[7]) if row[7] else None,
        )

    def list_jobs(self) -> list:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT job_id, status, progress, adapter_path, created_at FROM jobs ORDER BY created_at DESC")
        rows = cur.fetchall()
        conn.close()
        return [
            {
                "job_id": r[0],
                "status": r[1],
                "progress": float(r[2] or 0.0),
                "adapter_path": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    def cancel(self, job_id: str) -> bool:
        task = self.active_jobs.get(job_id)
        if task:
            task.cancel()
            self._update_status(job_id, status="cancelled", completed_at=datetime.utcnow().isoformat())
            return True
        return False
