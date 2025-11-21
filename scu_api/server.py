from __future__ import annotations

import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel

from scu_api.config import TrainingConfig
from scu_api.job_manager import JobManager

app = FastAPI(title="SCU Training API", version="0.1.0")
jobs = JobManager()


class SubmitRequest(BaseModel):
    base_model: str
    train_data: str
    steps: Optional[int] = None
    target_s: float = 0.01
    kp: float = 0.8
    ki: float = 0.15
    adapter_out: Optional[str] = None
    batch_size: int = 1
    lr: float = 5e-5
    fp16: bool = True
    max_texts: Optional[int] = None
    use_unsloth: bool = False


class SubmitResponse(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    adapter_path: Optional[str] = None
    error: Optional[str] = None


@app.post("/jobs", response_model=SubmitResponse)
async def submit_job(payload: SubmitRequest):
    cfg = TrainingConfig(
        base_model=payload.base_model,
        train_data=payload.train_data,
        steps=payload.steps,
        target_s=payload.target_s,
        kp=payload.kp,
        ki=payload.ki,
        adapter_out=payload.adapter_out or f"adapters/{payload.base_model.split('/')[-1]}_{asyncio.get_running_loop().time():.0f}",
        batch_size=payload.batch_size,
        lr=payload.lr,
        fp16=payload.fp16,
        max_texts=payload.max_texts,
        use_unsloth=payload.use_unsloth,
    )
    job_id = jobs.submit(cfg)
    return SubmitResponse(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    state = jobs.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatus(
        job_id=state.job_id,
        status=state.status,
        progress=state.progress,
        adapter_path=state.adapter_path,
        error=state.error,
    )


@app.get("/jobs")
async def list_jobs():
    return [
        {
            "job_id": s.job_id,
            "status": s.status,
            "progress": s.progress,
            "adapter_path": s.adapter_path,
        }
        for s in jobs.list().values()
    ]


@app.get("/jobs/{job_id}/adapter")
async def download_adapter(job_id: str):
    state = jobs.get(job_id)
    if not state or not state.adapter_path:
        raise HTTPException(status_code=404, detail="adapter not ready")

    adapter_path = Path(state.adapter_path)
    if not adapter_path.exists():
        raise HTTPException(status_code=404, detail="adapter missing from disk")

    if adapter_path.is_dir():
        tmp_dir = Path(tempfile.mkdtemp())
        archive_base = tmp_dir / f"{job_id}_adapter"
        archive_file = shutil.make_archive(str(archive_base), "zip", adapter_path)
        archive_path = Path(archive_file)
        filename = f"{job_id}_adapter.zip"
        return FileResponse(
            str(archive_path),
            filename=filename,
            media_type="application/zip",
            background=BackgroundTask(shutil.rmtree, tmp_dir, True),
        )

    return FileResponse(str(adapter_path), filename=adapter_path.name)


@app.get("/health")
async def health():
    return {"status": "ok", "jobs": len(jobs.list())}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
