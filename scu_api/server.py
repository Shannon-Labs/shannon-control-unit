from __future__ import annotations

import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel

from scu_api.config import TrainingConfig
from scu_api.service.job_queue import JobQueueManager
from scu_api.service.smart_config import auto_configure, detect_model_params
from scu_api.service.validator import ConfigValidator

app = FastAPI(title="SCU Training API", version="0.1.0")
jobs = JobQueueManager(db_path=Path("jobs.db"), max_concurrent=1)


class SubmitRequest(BaseModel):
    base_model: str
    train_data: str
    steps: Optional[int] = None
    epochs: int = 1
    target_s: float = 0.01
    kp: float = 0.8
    ki: float = 0.15
    deadband: float = 0.002
    lambda_init: float = 1.0
    lambda_min: float = 1e-4
    lambda_max: float = 2.0
    prior_sigma: float = 0.01
    adapter_out: Optional[str] = None
    batch_size: int = 1
    lr: float = 5e-5
    fp16: bool = True
    gradient_accumulation_steps: int = 4
    block_size: int = 1024
    max_texts: Optional[int] = None
    seed: int = 42
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
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


class AutoConfigRequest(BaseModel):
    model_id: str
    train_data_path: Optional[str] = None


@app.post("/validate")
async def validate_config(payload: SubmitRequest):
    """Validate training configuration before submission."""
    default_lora_modules = TrainingConfig().lora_target_modules

    config = TrainingConfig(
        base_model=payload.base_model,
        train_data=payload.train_data,
        steps=payload.steps,
        epochs=payload.epochs,
        target_s=payload.target_s,
        kp=payload.kp,
        ki=payload.ki,
        deadband=payload.deadband,
        lambda_init=payload.lambda_init,
        lambda_min=payload.lambda_min,
        lambda_max=payload.lambda_max,
        prior_sigma=payload.prior_sigma,
        adapter_out=payload.adapter_out or f"adapters/{payload.base_model.split('/')[-1]}_{int(asyncio.get_running_loop().time())}",
        batch_size=payload.batch_size,
        lr=payload.lr,
        fp16=payload.fp16,
        gradient_accumulation_steps=payload.gradient_accumulation_steps,
        block_size=payload.block_size,
        max_texts=payload.max_texts,
        seed=payload.seed,
        lora_r=payload.lora_r,
        lora_alpha=payload.lora_alpha,
        lora_dropout=payload.lora_dropout,
        lora_target_modules=payload.lora_target_modules or default_lora_modules,
        use_unsloth=payload.use_unsloth,
    )

    validator = ConfigValidator()
    report = validator.validate(config)

    return {
        "valid": report.passed,
        "errors": [{"message": i.message, "suggestion": i.suggestion} for i in report.errors()],
        "warnings": [{"message": i.message, "suggestion": i.suggestion} for i in report.warnings()],
    }


@app.post("/jobs", response_model=SubmitResponse)
async def submit_job(payload: SubmitRequest):
    default_lora_modules = TrainingConfig().lora_target_modules

    cfg = TrainingConfig(
        base_model=payload.base_model,
        train_data=payload.train_data,
        steps=payload.steps,
        epochs=payload.epochs,
        target_s=payload.target_s,
        kp=payload.kp,
        ki=payload.ki,
        deadband=payload.deadband,
        lambda_init=payload.lambda_init,
        lambda_min=payload.lambda_min,
        lambda_max=payload.lambda_max,
        prior_sigma=payload.prior_sigma,
        adapter_out=payload.adapter_out or f"adapters/{payload.base_model.split('/')[-1]}_{asyncio.get_running_loop().time():.0f}",
        batch_size=payload.batch_size,
        lr=payload.lr,
        fp16=payload.fp16,
        gradient_accumulation_steps=payload.gradient_accumulation_steps,
        block_size=payload.block_size,
        max_texts=payload.max_texts,
        seed=payload.seed,
        lora_r=payload.lora_r,
        lora_alpha=payload.lora_alpha,
        lora_dropout=payload.lora_dropout,
        lora_target_modules=payload.lora_target_modules or default_lora_modules,
        use_unsloth=payload.use_unsloth,
    )

    validator = ConfigValidator()
    report = validator.validate(cfg)

    if not report.passed:
        errors = "\n".join([f"ERROR: {i.message}" for i in report.errors()])
        raise HTTPException(status_code=400, detail=f"Validation failed:\n{errors}")

    await jobs.ensure_worker()
    job_id = await jobs.submit(cfg)
    return SubmitResponse(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    state = jobs.get_job(job_id)
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
    return jobs.list_jobs()


@app.post("/auto-config")
async def get_auto_configuration(request: AutoConfigRequest):
    """Get automatically configured parameters for a model."""
    try:
        config = auto_configure(model_id=request.model_id, train_data=request.train_data_path)
        param_count, _ = detect_model_params(request.model_id)
        return {
            "model_id": request.model_id,
            "estimated_params_GB": param_count,
            "suggested_config": config.dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/jobs/{job_id}/adapter")
async def download_adapter(job_id: str):
    state = jobs.get_job(job_id)
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
    return {"status": "ok", "jobs": len(jobs.list_jobs())}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
