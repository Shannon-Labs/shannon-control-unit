import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from scu_api.config import TrainingConfig
from scu_api.service.job_queue import JobQueueManager


@dataclass
class DummyMetrics:
    step: int
    data_bpt: float
    param_bpt: float
    total_bpt: float
    s_ratio: float
    lambda_value: float
    loss: float
    tokens_per_second: float
    eta_minutes: float


@pytest.mark.asyncio
async def test_job_queue_persists_and_marks_success(tmp_path):
    db_path = tmp_path / "jobs.db"
    queue = JobQueueManager(db_path=db_path, max_concurrent=1)

    cfg = TrainingConfig(base_model="gpt2", train_data="/tmp/data.txt", steps=1)

    async def fake_train(job_id, config, progress_cb):
        progress_cb(
            DummyMetrics(
                step=0,
                data_bpt=0.1,
                param_bpt=0.2,
                total_bpt=0.3,
                s_ratio=0.4,
                lambda_value=1.0,
                loss=0.5,
                tokens_per_second=10.0,
                eta_minutes=0.1,
            )
        )
        return Path("/tmp/adapter")

    queue._train_job = fake_train  # type: ignore[attr-defined]

    jid = await queue.submit(cfg)
    await queue._run_job(jid, cfg)

    job_state = queue.get_job(jid)
    assert job_state is not None
    assert job_state.status == "succeeded"
    assert job_state.progress == 100.0
