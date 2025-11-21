from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport


@pytest_asyncio.fixture
async def api_client(monkeypatch, tmp_path):
    from scu_api import server

    class FakeState:
        def __init__(self, job_id, status="queued", progress=0.0, adapter_path=None, error=None):
            self.job_id = job_id
            self.status = status
            self.progress = progress
            self.adapter_path = adapter_path
            self.error = error

    class FakeJobs:
        def __init__(self):
            self.jobs = {}

        async def ensure_worker(self):
            return None

        async def submit(self, cfg):
            job_id = f"job{len(self.jobs)}"
            self.jobs[job_id] = FakeState(job_id)
            return job_id

        def get_job(self, job_id):
            return self.jobs.get(job_id)

        def list_jobs(self):
            return [
                {
                    "job_id": state.job_id,
                    "status": state.status,
                    "progress": state.progress,
                    "adapter_path": state.adapter_path,
                    "error": state.error,
                }
                for state in self.jobs.values()
            ]

    fake_jobs = FakeJobs()
    monkeypatch.setattr(server, "jobs", fake_jobs)

    transport = ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, fake_jobs


@pytest.mark.asyncio
async def test_validate_and_job_lifecycle(api_client, tmp_path):
    client, fake_jobs = api_client

    data_file = tmp_path / "data.txt"
    data_file.write_text("hello")

    payload = {
        "base_model": "gpt2",
        "train_data": str(data_file),
        "steps": 1,
    }

    res = await client.post("/validate", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["valid"] is True

    submit = await client.post("/jobs", json=payload)
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]

    status = await client.get(f"/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "queued"

    listing = await client.get("/jobs")
    assert listing.status_code == 200
    assert any(job["job_id"] == job_id for job in listing.json())


@pytest.mark.asyncio
async def test_download_adapter(api_client, tmp_path):
    client, fake_jobs = api_client

    adapter_file = tmp_path / "adapter.bin"
    adapter_file.write_text("adapter")

    job_id = "job-download"
    fake_jobs.jobs[job_id] = type("FakeState", (), {
        "job_id": job_id,
        "status": "succeeded",
        "progress": 100.0,
        "adapter_path": str(adapter_file),
        "error": None,
    })()

    resp = await client.get(f"/jobs/{job_id}/adapter")
    assert resp.status_code == 200
    assert resp.content == adapter_file.read_bytes()
