import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm


class SCUClient:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip("/")
        self.session = requests.Session()

    def submit_job(self, base_model: str, train_data: str, wait: bool = False, **kwargs) -> Dict:
        payload = {"base_model": base_model, "train_data": train_data, **kwargs}
        response = self.session.post(f"{self.api_url}/jobs", json=payload)
        response.raise_for_status()
        job = response.json()

        if wait:
            return self.wait_for_completion(job["job_id"])
        return job

    def validate_config(self, **payload) -> Dict:
        response = self.session.post(f"{self.api_url}/validate", json=payload)
        response.raise_for_status()
        return response.json()

    def auto_configure(self, model_id: str, train_data: Optional[str] = None) -> Dict:
        response = self.session.post(
            f"{self.api_url}/auto-config",
            json={"model_id": model_id, "train_data_path": train_data},
        )
        response.raise_for_status()
        return response.json()

    def wait_for_completion(self, job_id: str, poll_interval: int = 5) -> Dict:
        with tqdm(total=100, desc=f"Job {job_id}") as pbar:
            while True:
                status = self.get_job_status(job_id)
                if status["status"] == "succeeded":
                    pbar.update(100 - pbar.n)
                    return status
                if status["status"] == "failed":
                    raise RuntimeError(f"Job failed: {status.get('error')}")
                progress = status.get("progress", 0) or 0
                pbar.update(progress - pbar.n)
                time.sleep(poll_interval)

    def get_job_status(self, job_id: str) -> Dict:
        response = self.session.get(f"{self.api_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def list_jobs(self) -> List[Dict]:
        response = self.session.get(f"{self.api_url}/jobs")
        response.raise_for_status()
        return response.json()

    def download_adapter(self, job_id: str, output_path: str) -> str:
        response = self.session.get(f"{self.api_url}/jobs/{job_id}/adapter", stream=True)
        response.raise_for_status()

        output_path = Path(output_path)
        if output_path.is_dir():
            filename = self._extract_filename(response.headers) or f"{job_id}_adapter.bin"
            output_path = output_path / filename

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return str(output_path)

    def health(self) -> Dict:
        response = self.session.get(f"{self.api_url}/health")
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _extract_filename(headers: Dict[str, str]) -> Optional[str]:
        dispo = headers.get("content-disposition") or headers.get("Content-Disposition")
        if not dispo:
            return None
        parts = dispo.split(";")
        for part in parts:
            if "filename=" in part:
                return part.split("=", 1)[1].strip().strip('"')
        return None
