import json
from pathlib import Path

from click.testing import CliRunner

from scu_api.cli.main import cli


class DummyClient:
    def __init__(self):
        self.submitted = None

    def submit_job(self, **kwargs):
        self.submitted = kwargs
        return {"job_id": "abc123", "status": "queued"}

    def get_job_status(self, job_id):
        return {"job_id": job_id, "status": "running", "progress": 50}

    def list_jobs(self):
        return [{"job_id": "abc123", "status": "queued", "progress": 0}]

    def download_adapter(self, job_id, output_path):
        Path(output_path).write_text("adapter")
        return str(output_path)

    def auto_configure(self, model_id, train_data=None):
        return {"suggested_config": {"lr": 1e-4, "batch_size": 2}}

    def health(self):
        return {"status": "ok"}


def test_cli_train_and_status(monkeypatch, tmp_path):
    dummy = DummyClient()
    monkeypatch.setattr("scu_api.cli.main.SCUClient", lambda url: dummy)

    runner = CliRunner()
    data_file = tmp_path / "train.txt"
    data_file.write_text("hello")

    res = runner.invoke(
        cli,
        [
            "--api-url",
            "http://example.com",
            "train",
            "--base-model",
            "gpt2",
            "--train-data",
            str(data_file),
            "--steps",
            "5",
        ],
    )
    assert res.exit_code == 0
    assert dummy.submitted["base_model"] == "gpt2"

    res_status = runner.invoke(cli, ["status", "abc123"])
    assert res_status.exit_code == 0
    assert "running" in res_status.output


def test_cli_auto_config_merge(monkeypatch, tmp_path):
    dummy = DummyClient()
    monkeypatch.setattr("scu_api.cli.main.SCUClient", lambda url: dummy)
    runner = CliRunner()

    data_file = tmp_path / "train.txt"
    data_file.write_text("hello")

    res = runner.invoke(
        cli,
        [
            "train",
            "--base-model",
            "gpt2",
            "--train-data",
            str(data_file),
            "--auto-config",
        ],
    )
    assert res.exit_code == 0
    assert dummy.submitted["batch_size"] == 2
    assert dummy.submitted["lr"] == 1e-4


def test_cli_download(monkeypatch, tmp_path):
    dummy = DummyClient()
    monkeypatch.setattr("scu_api.cli.main.SCUClient", lambda url: dummy)
    runner = CliRunner()

    output_file = tmp_path / "out.bin"
    res = runner.invoke(cli, ["download", "abc123", "--output", str(output_file)])
    assert res.exit_code == 0
    assert output_file.exists()


def test_cli_health(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr("scu_api.cli.main.SCUClient", lambda url: dummy)
    runner = CliRunner()
    res = runner.invoke(cli, ["health"])
    assert res.exit_code == 0
    assert "ok" in res.output
