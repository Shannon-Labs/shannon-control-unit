import pytest

from scu_api.config import TrainingConfig
from scu_api.service.validator import ConfigValidator


def test_gpu_memory_check(tmp_path):
    config = TrainingConfig(
        base_model="gpt2",
        train_data=str(tmp_path / "test.txt"),
        adapter_out="adapters/test",
        batch_size=100,
    )
    # create dummy dataset file
    (tmp_path / "test.txt").write_text("hello")

    validator = ConfigValidator()
    report = validator.validate(config)
    assert len(report.issues) > 0


def test_missing_dataset():
    config = TrainingConfig(
        base_model="gpt2",
        train_data="/nonexistent/file.txt",
        adapter_out="adapters/test",
    )
    validator = ConfigValidator()
    report = validator.validate(config)
    assert not report.passed
    assert any("not found" in i.message for i in report.errors())


def test_valid_config(tmp_path):
    data_file = tmp_path / "train.txt"
    data_file.write_text("hello world")

    config = TrainingConfig(
        base_model="gpt2",
        train_data=str(data_file),
        adapter_out="adapters/test",
        batch_size=1,
        steps=5,
    )
    validator = ConfigValidator()
    report = validator.validate(config)
    assert report.passed
